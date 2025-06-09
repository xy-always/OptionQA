import os
import json
import re
import argparse

from dotenv import load_dotenv

import numpy as np
import pandas as pd
from tabulate import tabulate

import openai
from openai import OpenAI
import tiktoken

from litellm import completion
from litellm.exceptions import BadRequestError
from together import Together
from data_util import MedQA_dataset, dataset
from eval_util import accuracy
import torch
from transformers import pipeline, set_seed, AutoConfig
from transformers import Conversation
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
torch.cuda.empty_cache()

# device = torch.device('cuda:0')

load_dotenv()  # take environment variables from .env.

# client = OpenAI(openai.api_key)
# models = client.models.list()
# print(models)
# openai.default_headers = {"x-foo": "true"}

## set ENV variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
os.environ["TOGETHERAI_API_KEY"] = os.getenv("TOGETHERAI_API_KEY")
# os.environ["TOGETHERAI_API_KEY"] = "bdf514f5a3a3c6dcdb0015ce540f2ae91d173a09959f2f0cb40ea1def6f4dac9"

# step 1：初始化generate模型，openai需要system，其他框架或者本地模型只需要导入模型

class GeneratorModel:
    def __init__(self, model_name_or_path, generator_type, device, max_context_window, system_prompt=None):
        self.model_name_or_path = model_name_or_path
        self.generator_type = generator_type
        self.system_prompt = system_prompt
        self.messages = []
        self.generated_responses = []

        self.seed = 42
        self.temperature = 0.1
        self.max_new_tokens = 2000
        self.max_context_window = max_context_window

        self.device = device

        if self.generator_type == 'openai':
            self.client = OpenAI(api_key=openai.api_key)
            if model_name_or_path == "gpt-3.5-turbo-0125": self.max_context_window = 16385
            if model_name_or_path == "gpt-3.5-turbo-1106": self.max_context_window = 16385
            if model_name_or_path == "gpt-3.5-turbo-0613": self.max_context_window = 4096
            if model_name_or_path == "gpt-3.5-turbo-16k-0613": self.max_context_window = 16385

            if model_name_or_path == "gpt-4-turbo-2024-04-09": self.max_context_window = 128000
            if model_name_or_path == "gpt-4-0125-preview": self.max_context_window = 128000
            if model_name_or_path == "gpt-4-1106-preview": self.max_context_window = 128000
            if model_name_or_path == "gpt-4-0613": self.max_context_window = 8192
            if model_name_or_path == "gpt-4-32k-0613": self.max_context_window = 32768

            self.add_system_prompt(self.system_prompt)

        if self.generator_type == 'litellm':
            self.client = Together(api_key=os.getenv("TOGETHERAI_API_KEY"))

        if self.generator_type == 'local':
            set_seed(self.seed)
            self.add_system_prompt(self.system_prompt)
            
           
            # for param in self.model.param
            if 'Llama-3.1' in model_name_or_path:
                self.conversational_pipeline = pipeline(
                "text-generation",
                model=model_name_or_path,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )
            else:
                self.model_config = AutoConfig.from_pretrained(model_name_or_path)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
                
                # with init_empty_weights():
                #     model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
                # num_layers = len(model.model.layers)
                # layers_per_gpu = num_layers // 4  # 这里应该是 96 层 / 4 = 24
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    torch_dtype=torch.bfloat16,        # FP16 降显存
                    device_map='auto',               # 自动探测并分配到所有 GPU
                    low_cpu_mem_usage=True           # 减少 CPU 内存峰值
                )
                self.conversation = Conversation()
                self.conversational_pipeline = pipeline('conversational',
                                        model=self.model,
                                        temperature=self.temperature,
                                        max_new_tokens=self.max_new_tokens,
                                        tokenizer=self.tokenizer,
                                        )
                # self.tokenizer = self.conversational_pipeline.tokenizer
            self.max_context_window = self.model_config.max_position_embeddings
        
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0


    def run(self, question_str, options):
        # generator_prompt = 'You are a doctor, please answer the medical questions based on the given options. Do not output any other words except option words such as A, B, C, D.\n'
        generator_prompt = 'Given the question, you must choose one of the options as your answer.\n'
        
        generator_prompt += f'Question: {question_str} \n'
        generator_prompt += f'{options}\n'
        print('generator_prompt:', generator_prompt)
        self.add_user_input(generator_prompt)

        if self.generator_type == 'openai':
            if not openai.api_key:
                raise ValueError("API key is required for OpenAI model")
            
            response = self.client.chat.completions.create(
                model=self.model_name_or_path,
                messages = self.messages,
                seed = self.seed,
                temperature = self.temperature
                )

            self.prompt_tokens += response.usage.prompt_tokens
            self.completion_tokens += response.usage.completion_tokens
            self.total_tokens += response.usage.total_tokens

            output_str = response.choices[0].message.content

        elif self.generator_type == 'litellm':

            def get_reponse():
                if self.model_name_or_path == 'together_ai/databricks/dbrx-instruct':

                    response = self.client.chat.completions.create(
                        model="databricks/dbrx-instruct",
                        messages=self.messages
                    )

                else:
                    # response = completion(model=self.model_name_or_path,
                    #                     messages=self.messages,
                    #                     temperature=self.temperature,
                    #                     max_tokens=self.max_new_tokens
                    #                     )
                    response = self.client.completions.create(
                            model=self.model_name_or_path,
                            prompt=generator_prompt,
                            temperature=self.temperature,
                            max_tokens=self.max_new_tokens
                        )
                                            
                return response

            try:
                response = get_reponse()
            except BadRequestError as e:
                error_message = str(e)
                if 'Input validation error' in error_message:
                    self.reset_messages()
                    print('Prompt + conversation history > maximal context length of the model --> memory resetted.')
                    # Resend the query
                    response = get_reponse()

            # output_str = response.choices[0].message.content
            output_str = response.choices[0].text
            
        elif self.generator_type == 'local':

            output_str = self.generate_with_local_model(generator_prompt)

            output_str = self.truncate_local_output(output_str)

            self.prompt_tokens += self.count_tokens(generator_prompt) + self.total_tokens
            self.completion_tokens += self.count_tokens(output_str)
            self.total_tokens = self.prompt_tokens + self.completion_tokens
            # revise: 每个question之后清空memory，不然会出错
            self.reset_memory()

        elif self.generator_type == 'mobile':
            # print('mobile prompt:', generator_prompt)
            # generator_prompt = self.system_prompt + '\n' + generator_prompt + '\n'
            # generator_prompt = '<s> [INST] <<SYS> \n' + {self.system_prompt} + '\n'+ ' <</SYS>> \n' + {generator_prompt} +'[/INST]\n'
            if 'llama-2' in self.model_name_or_path or 'Medllama' in self.model_name_or_path or 'meditron' in self.model_name_or_path:
                generator_prompt = f'<s> [INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n{generator_prompt} [/INST]\n'
            elif 'Qwen2.5' in self.model_name_or_path:
                generator_prompt = f'<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n{generator_prompt}<|im_end|>\n<|im_start|>assistant\n\n'
            elif 'Llama-3.1' in self.model_name_or_path or 'Med42' in self.model_name_or_path:
                generator_prompt = f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{self.system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{generator_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
            elif 'Mixtral' in self.model_name_or_path:
                generator_prompt = f'<s> [INST] {self.system_prompt}\n\n{generator_prompt} [/INST]'
            elif 'DeepSeek' in self.model_name_or_path:
                generator_prompt = f'<｜begin▁of▁sentence｜>{self.system_prompt}<｜User｜>{generator_prompt}<｜Assistant｜><think>\n'
            else:
                generator_prompt = self.system_prompt + '\n' + generator_prompt + '\n'
            # print('mobile prompt:', generator_prompt)
            command = f'/mnt/disk4/xy/MobileLLM/llama-cli -m \"{self.model_name_or_path}\" -p \"{generator_prompt}\" -n {self.max_new_tokens} -c 4096 --temp 0.1 -t 40 --skip-system-prompt' 
            print('command:', command)
            os.system(command)
            output_str = os.popen(command).read()

        else:
            raise ValueError("Invalid model type specified")

        self.add_assistant_response(output_str)

        return output_str

    def generate_with_local_model(self, prompt):
        # Create a conversational pipeline using a specified model
        self.conversation.add_user_input(prompt)
        if 'Llama-3.1' in self.model_name_or_path:
            response = self.conversational_pipeline(
                self.messages,
                do_sample=False,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
            )
            res = response[0]["generated_text"][-1]
            print('response:', res)

        else:
        # Generate a response using the conversational pipeline
            response = self.conversational_pipeline(self.conversation,
                                                    do_sample=False,
                                                    temperature=self.temperature,
                                                    max_new_tokens=self.max_new_tokens)

        # Extract the generated response
            res = response.generated_responses[-1]
            print('response:', res)
        return res

    def add_system_prompt(self, prompt):
        if prompt:
            self.messages.append({"role": "system", "content": prompt})
            return True
        else:
            return False

    def add_user_input(self, prompt):
        self.messages.append({"role": "user", "content": prompt})
        return True

    def add_assistant_response(self, prompt):
        # self.messages.append({"role": "assistant", "content": prompt})
        self.generated_responses.append(prompt)
        return True

    def reset_memory(self):
        self.messages = []
        self.generated_responses = []

        if self.generator_type == 'local':
            self.conversation.messages = []
        
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def reset_messages(self):
        self.messages = []
        if self.generator_type == 'local':
            self.conversation.messages = []
            self.add_system_prompt(self.system_prompt)

    def count_tokens(self, text):
        if self.generator_type == 'openai':
            try:
                encoding = tiktoken.encoding_for_model(self.model_name_or_path)
            except KeyError:
                print("Warning: model not found. Using cl100k_base encoding.")
                encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))

        if self.generator_type == 'local':
            return(len(self.tokenizer.tokenize(text)))
        
        if self.generator_type == 'litellm':
            return(len(self.tokenizer.tokenize(text)))
        
    def num_tokens_from_messages(self):
        """Return the number of tokens used by a list of messages."""

        tokens_per_message = 3
        tokens_per_name = 1
        
        num_tokens = 0
        for message in self.messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += self.count_tokens(value)
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def truncate_local_output(self, text):
        truncated_text = text  # Initialize with the full response

        stopping_phrases = ['### Inst',
                            '### inst',
                            '[INST',
                            '[inst',
                            '[/INST',
                            '[/inst']

        for stopping_phrase in stopping_phrases:
            if stopping_phrase in text:
                stop_index = text.find(stopping_phrase)
                truncated_text = text[:stop_index]
                break

        return truncated_text

class Benchmark:
    def __init__(self, data_dir, dataname='medqa'):
        self.data_dir = data_dir
        # if dataname == 'medqa':
        #     self.data = MedQA_dataset(dataname, data_dir)
        #     self.test_data = self.data.test_data()
        # elif dataname == 'medmcqa':
        self.test_data = dataset(dataname, data_dir).test_data()

    def generate_responses_for_question(self, generator_model, question_str, options, answer_idx):
        generator_response_str = generator_model.run(question_str, options)
        print('generator_response_str:', generator_response_str)
        response = {
            'question_str': question_str,
            'options': options,
            'generator_response_str': generator_response_str,
            'gold_answer': answer_idx # Placeholder for the correct answer
        }
        return response

    def generate_responses(self, generator_model):
        response_dict_list = []
        for i, d in enumerate(self.test_data):
            question_str = d['question']
            options = d['options']
            question_id = d['question_id']
            answer_idx = d['answer_idx']
            print(f'Question {i} / {len(self.test_data)} completed')
            response = self.generate_responses_for_question(generator_model, question_str, options, answer_idx)
            response_dict_list.append(response)

        return response_dict_list

def load_from_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    else:
        return None  # or raise an exception if the file is expected to exist

class Experiment:
    def __init__(self, data_dir, generator_model_name_or_path, generator_type, evaluator_model_name_or_path, device, max_context_window, dataset_name, generator_model=None):

        self.data_dir = data_dir
        
        self.generator_model_name_or_path = generator_model_name_or_path
        self.generator_type = generator_type
        self.evaluator_model_name_or_path = evaluator_model_name_or_path
        self.max_context_window = max_context_window
        self.device = device
        self.dataset_name = dataset_name
        self.benchmark = Benchmark(self.data_dir, self.dataset_name)

        if generator_model == None:
            system_prompt = 'You are a medical expert. Please answer the medical questions based on the given options. Do not output any other words except option words such as A, B, C, D.\n'
            self.generator_model = GeneratorModel(model_name_or_path=self.generator_model_name_or_path, generator_type=self.generator_type, device=self.device, max_context_window=self.max_context_window, system_prompt=system_prompt)
        else:
            self.generator_model = generator_model
            self.generator_model.reset_memory()

        # self.evaluator_model = EvaluatorModel(model=self.evaluator_model_name_or_path)


    def generate(self):
        self.response = self.benchmark.generate_responses(self.generator_model)
        print('response:', self.response)
        return(True)

    def evaluate(self):
        self.evaluation_results = accuracy(self.response)
        print('evaluation_results:', self.evaluation_results)
        return(True)

    # def aggregate(self):
    #     self.aggregated_results = self.aggregate_results()
    #     return(True)

    def run(self):
        # Step 1: Generate Responses
        # This step involves generating responses for each case and question,
        # including handling reask scenarios if needed.
        print('Step 1: Generate')
        self.generate()

        # Step 2: Evaluate Responses
        # Here, the generated responses are evaluated against the criteria.
        # This includes evaluating both the initial responses and any reask responses.
        print('Step 2: Evaluate')
        # model_name = self.generator_model_name_or_path.split('/')[-1]
        # self.response = load_from_json('/l/users/jason.xue/xy_project/med_edge/dataset/OptionQA/results/medqa/Llama3-Med42-8B_responses_tmp01.json')
        print('model_name:', self.generator_model_name_or_path)
        self.evaluate()

        # Step 3: Aggregate Results
        # Aggregating results for comprehensive analysis. This could involve
        # summarizing scores, calculating averages, or other relevant metrics.

        return self.response, self.evaluation_results

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run benchmark experiment")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory to load the data')
    parser.add_argument('--generator_model_name_or_path', type=str, required=True,
                        help='Path or name of the generator model')
    parser.add_argument('--generator_type', type=str, required=True,
                        help='Generator type ("openai" or "local")')
    parser.add_argument('--evaluator_model_name_or_path', type=str, required=True,
                        help='Path or name of the evaluator model')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the output')
    parser.add_argument('--device', type=int, required=True,
                        help='GPU device ID to use (0 for first GPU, etc.)')
    parser.add_argument('--max_context_window', type=int, default=4096,
                        help='Maximum context window size for the model')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Directory to save the output')

    # Parse arguments
    args = parser.parse_args()

    data_dir=args.data_dir
    output_dir=args.output_dir

    generator_model_name_or_path = args.generator_model_name_or_path
    generator_type = args.generator_type
    generator_model_name = generator_model_name_or_path.lower().split('/')[-1]

    evaluator_model_name_or_path = args.evaluator_model_name_or_path
    device = args.device
    max_context_window = args.max_context_window
    dataset_name = args.dataset_name
    parameter_list = []
    parameter_list.append(['Data Directory', data_dir])
    parameter_list.append(['Output Directory', output_dir])
    parameter_list.append(['Generator Model', generator_model_name])
    parameter_list.append(['Generator Type', generator_type])
    parameter_list.append(['Evaluator Model', evaluator_model_name_or_path])
    parameter_list.append(['Device', args.device])
    parameter_list.append(['Max Context Window', args.max_context_window])
    parameter_list.append(['dataset_name', args.dataset_name])

    # Define column names
    column_names = ['Parameter', 'Value']

    # Convert list of lists to DataFrame
    parameter_df = pd.DataFrame(parameter_list, columns=column_names)

    # Print the DataFrame using tabulate
    print(tabulate(parameter_df, tablefmt='pipe', headers='keys', showindex=False), '\n')

 
    torch.cuda.empty_cache()

    experiment = Experiment(data_dir=data_dir,
                            generator_model_name_or_path=generator_model_name_or_path,
                            generator_type=generator_type,
                            evaluator_model_name_or_path=evaluator_model_name_or_path,
                            device=device,
                            max_context_window=max_context_window,
                            dataset_name=dataset_name)
    response, evaluation_results = experiment.run()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_name = generator_model_name_or_path.split('/')[-1]
    with open(os.path.join(output_dir, model_name + '_responses.json'), 'w') as f:
        json.dump(response, f, indent=4)

    print('evaluation_results:', evaluation_results)
        
if __name__ == "__main__":
    main()