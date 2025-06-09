import os
import json
from eval_util import accuracy, EvaluatorModel
# read result json file
def load_from_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    else:
        raise FileNotFoundError(f"The file {file_path} does not exist.")

# evaluate Medqa performance 
response = load_from_json('/home/xy/med_edge/dataset/OptionQA_mobile/results/pubmedqa/ggml-model-Qwen2.5-Aloe-Beta-7B-q4_0.gguf_responses.json')
acc = accuracy(response)
print(f'MedQA accuracy: {acc}')