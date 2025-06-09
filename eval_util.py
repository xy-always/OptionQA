from sklearn.metrics import accuracy_score
import numpy as np
import openai
import json
import re
# -*- coding: utf-8 -*-

def locate_answer(sentence:str):

    ans = re.findall("^\s*(A|B|C|D)$", sentence)
    if len(ans) > 0:
        return ans[0].upper()
    
    ans = re.findall("^\s*(A|B|C|D) or", sentence)
    if len(ans) > 0:
        return ans[0].upper()
    
    ans = re.findall("^\s*(A|B|C|D) and", sentence)
    if len(ans) > 0:
        return ans[0].upper()
        
    ans = re.findall("^\s*(A|B|C|D)/", sentence)
    if len(ans) > 0:
        return ans[0].upper()
    
    ans = re.findall("^\s*(A|B|C|D),", sentence)
    if len(ans) > 0:
        return ans[0].upper()
    
    ans = re.findall("[Oo]ption (A|B|C|D)", sentence)
    if len(ans) > 0:
        return ans[0]

    ans = re.findall(":\s*(A|B|C|D)", sentence)
    if len(ans) > 0:
        return ans[0].upper()

    ans = re.findall("^\s*(A|B|C|D)\.", sentence)
    if len(ans) > 0:
        return ans[0].upper()

    ans = re.findall("^\s*(A|B|C|D)\"", sentence)
    if len(ans) > 0:
        return ans[0].upper()
    
    ans = re.findall("^\s*(A|B|C|D):", sentence)
    if len(ans) > 0:
        return ans[0].upper()

    return "A"

def locate_answer4pub_llama(sentence:str):

    sentence = sentence.split("Answer:")[-1]

    ans = re.findall("[Oo]ption (A|B|C|D)", sentence)
    if len(ans) > 0:
        return ans[0]

    ans = re.findall("OPTION (A|B|C|D)", sentence)
    if len(ans) > 0:
        return ans[0]

    ans = re.findall("^\s*(A|B|C|D)\"", sentence)
    if len(ans) > 0:
        return ans[0].upper()
    
    ans = re.findall("^\s*(A|B|C|D):", sentence)
    if len(ans) > 0:
        return ans[0].upper()    

    return "A"

def accuracy(results):
    preds = []
    labels = []
    for r in results:
        g_ans = r['generator_response_str']
        options = r['options']
        find = False
        preds.append(locate_answer(g_ans))
        # for k, v in options.items():
        #     if "Answer:" in g_ans:
        #         g_ans = g_ans.split("Answer:")[-1].strip()
            
        #     if len(g_ans.split()) <= 2:
        #         if k.lower() in g_ans.lower():
        #             preds.append(k)
        #             find = True
        #             break
        #     elif v.lower() in g_ans.lower():
        #         preds.append(k)
        #         find = True
        #         break
        # if not find:
        #     preds.append('A')
        labels.append(r['gold_answer'])
    if len(preds) != len(labels):
        raise ValueError("The number of predictions and labels must be the same.")
    return accuracy_score(labels, preds)


class EvaluatorModel:
    def __init__(self, model, n=11):
        self.model = model
        self.messages = []
        self.n = n
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        # self.client = openai.OpenAI(api_key=openai.api_key)
        self.client = openai.OpenAI()

    def run(self, generator_response_str, criteria_list):
        fail_rate = 1.0
        original_criteria_list = criteria_list.copy()
        while fail_rate > 0.5:
            criteria_json = json.dumps(criteria_list)
            evaluator_prompt = f'Given the criteria below, return a list of True or False for each criteria, depending whether the whole text below meet this criteria or not. Do not evaluate each bullet point of the answer separately. Do not justify your decision, just output True or False.\n Criteria: {criteria_json}\nText: {generator_response_str}'
            self.messages = [{"role": "user", "content": evaluator_prompt}]

            response = openai.chat.completions.create(
                model=self.model,
                messages=self.messages,
                n=self.n
            )

            self.prompt_tokens += response.usage.prompt_tokens
            self.completion_tokens += response.usage.completion_tokens
            self.total_tokens += response.usage.total_tokens
            
            boolean_list = [self.extract_booleans(choice.message.content) for choice in response.choices if len(self.extract_booleans(choice.message.content)) == len(criteria_list)]

            if not boolean_list:  # No valid evaluations
                # print('No valid evaluations! Repeat...')
                continue

            res_matrix = np.array(boolean_list)
            valid_evals_len = res_matrix.shape[0]
            fail_rate = self.calculate_fail_rate(valid_evals_len)

            if fail_rate > 0.5 and len(criteria_list) > 1:
                # Split criteria list in half and evaluate each part
                # print(f'fail_rate > 0.5! Split criteria list in half and evaluate each part.')
                mid_index = len(criteria_list) // 2
                res_boolean_list_1, res_mean_1, res_confidence_rate_1, fail_rate_1 = self.run(generator_response_str, criteria_list[:mid_index])
                res_boolean_list_2, res_mean_2, res_confidence_rate_2, fail_rate_2 = self.run(generator_response_str, criteria_list[mid_index:])

                # Combine results from both halves
                res_boolean_list = np.concatenate([res_boolean_list_1, res_boolean_list_2])
                res_mean = np.concatenate([res_mean_1, res_mean_2])
                res_confidence_rate = (res_confidence_rate_1 + res_confidence_rate_2) / 2
                fail_rate = (fail_rate_1 + fail_rate_2) / 2

                return res_boolean_list.tolist(), res_mean.tolist(), res_confidence_rate, np.round(fail_rate, 2)

            if fail_rate <= 0.5:
                # print('fail_rate <= 0.5, calculating final results...')
                res_boolean_list, res_mean, res_confidence_rate = self.calculate_major_vote(res_matrix)

                return res_boolean_list.tolist(), res_mean.tolist(), res_confidence_rate, np.round(fail_rate, 2)

        print('Unexpected failure!')
        return [np.nan]*len(criteria_list), [np.nan]*len(criteria_list), np.nan, np.nan  # In case of unexpected failure

    def extract_booleans(self, s):
        return [bool(re.match('true', i)) for i in re.findall(r'true|false', s.lower())]

    def calculate_major_vote(self, res_matrix):
        res_mean = np.nansum(res_matrix, axis=0) / res_matrix.shape[0]
        major_vote = (res_mean >= 0.5)
        confidence_rate = self.calculate_confidence_rate(res_mean)
        return major_vote, res_mean, np.round(confidence_rate, 4)

    def calculate_fail_rate(self, valid_evals_len):
        return 1 - valid_evals_len / self.n

    def calculate_confidence_rate(self, res_mean):
        return 1 - 2 * sum(abs(np.round(res_mean) - res_mean)) / len(res_mean)
