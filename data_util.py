import numpy as np
import pandas as pd
import json

class MedQA_dataset:
    def __init__(self, dataset_name, dataset_path):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path

    def test_data(self):
        test_path = f'{self.dataset_path}/test.json'
        print(f'Loading test data from {test_path}...')
        self.dataset = load_dataset(self.dataset_name, test_path)
        # print(self.dataset[0])
        all_data = []
        for i, d in enumerate(self.dataset):
            question = d['question']
            options = d['options']
            answer_idx = d['answer_idx']
            all_data.append({
                'question': question,
                'options': options,
                'answer_idx': answer_idx,
                'question_id': i
            })
        return all_data


class MedMCQA_dataset:
    def __init__(self, dataset_name, dataset_path):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path

    def test_data(self):
        test_path = f'{self.dataset_path}/benchmark.json'
        print(f'Loading test data from {test_path}...')
        self.dataset = load_dataset(self.dataset_name, test_path)
        all_data = []
        for i, d in enumerate(self.dataset):
            question = d['question']
            options = d['options']
            answer_idx = d['answer_idx']
            all_data.append({
                'question': question,
                'options': options,
                'answer_idx': answer_idx,
                'question_id': i
            })
        return all_data

class dataset:
    def __init__(self, dataset_name, dataset_path):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path

    def test_data(self):
        test_path = f'{self.dataset_path}/benchmark.json'
        print(f'Loading test data from {test_path}...')
        all_data = load_all_benchmark(self.dataset_name, test_path)
        return all_data


def load_dataset(dataset_name, dataset_path):
    if dataset_name.lower() == 'medqa':
        all = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                all.append(json.loads(line))
            return all
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

def load_all_benchmark(dataset_name, dataset_path):
    all_data = json.load(open(dataset_path, 'r', encoding='utf-8'))
    # print(all_data)
    dataset_name = dataset_name.lower()
    print(dataset_name)
    roi_data = []
    dataset_names = dataset_name.split('-')
    if len(dataset_names) > 1:
        dataset_name = dataset_names[0]
        subject = dataset_names[1]
    else:
        dataset_name = dataset_names[0]
        subject = None
    print(dataset_name, subject)
    for k, v in all_data.items():
        if k == dataset_name.lower():
            medqa_data = all_data[k]
            if subject:
                for k1, v1 in medqa_data.items():
                    if subject in k1.lower():
                        roi_data.append({
                            'question': v1['question'],
                            'options': v1['options'],
                            'answer_idx': v1['answer'],
                            'question_id': k1
                        })
                    else:
                        continue
            else:
                for k1, v1 in medqa_data.items():
                    roi_data.append({
                        'question': v1['question'],
                        'options': v1['options'],
                        'answer_idx': v1['answer'],
                        'question_id': k1
                        }
                    )
            
        else:
            continue
        return roi_data
# roi_data = load_all_benchmark('mmlu_medical_genetics', '/l/users/jason.xue/xy_project/med_edge/dataset/OptionQA/benchmark.json')
# print(len(roi_data))
