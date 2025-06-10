import json
def load_dataset(dataset_name, dataset_path):
    if dataset_name.lower() == 'medqa':
        all = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                all.append(json.loads(line))
            return all
    elif dataset_name.lower() == 'icliniq':
        all = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                all.append({
                    'question': item['input'],
                    'answer_icliniq': item['answer_icliniq'],
                })
            return all
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
class icliniq:
    def __init__(self, dataset_name, dataset_path):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path

    def test_data(self):
        test_path = f'{self.dataset_path}/iCliniq_2000.json'
        print(f'Loading test data from {test_path}...')
        self.dataset = load_dataset(self.dataset_name, test_path)
        # print(self.dataset[0])
        all_data = []
        for i, d in enumerate(self.dataset):
            question = d['question']
            answer_icliniq = d['answer_icliniq']
            all_data.append({
                'question': question,
                'answer_icliniq': answer_icliniq,
                'question_id': i
            })
        return all_data
