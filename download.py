from datasets import load_dataset

# download MedQA-USMLE-4-options dataset
# and save it in the specified directory
def download_dataset(dataset_name, save_dir, has_train=True, has_test=True, has_validation=False):
    """
    Downloads the MedQA-USMLE-4-options dataset and saves it in the specified directory.
    """
    print("Downloading {dataset_name} dataset...") 
    dataset = load_dataset(dataset_name)
    if has_train:
        dataset['train'].to_json(f'{save_dir}/train.json')
    if has_test:
        dataset['test'].to_json(f'{save_dir}/test.json')
    if has_validation:
        dataset['validation'].to_json(f'{save_dir}/validation.json')
    print("Dataset downloaded successfully.")

def download_mmlu_clinical(dataset_name, subject, save_dir, splits):
    """
    Downloads the MMLU Clinical dataset and saves it in the specified directory.
    """
    print("Downloading MMLU Clinical dataset...")
    dataset = load_dataset(dataset_name, subject)
    print(dataset)
    for s in splits:
        dataset[s].to_json(f'{save_dir}/{s}.json')
    print("MMLU Clinical dataset downloaded successfully.")

# download MedQA-USMLE-4-options dataset
# download_dataset('GBaker/MedQA-USMLE-4-options', '/l/users/jason.xue/xy_project/med_edge/dataset/OptionQA/MedQA')

# MedMCQA dataset: https://medmcqa.github.io/

# PubMedQA dataset: https://github.com/pubmedqa/pubmedqa

# LiveQA: https://github.com/abachaa/LiveQA_MedicalTask_TREC2017

# download MMLU Clinical dataset: clinical_knowledge, medical_genetics, anatomy, professional_medicine, college_biology, college_chemistry, college_medicine
download_mmlu_clinical('cais/mmlu', 'clinical_knowledge', '/l/users/jason.xue/xy_project/med_edge/dataset/OptionQA/MMLU/clinical_knowledge', splits=['dev', 'test', 'validation'])
download_mmlu_clinical('cais/mmlu', 'medical_genetics', '/l/users/jason.xue/xy_project/med_edge/dataset/OptionQA/MMLU/medical_genetics', splits=['dev', 'test', 'validation'])
download_mmlu_clinical('cais/mmlu', 'anatomy', '/l/users/jason.xue/xy_project/med_edge/dataset/OptionQA/MMLU/anatomy', splits=['dev', 'test', 'validation'])
download_mmlu_clinical('cais/mmlu', 'professional_medicine', '/l/users/jason.xue/xy_project/med_edge/dataset/OptionQA/MMLU/professional_medicine', splits=['dev', 'test', 'validation'])
download_mmlu_clinical('cais/mmlu', 'college_biology', '/l/users/jason.xue/xy_project/med_edge/dataset/OptionQA/MMLU/college_biology', splits=['dev', 'test', 'validation'])
download_mmlu_clinical('cais/mmlu', 'college_chemistry', '/l/users/jason.xue/xy_project/med_edge/dataset/OptionQA/MMLU/college_chemistry', splits=['dev', 'test', 'validation'])
download_mmlu_clinical('cais/mmlu', 'college_medicine', '/l/users/jason.xue/xy_project/med_edge/dataset/OptionQA/MMLU/college_medicine', splits=['dev', 'test', 'validation'])