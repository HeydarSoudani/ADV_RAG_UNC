import json
import random
from tqdm import tqdm
from datasets import Dataset

from data import examplers

BASE_DIR = '/home/hsoudani/ADV_RAG_UNC'


class BaseDataset:
    def __init__(self, dataset_name: str, split: str, fraction_of_data_to_use: float = 1.0):
        dataset_file = f"{BASE_DIR}/data/processed_files/{dataset_name}_{split}.jsonl"
        try:
            self.examplers = getattr(examplers, f'{dataset_name}_exps')
        except AttributeError:
            raise ValueError(f"The dataset '{dataset_name}' does not exist in the 'examplers' module.")

        data = []
        with open(dataset_file, 'r', encoding='utf-8') as file:
            for line in tqdm(file, desc="Converting dataset ..."):
                item = json.loads(line.strip())
                
                # === 
                data.append({
                    "qid": item["id"],
                    "question": item["question"],
                    "ground_truths": item["answers"],
                    "positive_ctxs": item['positive_ctxs'],
                    "negative_ctxs": item['negative_ctxs'],
                })
        
        if fraction_of_data_to_use < 1.0:
            random.shuffle(data)
            subset_length = int(len(data) * fraction_of_data_to_use)
            self.dataset = data[:subset_length]
        else:
            self.dataset = data




def get_dataset(dataset_name, split, fraction_of_data_to_use):
    dataset_file = f"{BASE_DIR}/datasets/processed_files/{dataset_name}_{split}.jsonl"
    
    data = []
    with open(dataset_file, 'r', encoding='utf-8') as file:
        for line in tqdm(file, desc="Converting dataset ..."):
            item = json.loads(line.strip())
            
            # === 
            data.append({
                "qid": item["id"],
                "question": item["question"],
                "ground_truths": item["answers"],
                "positive_ctxs": item['positive_ctxs'],
                "negative_ctxs": item['negative_ctxs'],
            })
    
    if fraction_of_data_to_use < 1.0:
        random.shuffle(data)
        subset_length = int(len(data) * fraction_of_data_to_use)
        test_data = data[:subset_length]
    else:
        test_data = data

    return test_data
   
 