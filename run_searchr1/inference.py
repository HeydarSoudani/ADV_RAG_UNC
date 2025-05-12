#!/usr/bin/env python3

import re
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import requests
import datasets
import argparse
import numpy as np
import transformers
from tqdm import tqdm
import time
from accelerate import Accelerator
from accelerate.utils import gather_object

from utils.general_utils import set_seed
from run_searchr1.retrieval_local import BM25Retriever, ContrieverRetriever, RerankRetriever, DenseRetriever
from run_searchr1.correctness import em_score, f1_score


# Define the custom stopping criterion
class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False

def get_think(text):
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None

def get_query(text):
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None

def get_answer(text):
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None

def get_document(text):
    pattern = re.compile(r"<document>(.*?)</document>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None

def get_critique(text):
    pattern = re.compile(r"<critique>(.*?)</critique>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None

def _passages2string(retrieval_result):
    # print(retrieval_result)
    format_reference = ''
    for idx, doc_item in enumerate(retrieval_result):
                    
        content = doc_item['contents']
        # content = doc_item['document']['contents']
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx+1} (Title: {title}) {text}\n"
    return format_reference

def _passages2string_v2(retrieval_result):
    # print(retrieval_result)
    format_reference = ''
    for idx, text in enumerate(retrieval_result):
        format_reference += f"Doc {idx+1} {text}\n"
    return format_reference

# For server retrieval
def search(query: str):
    payload = {
            "queries": [query],
            "topk": 3,
            "return_scores": True
        }
    results = requests.post("http://127.0.0.1:8000/retrieve", json=payload).json()['result']
                

    return _passages2string(results[0])

def searchr1_inference(args):
    # === MultiGPU setup =======================
    accelerator = Accelerator()
    device = accelerator.device
    
    if accelerator.is_main_process:
        print("\n== Search R1 Inference ...")
        print(f"""
            Model name:  {args.model_name_or_path}
            Dataset:     {args.dataset}/{args.subsec} ({args.fraction_of_data_to_use})
            Retriever:   {args.retriever_name} / ({args.retrieval_model_path})
            Seed:        {args.seed}
            Run:         {args.run}
        """.replace('        ', ''))
    
        # === Define CUDA device =======
        # args.device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"Number of available GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA is not available. No GPUs detected.")
        print('\n')
    
    # === Dataset ===============================
    dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', args.dataset)
    if args.subsec == 'train' and 'train' in dataset:
        print(f'Using the {args.dataset} train dataset...')
        test_dataset_ = dataset['train']
    elif 'test' in dataset:
        print(f'Using the {args.dataset} test dataset...')
        test_dataset_ = dataset['test']
    elif 'dev' in dataset:
        print(f'Using the {args.dataset} dev dataset...')
        test_dataset_ = dataset['dev']
    
    if args.fraction_of_data_to_use < 1.0:
        shuffled_dataset = test_dataset_.shuffle(seed=args.seed)
        num_samples = int(args.fraction_of_data_to_use * len(shuffled_dataset))
        test_dataset = shuffled_dataset.select(range(num_samples))
    elif args.fraction_of_data_to_use > 1.0:
        shuffled_dataset = test_dataset_.shuffle(seed=args.seed)
        test_dataset = shuffled_dataset.select(range(int(args.fraction_of_data_to_use)))
    else:
        test_dataset = test_dataset_
    
    if accelerator.is_main_process:
        sample_index = 0
        print(f"Length of Dataset: {len(test_dataset)}")
        print(f"Dataset example {sample_index}:")
        print(f"Id:             {test_dataset[sample_index]['id']}")
        print(f"Question:       {test_dataset[sample_index]['question']}")
        print(f"Answers:        {test_dataset[sample_index]['golden_answers']}")
    
    
    # === generator Model ======================
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16).to(device)
    target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
    stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])
    curr_eos = [151645, 151643] # for Qwen2.5 series models
    curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'


    # === Static Retriever ===================== 
    # if accelerator.is_main_process:
    if args.retriever_name == 'bm25':
        retriever = BM25Retriever(args)  
    elif args.retriever_name == 'contriever':
        retriever = ContrieverRetriever(args)
    elif args.retriever_name in ['rerank_l6', 'rerank_l12']:
        retriever = RerankRetriever(args)
    elif args.retriever_name in ['e5', 'bge']:
        retriever = DenseRetriever(args)

    # === Prompt ===============================
    prompt = """Answer the given question. \
    You must conduct reasoning inside <think> and </think> first every time you get new information. \
    After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
    You can search as many times as your want. \
    If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

    # === Read existing data ===================
    generated_qids = []
    generated_em = []
    if os.path.exists(args.inference_results_file):
        with open(args.inference_results_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'qid' in data:
                    generated_qids.append(data['qid'])
                    generated_em.append(data['em'])
    generated_qids = set(generated_qids)
    filtered_dataset = test_dataset.filter(lambda example: example['id'] not in generated_qids)
    
    # === Inference ============================
    em_evaluation = generated_em
    accelerator.wait_for_everyone() # sync GPUs
    with accelerator.split_between_processes(filtered_dataset) as test_dataset_shard:
        
        inference_results, path_results = [], []
        for i, sample in enumerate(tqdm(test_dataset_shard)):
            qid, question, gt_answers = sample['id'], sample['question'], sample['golden_answers']
            question = question.strip()
            if question[-1] != '?':
                question += '?'
                
            if qid in generated_qids:
                print(f"The answer for query {qid} has been already generated")
                continue # already processed
            
            input_prompt = prompt.format(question=question)
            if tokenizer.chat_template:
                input_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": input_prompt}],
                    add_generation_prompt=True,
                    tokenize=False
                )
        
            path, cnt = [], 0
            while True:
                input_ids = tokenizer.encode(input_prompt, return_tensors='pt').to(device)
                attention_mask = torch.ones_like(input_ids)
                
                # Generate text with the stopping criteria
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_token,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7
                )

                if outputs[0][-1].item() in curr_eos:
                    generated_tokens = outputs[0][input_ids.shape[1]:]
                    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    break

                generated_tokens = outputs[0][input_ids.shape[1]:]
                output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
                tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
                if tmp_query:
                    search_docs = retriever.search(tmp_query)
                    search_results = _passages2string(search_docs)
                else:
                    search_docs, search_results = [], ''

                path.append({
                    'think': get_think(output_text),
                    'search_query': tmp_query,
                    'docs': search_docs
                })

                search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
                input_prompt += search_text
                cnt += 1
            
            one_step_think = get_think(output_text)
            pred_answer = get_answer(output_text)
            path.append({'think': one_step_think, 'answer': pred_answer})
            if pred_answer:
                correctness_em = em_score(pred_answer, gt_answers)
                correctness_f1 = f1_score(pred_answer, gt_answers)
            else:
                correctness_em = 0
                correctness_f1 = {'f1': 0, 'precision': 0, 'recall': 0}
            
            item1 = {
                "qid": qid,
                "query": question,
                "gt_answers": gt_answers,
                "pred_answer": pred_answer,
                "em": correctness_em,
                "f1": correctness_f1
            }
            item2 = {
                "qid": qid,
                "query": question,
                "gt_answers": gt_answers,
                "pred_answer": pred_answer,
                "path": path
            }
            inference_results.append(item1)
            path_results.append(item2)
            em_evaluation.append(correctness_em)
    
    
    inference_results_gathered = gather_object(inference_results)
    path_results_gathered = gather_object(path_results)
    em_evaluation_gathered = gather_object(em_evaluation)
    
    if accelerator.is_main_process:
        # --- Save results
        with open(args.inference_results_file, "a", encoding="utf-8") as inf_f:
            for item in inference_results_gathered:
                inf_f.write(json.dumps(item) + "\n")
        
        with open(args.path_results_file, "a", encoding="utf-8") as pth_f:
            for item in path_results_gathered:
                pth_f.write(json.dumps(item) + "\n")
    
        # --- Print results
        print("\nEvaluation Result:")
        print(em_evaluation_gathered)
        print(f"EM: {np.mean(em_evaluation_gathered)*100}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model_name_or_path', type=str, default='PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo')
    # parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    
    parser.add_argument('--max_new_token', type=int, default=1024)
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='popqa', choices=[
        'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
    
    # Retriever
    parser.add_argument('--retriever_name', type=str, default='e5', choices=[
        'bm25', 'contriever', 'rerank_l6', 'rerank_l12', 'e5', 'bge'
    ])
    parser.add_argument('--corpus_path', type=str, default='data/search_r1_files/wiki-18.jsonl')
    parser.add_argument('--index_path', type=str, default='data/search_r1_files/e5_Flat.index', choices=[
        'data/search_r1_files/bm25',          # For BM25 & Rerank
        'data/search_r1_files/e5_Flat.index', # For E5
    ])
    parser.add_argument("--retrieval_model_path", type=str, default="intfloat/e5-base-v2", choices=[
        "cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-MiniLM-L12-v2", # For Rerank
        "intfloat/e5-base-v2" # For E5
    ])
    parser.add_argument('--retrieval_topk', type=int, default=3)
    parser.add_argument('--faiss_gpu', action='store_false', help='Use GPU for computation')
    parser.add_argument('--retrieval_pooling_method', type=str, default="mean")
    parser.add_argument('--retrieval_query_max_length', type=int, default=256)
    parser.add_argument('--retrieval_use_fp16', action='store_false', help='')
    parser.add_argument('--retrieval_batch_size', type=int, default=512)
    parser.add_argument("--bm25_k1", type=float, default=0.9)
    parser.add_argument("--bm25_b", type=float, default=0.4)
    
    # Others
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--run', type=str, default='run_22 (search_r1_mgpu)')
    parser.add_argument("--seed", type=int, default=10)
    
    args = parser.parse_args()
    
    # === Files ====================
    args.output_dir = f"run_output/{args.run}" 
    model_ = args.model_name_or_path.split('/')[-1]
    args.output_dir = f"{args.output_dir}/{model_}/{args.dataset}_{args.subsec}/{args.retriever_name}"
    args.inference_results_file = f"{args.output_dir}/inference_results.jsonl"
    args.path_results_file = f"{args.output_dir}/path_results.jsonl"
    os.makedirs(args.output_dir, exist_ok=True)
        
    ### === Run Steps =============
    set_seed(args.seed)
    searchr1_inference(args)
    
    
    # python run_searchr1/inference.py
    # accelerate launch --multi_gpu --num_processes 2 run_searchr1/inference.py
    # accelerate launch --num_processes 1 run_searchr1/inference.py
