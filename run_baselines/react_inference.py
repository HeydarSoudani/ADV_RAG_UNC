#!/usr/bin/env python3

# Ref: https://github.com/ysymyth/ReAct/blob/master/hotpotqa.ipynb

import re
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import datasets
import argparse
import jsonlines
import numpy as np
import transformers
from tqdm import tqdm

from utils.general_utils import set_seed
from run_searchr1.retrieval_local import BM25Retriever, ContrieverRetriever, RerankRetriever, DenseRetriever
from run_searchr1.correctness import em_score, f1_score
from run_searchr1.inference import StopOnSequence
from run_baselines.react_examples import examples


def react_inference(args):
    print("\n== ReAct Inference ...")
    print(f"""
        Model name:  {args.model_name_or_path}
        Dataset:     {args.dataset}/{args.subsec} ({args.fraction_of_data_to_use})
        Retriever:   {args.retriever_name} / ({args.retrieval_model_path})
        Seed:        {args.seed}
        Run:         {args.run}
    """.replace('        ', ''))

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
    
    sample_index = 0
    print(f"Length of Dataset: {len(test_dataset)}")
    print(f"Dataset example {sample_index}:")
    print(f"Id:             {test_dataset[sample_index]['id']}")
    print(f"Question:       {test_dataset[sample_index]['question']}")
    print(f"Answers:        {test_dataset[sample_index]['golden_answers']}")
    
    # === generator Model ======================
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
    curr_eos = [151645, 151643] # for Qwen2.5 series models

    observation_sequences = ["Observation", " Observation", "\nObservation", " \nObservation", "\n\nObservation", " \n\nObservation"]
    observation_stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(observation_sequences, tokenizer)])
    curr_eos = [151645, 151643] # for Qwen2.5 series models

    # === Static Retriever ===================== 
    if args.retriever_name == 'bm25':
        retriever = BM25Retriever(args)  
    elif args.retriever_name == 'contriever':
        retriever = ContrieverRetriever(args)
    elif args.retriever_name in ['rerank_l6', 'rerank_l12']:
        retriever = RerankRetriever(args)
    elif args.retriever_name in ['e5', 'bge']:
        retriever = DenseRetriever(args)

    # === Prompt ===============================
    instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps.
    Thought can reason about the current situation, and Action can be three types: 
    (1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
    (2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
    (3) Finish[answer], which returns the answer and finishes the task.
    Here are some examples.\n
    """
    webthink_examples = examples['webthink_simple6']
    examples_text = ''
    for example in webthink_examples:
        examples_text += f"Question: {example['question']}\n"
        for step_i, think_step in enumerate(example['steps']):
            for step_key, step_val in think_step.items():
                examples_text += f"{step_key} {step_i+1}: {step_val}\n"
        examples_text += "\n"
    
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
    
    # === Inference ============================
    em_evaluation = generated_em
    with jsonlines.open(args.inference_results_file, mode='a') as inf_file, jsonlines.open(args.path_results_file, mode='a') as path_file:
        for i, sample in enumerate(tqdm(test_dataset)):
            # if i == 10:
            #     break
            qid, question, gt_answers = sample['id'], sample['question'], sample['golden_answers']
            question = question.strip()
            if question[-1] != '?':
                question += '?'
                
            if qid in generated_qids:
                print(f"The answer for query {qid} has been already generated")
            else:
                input_prompt = instruction + examples_text + f"Question: {question}\n"
                if tokenizer.chat_template:
                    input_prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": input_prompt}],
                        add_generation_prompt=True,
                        tokenize=False
                    )
                
                path = []
                n_calls, n_badcalls = 0, 0
                for step_idx in range(1, 8):        
                    n_calls += 1
                    input_ids = tokenizer.encode(input_prompt, return_tensors='pt').to(args.device)
                    attention_mask = torch.ones_like(input_ids)
                    
                    outputs = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=args.max_new_token,
                        stopping_criteria=observation_stopping_criteria,
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=True,
                        temperature=0.7
                    )
                    generated_tokens = outputs[0][input_ids.shape[1]:]
                    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    thought, action = output_text.strip().split(f"\nAction {i}: ")
                
                
                # ------
                if pred_answer != None:
                    correctness_em = em_score(pred_answer, gt_answers)
                    correctness_f1 = f1_score(pred_answer, gt_answers)
                else:
                    correctness_em = 0
                    correctness_f1 = {'f1': 0, 'precision': 0, 'recall': 0}
                
                em_evaluation.append(correctness_em)
                path.append({'think': one_step_think, 'answer': pred_answer})
                
                # Save to files
                item1 = {
                    "qid": qid,
                    "query": question,
                    "gt_answers": gt_answers,
                    "pred_answer": pred_answer,
                    "em": correctness_em,
                    "f1": correctness_f1
                }
                inf_file.write(item1)
                item2 = {
                    "qid": qid,
                    "query": question,
                    "gt_answers": gt_answers,
                    "pred_answer": pred_answer,
                    "path": path
                }
                path_file.write(item2)
    
    
    # === Print results ========================
    print("\nEvaluation Result:")
    # print(em_evaluation)
    print(f"EM: {np.mean(em_evaluation)*100}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model_name_or_path', type=str, default='PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo')
    # parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    
    parser.add_argument('--max_new_token', type=int, default=1024)
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='hotpotqa', choices=[
        'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'
    ])
    parser.add_argument('--subsec', type=str, default='dev', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=500.0)
    
    # Retriever
    parser.add_argument('--retriever_name', type=str, default='rerank_l6', choices=[
        'bm25', 'contriever', 'rerank_l6', 'rerank_l12', 'e5', 'bge'
    ])
    parser.add_argument('--corpus_path', type=str, default='data/search_r1_files/wiki-18.jsonl')
    parser.add_argument('--index_path', type=str, default='data/search_r1_files/bm25', choices=[
        'data/search_r1_files/bm25',          # For BM25 & Rerank
        'data/search_r1_files/e5_Flat.index', # For E5
    ])
    parser.add_argument("--retrieval_model_path", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2", choices=[
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
    parser.add_argument('--run', type=str, default='run_4 (search_r1)')
    parser.add_argument("--seed", type=int, default=10)
    
    args = parser.parse_args()
    
    # === Files ====================
    args.output_dir = f"run_output/{args.run}" 
    model_ = args.model_name_or_path.split('/')[-1]
    output_dir = f"{args.output_dir}/{model_}/{args.dataset}_{args.subsec}/{args.retriever_name}"
    args.inference_results_file = f"{output_dir}/inference_results.jsonl"
    args.path_results_file = f"{output_dir}/path_results.jsonl"
    os.makedirs(output_dir, exist_ok=True)
    
    
    # === Define CUDA device =======
    args.device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. No GPUs detected.")
        
    ### === Run Steps =============
    set_seed(args.seed)
    react_inference(args)
    
    
    # python run_searchr1/inference.py
    