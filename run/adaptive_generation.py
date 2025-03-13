#!/usr/bin/env python3


import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import TruthTorchLM as ttlm

from utils.utils import set_seed
from src.dataset import BaseDataset
from src.generate import *


def adaptive_generation(args):
    print("\n== Adaptive Generation ...")
    print(f"""
        Model name:  {args.model_name_or_path}
        Dataset:     {args.dataset}/{args.subsec} ({args.fraction_of_data_to_use})
        RAG Method:  {args.rag_method} 
        Correctness: {args.accuracy_metric}
        Seed:        {args.seed}
    """.replace('        ', ''))
    
    # === Output files ==========================
    model_ = args.model_name_or_path.split('/')[-1]
    
    # === Generation Model ======================
    # model = "gpt-4o"
    # model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(args.device)
    
    
    # === Correctness Evaluator =================
    # if args.accuracy_metric == "exact_match":
    #     correctness_evaluator = ttlm.evaluators.ExactMatch()    
    
    # elif args.accuracy_metric == "model_judge":
    #     if 'gpt' in args.model_eval:
    #         correctness_evaluator = ttlm.evaluators.ModelJudge(model=args.model_eval, num_retries=3)
    #     else:
    #         # correctness_evaluator = ttlm.evaluators.ModelJudge(model=model, tokenizer=tokenizer, num_retries=3)
    #         # model_eval = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(args.device)
    #         model_eval = AutoModelForCausalLM.from_pretrained(args.model_eval, torch_dtype=torch.bfloat16, device_map='auto')
    #         tokenizer_eval = AutoTokenizer.from_pretrained(args.model_eval, use_fast=False)
    #         correctness_evaluator = ttlm.evaluators.ModelJudge(model=model_eval, tokenizer=tokenizer_eval, num_retries=3)
    

    # === Dataset Setup =========================
    dataset_ = BaseDataset(args.dataset, args.subsec, args.fraction_of_data_to_use)
    dataset = dataset_.dataset
    fewshot_examplers = dataset_.examplers[:args.fewshot] if len(dataset_.examplers) > 0 else []

    sample_index = 0
    print(f"Dataset example {sample_index}:")
    print(f"Id:             {dataset[sample_index]['qid']}")
    print(f"Question:       {dataset[sample_index]['question']}")
    print(f"Answers:        {dataset[sample_index]['ground_truths']}")
    print(f"Gold Context: \n{dataset[sample_index]['gold_contexts'][0]}")
 
 
    # === Select RAG Method =====================
    if args.rag_method == "no_retrieval":
        model = NoRAG(args)
    elif args.rag_method == "single_retrieval":
        model = SingleRAG(args)
    elif args.method == "fix_length_retrieval" or args.method == "fix_sentence_retrieval":
        model = FixLengthRAG(args)    
    
    # === Generation ============================
    for i in tqdm(range(len(dataset))):
        batch = dataset[i]
        pred = model.inference(batch["question"], fewshot_examplers)
        pred = pred.strip()
        ret = {
            "qid": batch["qid"], 
            "prediction": pred,
            "gold_answer": batch["ground_truths"]
        }
        print(f"{ret}\n")
    
    
    # === Save results ==========================





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--dataset', type=str, default='wikimultihopqa', choices=[
        'wikimultihopqa', 'hotpotqa', 'musique', 'iirc', 'multihop_rag',
        'nqgold', 'trivia', 'popqa',
        'factscore'
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--rag_method', type=str, default='no_retrieval', choices=[
        'no_retrieval', 'single_retrieval',
        'fix_length_retrieval', 'fix_sentence_retrieval',
        'flare', 'dragin'
    ])
    parser.add_argument('--retriever', type=str, default='bm25', choices=[
        'positive', 'negative', 'bm25', 'contriever', 'rerank', 'bge_m3', 'sgpt' # https://github.com/Muennighoff/sgpt
    ])
    parser.add_argument('--accuracy_metric', type=str, default="exact_match", choices=[
        'exact_match', 'model_judge', 'bem_score', 'bert_score', 'rouge_score'
    ])
    parser.add_argument('--model_eval', type=str, default='gpt-3.5-turbo') # meta-llama/Llama-3.1-8B-Instruct
    
    parser.add_argument('--use_counter', action='store_false')
    parser.add_argument('--fewshot', type=int, default=8)
    parser.add_argument('--generate_max_length', type=int, default=128)
    parser.add_argument('--fraction_of_data_to_use', type=float, default=0.01)
    parser.add_argument("--roc_auc_threshold", type=float, default=0.8)
    parser.add_argument('--num_generations', type=int, default=10)
    parser.add_argument('--max_new_tokens', type=int, default=32)
    parser.add_argument('--decoding_method', type=str, default='beam_search')
    parser.add_argument('--temperature', type=float, default='1.0')
    parser.add_argument('--num_beams', type=int, default='1')
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--run', type=str, default='run_0')
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()
    
    ### === Define CUDA device =================== 
    args.output_dir = f"run_output/{args.run}" 
    args.device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. No GPUs detected.")
        
    
    ### === Run Steps ============================
    set_seed(args.seed)
    adaptive_generation(args)
    
    
    # python run/adaptive_generation.py

