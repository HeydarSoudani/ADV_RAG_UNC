#!/usr/bin/env python3


import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import TruthTorchLM as ttlm

from utils.utils import set_seed
from src.dataset import BaseDataset
from src.generate import *
from src.evaluate import CorrectnessEval


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
    
    rag_method_ = args.rag_method if args.rag_method == 'no_retrieval' else f"{args.rag_method}_{args.retriever_model}"
    generations_output_file = f'{args.output_dir}/{model_}/{args.dataset}_{args.subsec}/{rag_method_}_generations.jsonl'
    

    # === Dataset & Metric Setup ================
    correctness = CorrectnessEval()
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
    elif args.rag_method == "fix_length_retrieval" or args.rag_method == "fix_sentence_retrieval":
        model = FixLengthRAG(args)
    elif args.rag_method == 'flare':
        model = FLARE_RAG(args)
    elif args.rag_method == 'dragin':
        model = DRAGIN_RAG(args)
    else:
        raise NotImplementedError
    
    
    def get_answer(text):
        parts = text.split("So, the answer is", 1)  # Split at the first occurrence
        return parts[1].strip() if len(parts) > 1 else ""
    
    # === Generation ============================
    # === Save results ==========================
    os.makedirs(os.path.dirname(generations_output_file), exist_ok=True)
    with open(generations_output_file, 'w', encoding='utf-8') as file:
        for i in tqdm(range(len(dataset))):
            batch = dataset[i]
            cot = model.inference(batch["question"], batch["qid"], fewshot_examplers).strip()
            final_ans = model.regenerate(batch["question"], fewshot_examplers, cot).strip() if "So, the answer is" not in cot else get_answer(cot)
            em_socre = correctness.exact_match_score(final_ans, batch["ground_truths"])
            f1_score = correctness.f1_score(final_ans, batch["ground_truths"])
            
            item = {
                "qid": batch["qid"],
                "question": batch["question"],
                "em_score": em_socre['correct'],
                "f1_score": f1_score,
                "gold_answer": batch["ground_truths"],
                "cot": cot,
                "pred": final_ans,
            }
            file.write(json.dumps(item, ensure_ascii=False) + '\n')





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--dataset', type=str, default='hotpotqa', choices=[
        'wikimultihopqa', 'hotpotqa', 'musique', 'iirc', 'multihop_rag',
        'nqgold', 'trivia', 'popqa',
        'factscore'
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--rag_method', type=str, default='single_retrieval', choices=[
        'no_retrieval', 'single_retrieval',
        'fix_length_retrieval', 'fix_sentence_retrieval',
        'flare', 'dragin'
    ])
    parser.add_argument('--retriever_model', type=str, default='rerank', choices=[
        'positive', 'negative', 'bm25', 'contriever', 'rerank', 'bge_m3', 'sgpt' # https://github.com/Muennighoff/sgpt
    ])
    parser.add_argument('--accuracy_metric', type=str, default="exact_match", choices=[
        'exact_match', 'model_judge', 'bem_score', 'bert_score', 'rouge_score'
    ])
    parser.add_argument('--model_eval', type=str, default='gpt-3.5-turbo') # meta-llama/Llama-3.1-8B-Instruct
    
    parser.add_argument("--roc_auc_threshold", type=float, default=0.8)
    parser.add_argument('--fraction_of_data_to_use', type=float, default=0.05)
    parser.add_argument('--use_counter', action='store_false')
    parser.add_argument('--fewshot', type=int, default=6)
    parser.add_argument('--generate_max_length', type=int, default=100)
    parser.add_argument('--num_generations', type=int, default=10)
    parser.add_argument('--decoding_method', type=str, default='beam_search')
    parser.add_argument('--temperature', type=float, default='1.0')
    parser.add_argument('--num_beams', type=int, default='1')
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument("--bm25_k1", type=float, default=0.9)
    parser.add_argument("--bm25_b", type=float, default=0.4)
    parser.add_argument('--retrieve_topk', type=int, default=3)
    parser.add_argument('--retrieve_max_query_length', type=int, default=64)
    parser.add_argument('--generate_fix_length', type=int, default=32)
    
    parser.add_argument('--modifier_method', type=str, default='token', choices=['token', 'entity'])          # for FLARE
    parser.add_argument('--query_formulation', type=str, default='real_words', choices=[                      # for FLARE & DRAGIN
        'direct', 'forward_all',
        'current', 'current_wo_wrong', 'last_sentence', 'last_n_tokens', 'real_words',
    ])
    parser.add_argument('--sentence_solver', type=str, default='avg', choices=['avg', 'max', 'min'])          # for FLARE
    parser.add_argument('--hallucination_threshold', type=float, default=0.08, choices=['avg', 'max', 'min']) # for FLARE
    parser.add_argument('--retrieve_keep_top_k', type=int, default=25)                                        # for DRAGIN
    
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

