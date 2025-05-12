#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import argparse
import numpy as np
from utils.general_utils import set_seed
from run_searchr1.correctness import em_score, f1_score, subem_score

def sr1_evaluation(args):
    
    # === Dataset & Metric Setup ================
    em_evaluation = []
    with open(args.inference_results_file, 'r') as infile:
        for line in infile:
            data = json.loads(line)
            gt_answers = data['gt_answers']
            pred_answer = data['pred_answer']
            # em_socre = em_score(pred_answer, gt_answers)
            # pred_answers = data['pred_answers']
            # em_socre = em_score_v2(pred_answers, gt_answers)
            em_socre = subem_score(pred_answer, gt_answers)
            em_evaluation.append(em_socre)
    
    # === Print results ========================
    print("\nEvaluation Result:")
    print(f"EM: {np.mean(em_evaluation)*100}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model_name_or_path', type=str, default='PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo')
    # parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    
    parser.add_argument('--max_new_token', type=int, default=1024)
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='bamboogle', choices=[
        'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
    
    # Retriever
    parser.add_argument('--retriever_name', type=str, default='rerank_l6', choices=[
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
    parser.add_argument('--run', type=str, default='run_4 (search_r1)')
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
    sr1_evaluation(args)
    
    
    # python run_searchr1/sr1_evaluation.py
