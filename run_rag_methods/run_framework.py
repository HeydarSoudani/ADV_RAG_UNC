#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse

from utils.general_utils import set_seed
from run_rag_methods.rag_generation import rag_generation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--max_new_tokens', type=int, default=128) # generate_max_length
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='bamboogle', choices=[
        'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
    parser.add_argument("--enable_fewshot_examples", action="store_true", help="")
    parser.add_argument('--fewshot', type=int, default=6)
    
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
    parser.add_argument('--retrieval_query_max_length', type=int, default=64)
    parser.add_argument('--retrieval_use_fp16', action='store_false', help='')
    parser.add_argument('--retrieval_batch_size', type=int, default=512)
    parser.add_argument("--bm25_k1", type=float, default=0.9)
    parser.add_argument("--bm25_b", type=float, default=0.4)
    
    # RAG setup
    parser.add_argument('--rag_method', type=str, default='self_ask', choices=[
        'direct_inference', 'cot_inference', 'cot_single_retrieval',
        'fix_length_retrieval', 'fix_sentence_retrieval', 'ircot',
        'flare', 'dragin',
        'react', 'self_ask', 'search_o1', 'search_r1'
    ])
    parser.add_argument('--generate_fix_length', type=int, default=25)
    parser.add_argument('--modifier_method', type=str, default='token', choices=['token', 'entity'])          # for FLARE
    parser.add_argument('--query_formulation', type=str, default='real_words', choices=[                      # for FLARE & DRAGIN
        'direct', 'forward_all',
        'real_words', 'current', 'current_wo_wrong', 'last_sentence', 'last_n_tokens',
    ])
    parser.add_argument('--sentence_solver', type=str, default='avg', choices=['avg', 'max', 'min'])          # for FLARE
    parser.add_argument('--hallucination_threshold', type=float, default=1.8)                                 # for FLARE & DRAGIN
    parser.add_argument('--retrieve_keep_top_k', type=int, default=25)                                        # for DRAGIN
    parser.add_argument('--check_real_words', action='store_false')                                           # for DRAGIN
    parser.add_argument('--max_iter', type=int, default=5)                                                    # for IRCoT & FLARE
    
    # Others
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--run', type=str, default='run_1 (rag_methods_2k)')
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument('--use_counter', action='store_false')
    
    args = parser.parse_args()
    
    # === Files ===================================
    model_ = args.model_name_or_path.split('/')[-1]
    if args.rag_method == 'no_retrieval':
        args.output_dir = f"run_output/{args.run}/{model_}/{args.dataset}_{args.subsec}/{args.rag_method}"
    else:
        args.output_dir = f"run_output/{args.run}/{model_}/{args.dataset}_{args.subsec}/{args.rag_method}_{args.retriever_name}"
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ---
    if args.rag_method in ['flare', 'dragin']:
        args.inference_results_file = f"{args.output_dir}/inference_results_th{args.hallucination_threshold}.jsonl"
    else:
        args.inference_results_file = f"{args.output_dir}/inference_results.jsonl"
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

    
    ### === Run Steps ============================
    set_seed(args.seed)
    rag_generation(args)
    
    # python run_rag_methods/run_framework.py
    # accelerate launch --multi_gpu run_rag_methods/run_framework.py
