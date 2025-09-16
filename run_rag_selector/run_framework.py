#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import argparse

from utils.general_utils import set_seed
from run_rag_selector.training import training
from run_rag_selector.inference import inference
from run_rag_selector.data_preparation import data_preparation
from run_rag_selector.src.ideal_selector import get_ideal_selector
from run_rag_selector.src.wo_training_selector import wo_training_selector

def main(args):
    if args.get_ideal:
        get_ideal_selector(args)
    
    else:
        dataset = data_preparation(args)    
        if args.with_training:
            training(args, dataset)
            inference(args, dataset)
        else:
            wo_training_selector(args, dataset)

def stat_testing(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--selector_model_name_or_path', type=str, default='answerdotai/ModernBERT-large', choices=[
        'answerdotai/ModernBERT-large', 'BAAI/bge-large-en-v1.5', 'google/embeddinggemma-300m',
        'Alibaba-NLP/gte-Qwen2-7B-instruct', 'Alibaba-NLP/gte-Qwen2-1.5B-instruct' # https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct
    ])
    parser.add_argument('--secondary_model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--cache_dir', type=str, default='./cache/')
    parser.add_argument("--data_cache_dir", type=str, default="./run_rag_selector/datasets")
    parser.add_argument("--max_input_tokens", type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--dataset', type=str, default='hotpotqa', choices=[
        'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'
    ])
    parser.add_argument('--subsec', type=str, default='dev', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--prompt_format', type=str, default='x_o_c_sq_dc', choices=[
        'x_o', 'x_o_sq', 'x_o_th', 'x_o_dc', 'x_o_g', 'x_o_g_sq', 'x_o_g_dc', 'x_o_sq_dc', 'x_o_sq_th_dc',
        'x_o_c', 'x_o_c_sq', 'x_o_c_th', 'x_o_c_dc', 'x_o_c_g', 'x_o_c_g_sq', 'x_o_c_g_dc', 'x_o_c_sq_dc', 'x_o_c_sq_th_dc',
    ])
    parser.add_argument('--n_docs_prompt', type=int, default=3)
    parser.add_argument('--retriever_name', type=str, default='rerank_l6', choices=[
        'bm25', 'contriever', 'rerank_l6', 'rerank_l12', 'e5', 'bge', 'reasonir'
    ])
    parser.add_argument('--consistency_method', type=str, default='rag_consistency', choices=[
        'self_consistency', 'reasoning_consistency', 'rag_consistency'
    ])
    # 
    parser.add_argument('--get_ideal', action='store_true')
    parser.add_argument('--with_training', action='store_false')
    parser.add_argument('--with_clustering', action='store_false')
    parser.add_argument('--confidence_score_injection', type=str, default='in_input', choices=['in_input', 'in_representation'])
    parser.add_argument('--training_method', type=str, default='pairwise', choices=['pairwise', 'listwise'])
    # 
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default="2e-5")
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=4)
    # 
    parser.add_argument('--run_train', type=str, default='run_1 (rag_methods_2k)')
    parser.add_argument('--run_test', type=str, default='run_2 (rag_methods_1k)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--retry", type=int, default=3)
    args = parser.parse_args()
    
    # === Define CUDA device =======
    args.device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. No GPUs detected.")
    
    # === Add variables
    args.rag_methods = ['self_ask', 'react', 'search_o1', 'research', 'search_r1']
    args.semantic_equivalence_prompt_file = "run_mcts/run_mcts_two_actions/prompts/semantic_equivalence_prompt_template.txt"
    
    model_ = args.selector_model_name_or_path.split('/')[-1]
    args.saved_model_name_or_path = f"run_rag_selector/models/{args.dataset}/{args.training_method}_confidence_{args.confidence_score_injection}_{args.prompt_format}/{model_}"
    
    clustering_text = 'clustering' if args.with_clustering  else 'wo_clustering'
    results_dir = f"run_output/{args.run_test}/rag_selector/{args.dataset}_{args.subsec}_{args.consistency_method}"
    os.makedirs(results_dir, exist_ok=True)
    args.save_results_path = f"{results_dir}/{args.training_method}_confidence_{args.confidence_score_injection}_{clustering_text}_{args.prompt_format}_results.jsonl"
    
    set_seed(args.seed)
    main(args)
    
    # ---------------------------------
    
    stat_testing(args)
    
    
    # python run_rag_selector/run_framework.py
    # accelerate launch --multi_gpu run_rag_selector/run_framework.py

