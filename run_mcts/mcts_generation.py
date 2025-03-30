#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import argparse

from utils.general_utils import set_seed
from mcts_generation import mcts_generation
from src_mcts.generate import Generator

def mcts_generation(args):
    print("\n== MCTS Generation ...")
    print(f"""
        Model name:  {args.model_name_or_path}
        Dataset:     {args.dataset}/{args.subsec} ({args.fraction_of_data_to_use})
        Retriever:   {args.retriever_model}
        Seed:        {args.seed}
    """.replace('        ', ''))
    
    # === Output files ==========================
    model_ = args.model_name_or_path.split('/')[-1]
    
    # === Dataset ===============================
    
    
    # === Model Definition ======================
    generator = Generator()
    
    
    # === Generation ============================
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--dataset', type=str, default='wikimultihopqa', choices=[
        'wikimultihopqa', 'hotpotqa', 'musique', 'iirc', 'multihop_rag',
        'nqgold', 'trivia', 'popqa',
        'factscore'
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--retriever_model', type=str, default='bm25', choices=[
        'positive', 'negative', 'bm25', 'contriever', 'rerank', 'bge_m3', 'sgpt'
    ])
    parser.add_argument("--bm25_k1", type=float, default=0.9)
    parser.add_argument("--bm25_b", type=float, default=0.4)
    parser.add_argument('--fraction_of_data_to_use', type=float, default=0.6)
    parser.add_argument('--fewshot', type=int, default=6)
    parser.add_argument('--generate_max_length', type=int, default=64)
    parser.add_argument('--retrieve_topk', type=int, default=3)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--run', type=str, default='run_1 (300s-ct)')
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
    mcts_generation(args)