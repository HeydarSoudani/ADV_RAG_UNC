#!/usr/bin/env python3


import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import argparse

from utils.utils import set_seed
from adaptive_generation import adaptive_generation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='meta-llama/Llama-2-7b-chat-hf')
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
    parser.add_argument('--retriever_model', type=str, default='bm25', choices=[
        'positive', 'negative', 'bm25', 'contriever', 'rerank', 'bge_m3', 'sgpt' # https://github.com/Muennighoff/sgpt
    ])
    parser.add_argument("--bm25_k1", type=float, default=0.9)
    parser.add_argument("--bm25_b", type=float, default=0.4)
    parser.add_argument('--use_counter', action='store_false')
    parser.add_argument('--fraction_of_data_to_use', type=float, default=0.6)
    parser.add_argument('--fewshot', type=int, default=6)
    parser.add_argument('--generate_max_length', type=int, default=64)
    parser.add_argument('--retrieve_topk', type=int, default=3)
    parser.add_argument('--retrieve_max_query_length', type=int, default=64)
    parser.add_argument('--generate_fix_length', type=int, default=25)
    
    parser.add_argument('--modifier_method', type=str, default='token', choices=['token', 'entity'])          # for FLARE
    parser.add_argument('--query_formulation', type=str, default='real_words', choices=[                      # for FLARE & DRAGIN
        'direct', 'forward_all',
        'real_words', 'current', 'current_wo_wrong', 'last_sentence', 'last_n_tokens',
    ])
    parser.add_argument('--sentence_solver', type=str, default='avg', choices=['avg', 'max', 'min'])          # for FLARE
    parser.add_argument('--hallucination_threshold', type=float, default=0.6)                                 # for FLARE & DRAGIN
    parser.add_argument('--retrieve_keep_top_k', type=int, default=25)                                        # for DRAGIN
    parser.add_argument('--check_real_words', action='store_false')                                           # for DRAGIN
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--run', type=str, default='run_1 (300s-ct)')
    parser.add_argument("--seed", type=int, default=10)
    
    # parser.add_argument('--accuracy_metric', type=str, default="exact_match", choices=[
    #     'exact_match', 'model_judge', 'bem_score', 'bert_score', 'rouge_score'
    # ])
    # parser.add_argument('--model_eval', type=str, default='gpt-3.5-turbo') # meta-llama/Llama-3.1-8B-Instruct
    # parser.add_argument("--roc_auc_threshold", type=float, default=0.8)
    # parser.add_argument('--num_generations', type=int, default=10)
    # parser.add_argument('--decoding_method', type=str, default='beam_search')
    # parser.add_argument('--temperature', type=float, default='1.0')
    # parser.add_argument('--num_beams', type=int, default='1')
    # parser.add_argument('--top_p', type=float, default=1.0)
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
