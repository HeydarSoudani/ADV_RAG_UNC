#!/usr/bin/env python3

# Src code: https://github.com/zhentingqi/rStar

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse

from utils.general_utils import set_seed
from run_mcts.mcts_generator import mcts_generation
from run_mcts.mcts_discriminator import mcts_discrimination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='nq', choices=[
        'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=2000.0)
    parser.add_argument("--enable_fewshot_examples", action="store_true", help="")
    parser.add_argument('--fewshot', type=int, default=6)
    
    # Retriever
    parser.add_argument('--retriever_name', type=str, default='rerank_l6', choices=[
        'bm25', 'contriever', 'rerank_l6', 'rerank_l12', 'e5', 'bge', 'reasonir'
    ])
    parser.add_argument('--corpus_path', type=str, default='data/search_r1_files/wiki-18.jsonl')
    parser.add_argument('--index_path', type=str, default='data/search_r1_files/bm25', choices=[
        'data/search_r1_files/bm25',          # For BM25 & Rerank
        'data/search_r1_files/e5_Flat.index', # For E5
        'data/search_r1_files/reasonir_Flat.index', # For ReasonIR
    ])
    parser.add_argument("--retrieval_model_path", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2", choices=[
        "cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-MiniLM-L12-v2", # For Rerank
        "intfloat/e5-base-v2",  # For E5
        "reasonir/ReasonIR-8B", # For ReasonIR
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
    parser.add_argument('--run', type=str, default='run_4 (mcts_500_rollout4)')
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument('--use_counter', action='store_false')
    
    # MCTS: Generator ---
    parser.add_argument("--enable_critique", action="store_true", help="")
    parser.add_argument("--enable_doc_generation", action="store_true", help="")
    parser.add_argument("--verbose", action="store_true", help="extra login")
    parser.add_argument("--mcts_discount_factor", type=float, default=1.0)
    parser.add_argument("--mcts_exploration_weight", type=float, default=2.0)
    parser.add_argument("--mcts_weight_scheduler", choices=["exp", "lin", "const"], default="const")
    parser.add_argument("--save_tree", action="store_true")
    parser.add_argument("--num_rollouts", type=int, default=4)
    parser.add_argument("--max_depth_allowed", type=int, default=6)
    parser.add_argument("--num_votes", type=int, default=2)
    parser.add_argument("--mcts_num_last_votes", type=int, default=5)
    parser.add_argument("--enable_potential_score", action="store_true")
    
    # MCTS: Discrimination ---
    parser.add_argument('--discriminator_method', type=str, default='majority_voting', choices=[
        'majority_voting', 'best_of_n', 'reasoning_consistency', 'rag_consistency', 'llm_selector'
    ])
    parser.add_argument("--cutoff_rollout", type=int, default=-1)
    parser.add_argument("--start_idx", type=int, default=-1)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--mask_left_boundary", type=float, default=0.2)
    parser.add_argument("--mask_right_boundary", type=float, default=0.5)
    parser.add_argument("--num_masked_solution_traces", type=int, default=5)
    parser.add_argument("--rc_mode", type=str, default="mid", choices=["loose", "mid", "strict", "maj"])
    parser.add_argument("--rc_temperature", type=float, default=1.0)
    parser.add_argument("--rc_n_completions", type=int, default=1)
    parser.add_argument("--rc_criteria", type=str, default="freq", choices=["freq", "reward"])
    parser.add_argument("--threshold", type=float, default=0.999)
    parser.add_argument("--extend_rc_mode", type=str, default="majority_vote", choices=["original", "BoN", "majority_vote"])
    parser.add_argument("--best_of", type=int, default=5)
    args = parser.parse_args()
    
    
    # === Files ==================================
    model_ = args.model_name_or_path.split('/')[-1]
    args.output_dir = f"run_output/{args.run}/{model_}/{args.dataset}_{args.subsec}/{args.retriever_name}"
    args.generation_trees_results_dir = f'{args.output_dir}/generation_trees'
    args.discrimination_results_file = f"{args.output_dir}/discrimination_results_{args.discriminator_method}.jsonl"
    os.makedirs(args.generation_trees_results_dir, exist_ok=True)
    
    # === Prompt files ===========================
    args.query_decomposition_prompt_file = "run_mcts/prompts/query_decomposition_prompt_template.txt"
    args.semantic_equivalence_prompt_file = "run_mcts/prompts/semantic_equivalence_prompt_template.txt"
    
    ### === Run Steps ============================
    set_seed(args.seed)
    # mcts_generation(args)
    mcts_discrimination(args)
    
    # python run_mcts/run_framework.py
    # accelerate launch --multi_gpu run_mcts/run_framework.py
    

