#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import argparse
import numpy as np

from utils.general_utils import set_seed
from src_adaptive.evaluate import CorrectnessEval


def mcts_evaluation(args):
    print("\n== MCTS Evaluation ...")
    print(f"""
        Model name:  {args.model_name_or_path}
        Dataset:     {args.dataset}/{args.subsec} ({args.fraction_of_data_to_use})
        Retriever:   {args.retriever_model}
        Rollouts:    {args.num_rollouts}
        Seed:        {args.seed}
        Run:         {args.run}
    """.replace('        ', ''))
    
    # === Dataset & Metric Setup ================
    correctness = CorrectnessEval()
    correctness_res = {
        'EM': [],
        'F1': [],
        'Recall': [],
        'Precision': [],
    }
    with open(args.discriminate_results_file, 'r') as infile:
        for line in infile:
            data = json.loads(line)
            gt_answers = data['gt_answers']
            pred_answer = data['pred_answer']
            em_socre = correctness.exact_match_score(pred_answer, gt_answers)
            f1_score = correctness.f1_score(pred_answer, gt_answers)
            
            correctness_res['EM'].append(em_socre['correct'])
            correctness_res['F1'].append(f1_score['f1'])
            correctness_res['Recall'].append(f1_score['recall'])
            correctness_res['Precision'].append(f1_score['precision'])
            
            
    # === Save results ==========================
    reuslts_dict = {
        'EM': np.mean(correctness_res['EM'])*100,
        'F1': np.mean(correctness_res['F1'])*100,
        'Recall': np.mean(correctness_res['Recall'])*100,
        'Precision': np.mean(correctness_res['Precision'])*100,
    }
    with open(args.evaluate_results_file, 'w') as file:
        json.dump(reuslts_dict, file, indent=4)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--dataset', type=str, default='hotpotqa', choices=[
        'wikimultihopqa', 'hotpotqa', 'musique', 'iirc', 'multihop_rag',
        'nqgold', 'trivia', 'popqa',
        'factscore'
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--retriever_model', type=str, default='rerank', choices=[
        'positive', 'negative', 'bm25', 'contriever', 'rerank', 'bge_m3', 'sgpt', 'mistral_e5' # intfloat/e5-mistral-7b-instruct -> from "Search-R1"
    ])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=0.2)
    parser.add_argument('--fewshot', type=int, default=6)
    parser.add_argument("--bm25_k1", type=float, default=0.9)
    parser.add_argument("--bm25_b", type=float, default=0.4)
    parser.add_argument('--retrieve_topk', type=int, default=3)
    parser.add_argument('--retrieve_max_query_length', type=int, default=64)
    parser.add_argument('--max_new_token', type=int, default=512)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--run', type=str, default='run_7 (prompt_test)')
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument('--use_counter', action='store_false')
    
    # MCTS ---
    parser.add_argument("--verbose", action="store_true", help="extra login")
    parser.add_argument("--mcts_discount_factor", type=float, default=1.0)
    parser.add_argument("--mcts_exploration_weight", type=float, default=2.0)
    parser.add_argument("--mcts_weight_scheduler", choices=["exp", "lin", "const"], default="const")
    parser.add_argument("--save_tree", action="store_true")
    parser.add_argument("--num_rollouts", type=int, default=4)
    parser.add_argument("--max_depth_allowed", type=int, default=4)
    parser.add_argument("--num_votes", type=int, default=1)
    parser.add_argument("--mcts_num_last_votes", type=int, default=5)
    parser.add_argument("--enable_potential_score", action="store_true")
    parser.add_argument("--num_subquestions", type=int, default=3, help="Number of trials for proposing the next subquestion")
    
    # Discrimination ---
    parser.add_argument("--cutoff_rollout", type=int, default=-1)
    parser.add_argument("--start_idx", type=int, default=-1)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--mask_left_boundary", type=float, default=0.2)
    parser.add_argument("--mask_right_boundary", type=float, default=0.5)
    parser.add_argument("--num_masked_solution_traces", type=int, default=4)
    parser.add_argument("--rc_mode", type=str, default="mid", choices=["loose", "mid", "strict", "maj"])
    parser.add_argument("--rc_temperature", type=float, default=1.0)
    parser.add_argument("--rc_n_completions", type=int, default=1)
    parser.add_argument("--rc_criteria", type=str, default="freq", choices=["freq", "reward"])
    parser.add_argument("--threshold", type=float, default=0.999)
    parser.add_argument("--extend_rc_mode", type=str, default="majority_vote", choices=["original", "BoN", "majority_vote"])
    parser.add_argument("--best_of", type=int, default=5)
    
    args = parser.parse_args()
    
    # === Files ====================
    args.output_dir = f"run_output/{args.run}" 
    model_ = args.model_name_or_path.split('/')[-1]
    args.generation_trees_results_dir = f'{args.output_dir}/{model_}/{args.dataset}_{args.subsec}/{args.retriever_model}/generation_trees'
    args.discriminate_results_file = f"{args.output_dir}/{model_}/{args.dataset}_{args.subsec}/{args.retriever_model}/discriminate_results.jsonl"
    args.evaluate_results_file = f"{args.output_dir}/{model_}/{args.dataset}_{args.subsec}/{args.retriever_model}/evaluate_results.jsonl"
    args.statistics_results_file = f"{args.output_dir}/{model_}/{args.dataset}_{args.subsec}/{args.retriever_model}/statistics_results.jsonl"
    os.makedirs(args.generation_trees_results_dir, exist_ok=True)
    
    # === Prompt files =============
    args.query_decomposition_prompt_file = "prompts_mcts/query_decomposition_prompt_template.txt"
    args.semantic_equivalence_prompt_file = "prompts_mcts/semantic_equivalence_prompt_template.txt"
    
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
    mcts_evaluation(args)
    
    
    # python run_mcts/mcts_evaluation.py
    