#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import argparse
from tqdm import tqdm, trange
import TruthTorchLM as ttlm

from utils.general_utils import set_seed, read_jsonl
from src_mcts.generate_with_uncertainty import GeneratorUNC, ExactMatch


def mcts_uncertainty_evaluation(args):
    print("\n== MCTS Uncertainty Evaluation ...")
    print(f"""
        Model name:  {args.model_name_or_path}
        Dataset:     {args.dataset}/{args.subsec} ({args.fraction_of_data_to_use})
        Retriever:   {args.retriever_model}
        Rollouts:    {args.num_rollouts}
        Seed:        {args.seed}
        Run:         {args.run}
    """.replace('        ', ''))
    
    # === 
    generator_with_unc = GeneratorUNC(args)
    correctness_evaluator = ExactMatch()
    unc_eval_metrics = ['auroc'] # , 'auprc', 'prr'
    
    # === Main Loop =============================
    entries = os.listdir(args.generation_trees_results_dir)
    query_ids = [entry for entry in entries if os.path.isdir(os.path.join(args.generation_trees_results_dir, entry))]
    
    output_dict = {}
    for idx, qid in tqdm(enumerate(query_ids)):
        
        if idx == 10:
            break
        
        final_solutions_file = f"{args.generation_trees_results_dir}/{qid}/final_solutions.jsonl"
        trace_js = read_jsonl(final_solutions_file)
        
        for i, trajectory in enumerate(trace_js):
            trace = trajectory["trace"] if "trace" in trajectory else trajectory
            depth = len(trace.keys()) - 1
            
            question = trace[list(trace.keys())[0]]['user_question']
            ground_truths = trace[list(trace.keys())[0]]['ground_truth']
            prediction = trace[list(trace.keys())[-1]]['think_answer']['answer']
            
            if depth not in list(output_dict.keys()):
                output_dict[depth] = {
                    'qid': [],
                    'question_text': [],
                    'ground_truths': [],
                    'generations': [],
                    'generation_correctness': [],
                }
                for ue_method in generator_with_unc.truth_methods_name:
                    output_dict[depth][ue_method] = {}
                    output_dict[depth][ue_method]['truth_values'] = []
                    output_dict[depth][ue_method]['normalized_truth_values'] = []
            
            input_prompt = generator_with_unc.get_prompt_text(trace)
            truth_dict = generator_with_unc.generate(input_prompt, question)
            is_correct = correctness_evaluator(question, prediction, ground_truths)
            
            print(qid)
            print(ground_truths)
            print(prediction)
            print(truth_dict['generated_text'])
            print(truth_dict['method_specific_outputs'][0]['generated_texts'])
            print(is_correct)
            print('----')
            
            # --
            output_dict[depth]['qid'].append(qid)
            output_dict[depth]['question_text'].append(question)
            output_dict[depth]['ground_truths'].append(ground_truths)
            output_dict[depth]['generations'].append(truth_dict['generated_text'])
            output_dict[depth]['generation_correctness'].append(is_correct)
            for idx, ue_method in enumerate(generator_with_unc.truth_methods_name):
                output_dict[depth][ue_method]['truth_values'].append(truth_dict['unnormalized_truth_values'][idx])
                output_dict[depth][ue_method]['normalized_truth_values'].append(truth_dict['normalized_truth_values'][idx])

    # --
    eval_obj = {}
    for dth in list(output_dict.keys()):
        print(f'Evaluating depth {dth}')
        eval_obj[dth] = {
            'samples': len(output_dict[dth]['qid']),
            'ExactMatch': (sum(output_dict[dth]['generation_correctness'])/len(output_dict[dth]['generation_correctness'])) * 100,
        }
        
        for idx, ue_method in enumerate(generator_with_unc.truth_methods_name):
            eval_dict = ttlm.utils.metric_score(
                unc_eval_metrics,
                output_dict[dth]['generation_correctness'],
                output_dict[dth][ue_method]['truth_values'],
                output_dict[dth][ue_method]['normalized_truth_values'],
                seed=args.seed
            )
            eval_obj[dth][ue_method] = round(float(eval_dict['auroc']), 3)
    eval_obj = dict(sorted(eval_obj.items()))
    print(eval_obj)        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--dataset', type=str, default='hotpotqa', choices=[
        'wikimultihopqa', 'hotpotqa', 'musique', 'iirc', 'multihop_rag',
        'nqgold', 'trivia', 'popqa',
        'factscore'
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--retriever_model', type=str, default='rerank', choices=[
        'positive', 'negative', 'bm25', 'contriever', 'rerank', 'bge_m3', 'sgpt', 'mistral_e5' # intfloat/e5-mistral-7b-instruct -> from "Search-R1"
    ])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=0.008)
    parser.add_argument('--fewshot', type=int, default=6)
    parser.add_argument("--bm25_k1", type=float, default=0.9)
    parser.add_argument("--bm25_b", type=float, default=0.4)
    parser.add_argument('--retrieve_topk', type=int, default=3)
    parser.add_argument('--retrieve_max_query_length', type=int, default=64)
    parser.add_argument('--max_new_token', type=int, default=32)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--run', type=str, default='run_1 (rollout_4)')
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument('--use_counter', action='store_false')
    
    # MCTS ---
    parser.add_argument("--verbose", action="store_true", help="extra login")
    parser.add_argument("--mcts_discount_factor", type=float, default=1.0)
    parser.add_argument("--mcts_exploration_weight", type=float, default=2.0)
    parser.add_argument("--mcts_weight_scheduler", choices=["exp", "lin", "const"], default="const")
    parser.add_argument("--save_tree", action="store_true")
    parser.add_argument("--num_rollouts", type=int, default=2)
    parser.add_argument("--max_depth_allowed", type=int, default=2)
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
    mcts_uncertainty_evaluation(args)
    
    
    # python run_mcts/mcts_uncertainty_evaluation.py
    

