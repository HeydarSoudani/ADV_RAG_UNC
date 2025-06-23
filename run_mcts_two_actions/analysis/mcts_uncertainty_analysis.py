#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import argparse
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import roc_auc_score

from utils.general_utils import set_seed, read_jsonl
from run_rag_methods.src.correctness import em_score



def get_auroc(correctness, confidence):
    try:
        auroc = roc_auc_score(correctness, confidence)
    except:
        print(
            "Auroc couldn't be calculated because there is only one class. Returning 0.5 as auroc."
        )
        auroc = 0.5
    return auroc

def uncertainty_analysis(args):
    
    entries = os.listdir(args.generation_trees_results_dir)
    query_ids = [entry for entry in entries if os.path.isdir(os.path.join(args.generation_trees_results_dir, entry))]
    sorted_query_ids = sorted(query_ids, key=lambda x: int(x.split('_')[1]))

    overall_results = {
        "pos_pos": {
            "param": {'confidence': [], 'entropy': [], 'PE': [], 'SE': []},
            "cont": {'confidence': [], 'entropy': [], 'PE': [], 'SE': []},
        },
        "neg_neg": {
            "param": {'confidence': [], 'entropy': [], 'PE': [], 'SE': []},
            "cont": {'confidence': [], 'entropy': [], 'PE': [], 'SE': []},
        },
        "neg_pos": {
            "param": {'confidence': [], 'entropy': [], 'PE': [], 'SE': []},
            "cont": {'confidence': [], 'entropy': [], 'PE': [], 'SE': []},
        },
        "pos_neg": {
            "param": {'confidence': [], 'entropy': [], 'PE': [], 'SE': []},
            "cont": {'confidence': [], 'entropy': [], 'PE': [], 'SE': []},
        },
    }
    
    def get_group(only_q_em, solution_em):
        if only_q_em == 1 and solution_em == 1:
            return "pos_pos"
        if only_q_em == 0 and solution_em == 0:
            return "neg_neg"
        if only_q_em == 0 and solution_em == 1:
            return "neg_pos"
        if only_q_em == 1 and solution_em == 0:
            return "pos_neg"

    def compute_mean(lst):
        return sum(lst) / len(lst) if lst else 0

    for i, qid in enumerate(tqdm(sorted_query_ids)):
        
        final_solutions_file = f"{args.generation_trees_results_dir}/{qid}/final_solutions.jsonl"
        all_traces = read_jsonl(final_solutions_file)
        gt_answers = all_traces[0]["trace"]["0"]["ground_truth"]
        
        # Get only query information
        only_q_param_unc = {}
        for trace_id, trace in enumerate(all_traces):
            if len(trace["trace"]) == 2:
                only_q_prediction = trace["trace"]['1']['think_answer']['answer']
                for unc_method_title, value in trace["trace"]['1']['think_answer']['ue_scores']['param'].items():
                    only_q_param_unc[unc_method_title] = value['uncertainty']     
        only_q_em = em_score(only_q_prediction, gt_answers)
        
        # print(only_q_param_unc)
        # print(only_q_em)
        
        for trace_id, trace in enumerate(all_traces):
            trace_ = trace["trace"]
            if len(trace_) > 2:
                trace_ = {int(key): val for key, val in  trace_.items()}
                last_depth_key = list(trace_.keys())[-1]
                last_node_type = list(trace_[last_depth_key].keys())[0] 
                final_answer = trace_[last_depth_key][last_node_type]["answer"]
                solution_em = em_score(final_answer, gt_answers)
                
                gp_title = get_group(only_q_em, solution_em)
                for key, value in trace_[last_depth_key][last_node_type]["ue_scores"].items():
                    for ue_method, value2 in value.items():
                        overall_results[gp_title][key][ue_method].append(only_q_param_unc[ue_method] - value2["uncertainty"])  

    print(overall_results)
    print('\n')
    
    # Transforming the dictionary
    averaged_results = {
        outer_key: {
            inner_key: {
                metric: compute_mean(values)
                for metric, values in inner_dict.items()
            }
            for inner_key, inner_dict in outer_dict.items()
        }
        for outer_key, outer_dict in overall_results.items()
    }
    
    print(averaged_results)

def depth_based_ue_evaluation(args):
    entries = os.listdir(args.generation_trees_results_dir)
    query_ids = [entry for entry in entries if os.path.isdir(os.path.join(args.generation_trees_results_dir, entry))]
    
    overall_results = {}
    
    for i, qid in enumerate(tqdm(query_ids)):
        final_solutions_file = f"{args.generation_trees_results_dir}/{qid}/final_solutions.jsonl"
        all_traces = read_jsonl(final_solutions_file)
        gt_answers = all_traces[0]["trace"]["0"]["ground_truth"]
        
        for trace_id, trace in enumerate(all_traces):
            trace_ = trace["trace"]
            assert len(trace_) > 1
            depth = len(trace_) - 1
            
            if depth not in list(overall_results.keys()):
                overall_results[depth] = {
                    "correctness": [],
                    "param": {'entropy': [], 'confidence': [], 'PE': [], 'SE': []},
                    "cont": {'entropy': [], 'confidence': [], 'PE': [], 'SE': []},
                }
            
            last_depth_key = list(trace_.keys())[-1]
            last_node_type = list(trace_[last_depth_key].keys())[0] 
            final_answer = trace_[last_depth_key][last_node_type]["answer"]
            trace_em = em_score(final_answer, gt_answers)

            overall_results[depth]['correctness'].append(trace_em)
            
            for key, value in trace_[last_depth_key][last_node_type]["scores"][1].items():
                    for ue_method, value2 in value.items():
                        overall_results[depth][key][ue_method].append(value2["uncertainty"])
    sorted_overall_results = dict(sorted(overall_results.items()))
    
    def nested_dict():
        return defaultdict(nested_dict)
    
    auroc_results = {}
    for depth_, value in sorted_overall_results.items():
        correctness = value['correctness']
        correctness_inverted = list(map(lambda x: 1 - x, correctness))
        auroc_results[depth_] = {}
        auroc_results[depth_]['correctness'] = sum(correctness)/len(correctness)
        auroc_results[depth_]['n_samples'] = len(correctness)
        for j in ['param', 'cont']:
            auroc_results[depth_][j] = {}
            for ue_method, ue_values in sorted_overall_results[depth_][j].items():
                auroc_results[depth_][j][ue_method] = get_auroc(correctness_inverted, ue_values)
    

    print('\n')    
    for depth__, value_ in auroc_results.items():
        param_txt = " | ".join([f"{val:3f}" for key, val in value_['param'].items()])
        cont_txt = " | ".join([f"{val:3f}" for key, val in value_['cont'].items()])
        print(f"D {depth__}: {value_['n_samples']:3d} | {value_['correctness']:2f} || {param_txt} || {cont_txt} ")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--max_new_token', type=int, default=1024)
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='bamboogle', choices=[
        'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
    parser.add_argument("--enable_fewshot_examples", action="store_true", help="")
    
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
    parser.add_argument('--retrieval_query_max_length', type=int, default=256)
    parser.add_argument('--retrieval_use_fp16', action='store_false', help='')
    parser.add_argument('--retrieval_batch_size', type=int, default=512)
    parser.add_argument("--bm25_k1", type=float, default=0.9)
    parser.add_argument("--bm25_b", type=float, default=0.4)
    
    # Others
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--run', type=str, default='run_16 (with_unc_roll4)')
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument('--use_counter', action='store_false')
    
    args = parser.parse_args()
    
    # === Files ====================
    model_ = args.model_name_or_path.split('/')[-1]
    output_dir = f"run_output/{args.run}/{model_}/{args.dataset}_{args.subsec}/{args.retriever_name}"
    args.generation_trees_results_dir = f'{output_dir}/generation_trees'
    
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
    # uncertainty_analysis(args)
    depth_based_ue_evaluation(args)
    
    
    # python run_mcts/mcts_uncertainty_analysis.py
