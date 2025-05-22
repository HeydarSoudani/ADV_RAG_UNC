#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import random
import argparse
import datasets
import numpy as np
from tqdm import tqdm, trange

from utils.general_utils import set_seed, read_jsonl
from run_rag_methods.src.correctness import em_score
from run_searchr1.retrieval_local import BM25Retriever, ContrieverRetriever, RerankRetriever, DenseRetriever
from src_adaptive.dataset import BaseDataset

from run_mcts.searchr1_discrimination import SemanticEquivalenceGenerator
# from src_adaptive.retrieve import BM25, Rerank
from src_mcts.generate_node import Generator
from src_mcts.MCTS_backbone import MCTS_Searcher
from src_mcts.MCTS_reasoning_v2 import Reasoning_MCTS_Node
from utils.mcts_utils import (
    Node_Type,
    stochastic_find_best_solution,
    print_tree_from_root
)


def mcts_discrimination(args):
    print("\n== MCTS Discrimination ...")
    print(f"""
        Model name:  {args.model_name_or_path}
        Dataset:     {args.dataset} / {args.subsec} ({args.fraction_of_data_to_use})
        Retriever:   {args.retriever_name} / ({args.retrieval_model_path})
        Rollouts:    {args.num_rollouts}
        Seed:        {args.seed}
        Run:         {args.run}
    """.replace('        ', ''))
    
    # === Input ================================
    entries = os.listdir(args.generation_trees_results_dir)
    query_ids = [entry for entry in entries if os.path.isdir(os.path.join(args.generation_trees_results_dir, entry))]
    sorted_query_ids = sorted(query_ids, key=lambda x: int(x.split('_')[1])) 
    # Output
    generated_qids = [name for name in os.listdir(args.discrimination_trees_results_dir) if os.path.isdir(os.path.join(args.discrimination_trees_results_dir, name))]
    
    # === Static Retriever ===================== 
    if args.retriever_name == 'bm25':
        retriever = BM25Retriever(args)  
    elif args.retriever_name == 'contriever':
        retriever = ContrieverRetriever(args)
    elif args.retriever_name == 'rerank':
        retriever = RerankRetriever(args)
    elif args.retriever_name in ['e5', 'bge']:
        retriever = DenseRetriever(args)
    
    
    # === Model Definition ======================    
    node_generator = Generator(args, retriever, mcts_type="discrimination")
    se_model = SemanticEquivalenceGenerator(args)
    
    
    # === Tree Generation =======================
    em_evaluation = []
    with open(args.discriminate_results_file, 'w', encoding='utf-8') as inf_file:
        for idx, qid in enumerate(tqdm(sorted_query_ids)):
            print('\n-------------------')
            # if idx == 5:
            #     break
            
            final_solutions_file = f"{args.generation_trees_results_dir}/{qid}/final_solutions.jsonl"
            trace_js = read_jsonl(final_solutions_file)  
            question = trace_js[0]["trace"]["0"]["user_question"]
            question = question.strip()
            if question[-1] != '?':
                question += '?'
                
            gt_answers = trace_js[0]["trace"]["0"]["ground_truth"]
            options = [s["trace"][list(s["trace"].keys())[-1]]['think_answer']["answer"] for s in trace_js]
            options = se_model._filter_none(options)
            options = se_model._filter_long(options)
            options = se_model._filter_white_space(options)
            options = se_model._filter_stop_words(options)
            cls_options = se_model.cluster_by_meaning(question, options)
            print(cls_options)

            if len(cls_options) == 0:
                pred_answer = ''
            elif len(cls_options) == 1:
                pred_answer = random.choice(cls_options[0])
            else:
                unq_options = [random.choice(cls_) for cls_ in cls_options]
                
                # = Tree Geeration ... 
                # if qid in generated_qids:
                #     print(f"The MCTS for query {qid} has been already generated")
                # else:
                print(f"Generating MCTS for query {qid} ...")

                #! build an MCTS searcher
                mcts_searcher = MCTS_Searcher(
                    exploration_weight=args.mcts_exploration_weight,
                    weight_scheduler=args.mcts_weight_scheduler,
                    num_rollouts=args.num_rollouts,
                    discount=args.mcts_discount_factor,
                    verbose=args.verbose,
                )
                #! build the MCTS tree
                root_node = Reasoning_MCTS_Node(
                    parent=None,
                    depth=0,
                    node_type=Node_Type.USER_QUESTION,
                    verbose=args.verbose,
                    generator=node_generator,
                    question_id=qid,
                    user_question=question,
                    gt_answer=gt_answers,
                    gt_reasoning_steps=[],
                    answer_candidates=unq_options,
                    max_depth_allowed=args.max_depth_allowed,
                    enable_potential_score=args.enable_potential_score,
                )
                
                #! do rollout 
                model_all_solutions = []
                model_rollout_nodes = []
                for i in (pbar := trange(args.num_rollouts, disable=True, position=0)):
                    rollout_node = mcts_searcher.do_rollout(root_node, i)
                    model_rollout_nodes.append(rollout_node)
                    all_solution_nodes, all_solutions = stochastic_find_best_solution(
                        root_node, enable_potential_score=args.enable_potential_score
                    )
                    model_all_solutions.append(all_solutions)
                    
                    os.makedirs(f"{args.discrimination_trees_results_dir}/{qid}", exist_ok=True)
                    with open(f"{args.discrimination_trees_results_dir}/{qid}/rollout_{i}.tree", "w") as f:
                        print_tree_from_root(
                            mcts_searcher=mcts_searcher,
                            rollout_id=i,
                            root_node=root_node,
                            chosen_node=None,
                            file=f,
                        )
                
                #! record final traces
                disc_trace_js = [{"trace": node.solution_trace, "rollout_id": node.rollout_id} for node in all_solution_nodes]
                disc_options = [s["trace"][list(s["trace"].keys())[-1]]['think_answer']["answer"] for s in disc_trace_js]
                disc_cls_options = se_model.cluster_by_meaning(question, disc_options)
                print(disc_cls_options)
                
                if len(disc_cls_options) == 0:
                    pred_answer = ''
                elif len(disc_cls_options) == 1:
                    pred_answer = random.choice(disc_cls_options[0])
                else:
                    pred_answer = random.choice([random.choice(cls_) for cls_ in disc_cls_options])

            print(pred_answer)
            correctness_em = em_score(pred_answer, gt_answers)
            em_evaluation.append(correctness_em)
            item = {
                "qid": qid,
                "query": question,
                "em": correctness_em,
                "gt_answers": gt_answers,
                "final_answer": pred_answer,
                "cluster_options": cls_options,
            }
            inf_file.write(json.dumps(item) + '\n')
            
            
    # === Print results ========================
    print("\nEvaluation Result:")
    print(f"EM: {np.mean(em_evaluation)*100}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model_name_or_path_gen', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-3B-Instruct')
    parser.add_argument('--max_new_token', type=int, default=1024)
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='bamboogle', choices=[
        'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
    
    # Retriever
    parser.add_argument('--retriever_name', type=str, default='rerank', choices=[
        'bm25', 'contriever', 'rerank', 'e5'
    ])
    parser.add_argument('--corpus_path', type=str, default='data/search_r1_files/wiki-18.jsonl')
    parser.add_argument('--index_path', type=str, default='data/search_r1_files/bm25', choices=[
        'data/search_r1_files/bm25',          # For BM25 & Rerank
        'data/search_r1_files/e5_Flat.index', # For E5
    ])
    parser.add_argument("--retrieval_model_path", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2", choices=[
        "intfloat/e5-base-v2" # For E5
        "cross-encoder/ms-marco-MiniLM-L12-v2" # For Rerank | cross-encoder/ms-marco-MiniLM-L-6-v2
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
    parser.add_argument('--run', type=str, default='run_5 (edited_prompt_roll4)')
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
    model_ = args.model_name_or_path_gen.split('/')[-1]
    output_dir = f"run_output/{args.run}/{model_}/{args.dataset}_{args.subsec}/{args.retriever_name}"
    args.generation_trees_results_dir = f'{output_dir}/generation_trees'
    args.discrimination_trees_results_dir = f"{output_dir}/discrimination_trees"
    args.discriminate_results_file = f"{output_dir}/mcts_discriminate_results.jsonl"
    os.makedirs(args.discrimination_trees_results_dir, exist_ok=True)
    
    # === Prompt files =============
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
    mcts_discrimination(args)
    
    
    # python run_mcts/mcts_discrimination.py --verbose
    


