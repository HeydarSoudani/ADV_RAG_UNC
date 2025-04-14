#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import argparse
from tqdm import tqdm, trange

from utils.general_utils import set_seed
from src_adaptive.dataset import BaseDataset

from src_adaptive.retrieve import BM25, Rerank
from src_mcts.evaluate import Evaluator
from src_mcts.generate_node import Generator
from src_mcts.MCTS_backbone import MCTS_Searcher
from src_mcts.MCTS_reasoning_v2 import Reasoning_MCTS_Node
from utils.mcts_utils import (
    Node_Type,
    stochastic_find_best_solution,
    print_tree_from_root
)


def mcts_generation(args):
    print("\n== MCTS Generation ...")
    print(f"""
        Model name:  {args.model_name_or_path}
        Dataset:     {args.dataset}/{args.subsec} ({args.fraction_of_data_to_use})
        Retriever:   {args.retriever_model}
        Rollouts:    {args.num_rollouts}
        Seed:        {args.seed}
        Run:         {args.run}
    """.replace('        ', ''))
    
    # === Output files ==========================
    generated_qids = [name for name in os.listdir(args.generation_trees_results_dir) if os.path.isdir(os.path.join(args.generation_trees_results_dir, name))]

    # === Dataset ===============================
    dataset_ = BaseDataset(args.dataset, args.subsec, args.fraction_of_data_to_use)
    dataset = dataset_.dataset
    
    sample_index = 0
    print(f"Dataset example {sample_index}:")
    print(f"Id:             {dataset[sample_index]['qid']}")
    print(f"Question:       {dataset[sample_index]['question']}")
    print(f"Answers:        {dataset[sample_index]['ground_truths']}")
    print(f"Reasoning Steps:{dataset[sample_index]['reasoning_steps']}")
    print(f"Gold Context: \n{dataset[sample_index]['positive_ctxs'][0]}\n\n")
    
    
    # === Model Definition ======================    
    retriever = BM25(args) if args.retriever_model == 'bm25' else Rerank(args) if args.retriever_model == 'rerank' else ""
    evaluator = Evaluator()
    node_generator = Generator(args, retriever, evaluator)
    
    
    # === Generation =============================
    for i, data_item in enumerate(tqdm(dataset)):
        qid = data_item["qid"]
        user_query = data_item["question"]
        gt_answer = data_item["ground_truths"]
        gt_reasoning_steps = data_item["reasoning_steps"]
        
        if qid in generated_qids:
            print(f"The MCTS for query {qid} has been already generated")
        else:
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
                user_question=user_query,
                gt_answer=gt_answer,
                gt_reasoning_steps=gt_reasoning_steps,
                max_depth_allowed=args.max_depth_allowed,
                enable_potential_score=args.enable_potential_score,
            )
            
            model_solutions = []
            model_all_solutions = []
            model_rollout_nodes = []
            for i in (pbar := trange(args.num_rollouts, disable=True, position=0)):
                rollout_node = mcts_searcher.do_rollout(root_node, i)
                model_rollout_nodes.append(rollout_node)

                all_solution_nodes, all_solutions = stochastic_find_best_solution(
                    root_node, node_generator.evaluator, enable_potential_score=args.enable_potential_score
                )
                model_all_solutions.append(all_solutions)

                os.makedirs(f"{args.generation_trees_results_dir}/{qid}", exist_ok=True)
                with open(f"{args.generation_trees_results_dir}/{qid}/rollout_{i}.tree", "w") as f:
                    print_tree_from_root(
                        mcts_searcher=mcts_searcher,
                        rollout_id=i,
                        root_node=root_node,
                        chosen_node=None,
                        file=f,
                    )

            #! record final traces
            js = [{"trace": node.solution_trace, "rollout_id": node.rollout_id} for node in all_solution_nodes]
            with open(f"{args.generation_trees_results_dir}/{qid}/final_solutions.jsonl", "w") as f:
                for item in js:
                    f.write(json.dumps(item) + "\n")
            js2 = [{"trace": node.solution_trace, "rollout_id": i} for i, node in enumerate(model_rollout_nodes)]
            with open(f"{args.generation_trees_results_dir}/{qid}/rollout_solutions.jsonl", "w") as f:
                for item in js2:
                    f.write(json.dumps(item) + "\n")

            if args.enable_potential_score:
                js = [node.potential_answers_history for node in all_solution_nodes]
                with open(f"{args.generation_trees_results_dir}/{qid}/potentials.json", "w") as f:
                    json.dump(js, f)


    # === Save results ==========================
    reuslts_dict = {
        'Tokens': node_generator.counter.token / len(dataset),
        'Sentences': node_generator.counter.sentence / len(dataset),
        'Retrieves': node_generator.counter.retrieve / len(dataset)
    }
    with open(args.statistics_results_file, 'w') as file:
        json.dump(reuslts_dict, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--dataset', type=str, default='musique', choices=[
        'wikimultihopqa', 'hotpotqa', 'musique', 'iirc', 'multihop_rag',
        'nqgold', 'trivia', 'popqa',
        'factscore'
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--retriever_model', type=str, default='rerank', choices=[
        'positive', 'negative', 'bm25', 'contriever', 'rerank', 'bge_m3', 'sgpt', 'mistral_e5' # intfloat/e5-mistral-7b-instruct -> from "Search-R1"
    ])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=0.004)
    parser.add_argument('--fewshot', type=int, default=6)
    parser.add_argument("--bm25_k1", type=float, default=0.9)
    parser.add_argument("--bm25_b", type=float, default=0.4)
    parser.add_argument('--retrieve_topk', type=int, default=3)
    parser.add_argument('--retrieve_max_query_length', type=int, default=64)
    parser.add_argument('--max_new_token', type=int, default=512)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--run', type=str, default='run_7 (counter_test)')
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
    mcts_generation(args)
    
    
    # python run_mcts/mcts_generation.py --verbose
    