#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import glob
import torch
import argparse
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import gather_object

from utils.general_utils import set_seed, read_jsonl
from run_rag_methods.src.correctness import em_score, subem_score, f1_score, em_score_v2
from run_mcts_two_actions.src.discriminator_methods import *


def mcts_discrimination(args):
    # === MultiGPU setup =======================
    accelerator = Accelerator()
    device = accelerator.device
    
    if accelerator.is_main_process:
        print("\n== MCTS Discrimination ...")
        print(f"""
            Model name:    {args.model_name_or_path}
            Discriminator: {args.discriminator_method}
            Dataset:       {args.dataset} / {args.subsec} ({args.fraction_of_data_to_use})
            Retriever:     {args.retriever_name} / ({args.retrieval_model_path})
            Rollouts:      {args.num_rollouts} ({args.max_depth_allowed})
            Seed:          {args.seed}
            Run:           {args.run}
        """.replace('        ', ''))
    
        # === Define CUDA device ================
        # args.device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"Number of available GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA is not available. No GPUs detected.")
    
    # === Read generated qids ===================
    entries = os.listdir(args.generation_trees_results_dir)
    query_ids = [entry for entry in entries if os.path.isdir(os.path.join(args.generation_trees_results_dir, entry))]
    sorted_query_ids = sorted(query_ids, key=lambda x: int(x.split('_')[1]))

    # === Read existing data ===================
    generated_qids = []
    generated_em = []
    if os.path.exists(args.discrimination_results_file):
        with open(args.discrimination_results_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'qid' in data:
                    generated_qids.append(data['qid'])
                    generated_em.append(data['em'])
    generated_qids = set(generated_qids)
    filtered_sorted_query_ids = [id_ for id_ in sorted_query_ids if id_ not in generated_qids]


    # === Select Discrimination Method ==========
    if args.discriminator_method == "majority_voting":
        discriminator = MajorityVoting(args, device)
    elif args.discriminator_method == "reasoning_consistency":
        discriminator = ReasoningConsistency(args, device)
    elif args.discriminator_method == "rag_consistency":
        discriminator = RagConsistency(args, device)
    elif args.discriminator_method == "llm_selector":
        discriminator = LlmSelector(args, device)
    else:
        raise NotImplementedError
    
    # === Main Loop =============================
    em_evaluation = generated_em
    accelerator.wait_for_everyone()
    with accelerator.split_between_processes(filtered_sorted_query_ids) as sorted_query_ids_shard:
        discriminate_results_file_ranked = f"{args.output_dir}/discrimination_results_{args.discriminator_method}_rank{accelerator.process_index}.jsonl"
        with open(discriminate_results_file_ranked, 'w', encoding='utf-8') as res_f:
            for i, qid in enumerate(tqdm(sorted_query_ids_shard, desc=f"[Rank {accelerator.process_index}]")):
                # if i == 10:
                #     break
                # === Generating answer candidates
                final_solutions_file = f"{args.generation_trees_results_dir}/{qid}/final_solutions.jsonl"
                all_traces = read_jsonl(final_solutions_file)
                gt_answers = all_traces[0]["trace"]["0"]["ground_truth"]
                question = all_traces[0]["trace"]["0"]["user_query"]
                question = question.strip()
                if question[-1] != '?':
                    question += '?'
                
                pred_answer, candidates = discriminator.inference(qid, question, gt_answers, all_traces)
                
                if pred_answer:
                    correctness_em = em_score(pred_answer, gt_answers)
                    correctness_f1 = f1_score(pred_answer, gt_answers)
                else:
                    correctness_em = 0
                    correctness_f1 = {'f1': 0, 'precision': 0, 'recall': 0}
                    
                item = {
                    "qid": qid,
                    "query": question,
                    "gt_answers": gt_answers,
                    "pred_answer": pred_answer,
                    "em": correctness_em,
                    "f1": correctness_f1,
                    "candidates": candidates,
                }
                res_f.write(json.dumps(item) + '\n')
                em_evaluation.append(correctness_em)

    em_evaluation_gathered = gather_object(em_evaluation)
    if accelerator.is_main_process:
        print("\nEvaluation Result:")
        print(f"EM: {np.mean(em_evaluation_gathered)*100}")

def merge_result_files(args):
    results_shard_files = f"{args.output_dir}/discrimination_results_{args.discriminator_method}_rank*.jsonl"
    results_shard_files = sorted(glob.glob(results_shard_files))
    with open(args.discrimination_results_file, "a") as fout:
        for shard_file in results_shard_files:
            if shard_file == args.discrimination_results_file:
                continue
            with open(shard_file, "r") as fin:
                for line in fin:
                    fout.write(line)
            os.remove(shard_file)
            print(f"Deleted shard file: {shard_file}")

def mcts_evaluation(args):
    em_evaluation = []
    with open(args.discrimination_results_file, 'r') as infile:
        for line in infile:
            data = json.loads(line)
            gt_answers = data['gt_answers']
            pred_answer = data['pred_answer']
            candidates = [k for k, v in data['candidates'].items()]
            
            # em_socre = em_score(pred_answer, gt_answers)
            em_socre = em_score_v2(candidates, gt_answers)
            # em_socre = subem_score(pred_answer, gt_answers)
            em_evaluation.append(em_socre)
            
    # === Print results ========================
    print("\nEvaluation Result:")
    print(f"EM: {np.mean(em_evaluation)*100}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--paraphrase_model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='bamboogle', choices=[
        'nq', 'triviaqa', 'popqa', '2wikimultihopqa', 'hotpotqa', 'musique', 'bamboogle'
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
    parser.add_argument("--enable_fewshot_examples", action="store_true", help="")
    
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
    parser.add_argument('--run', type=str, default='run_7 (mcts_multi_actions_500_rollout4)')
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument('--use_counter', action='store_false')
    
    # MCTS ---
    parser.add_argument('--discriminator_method', type=str, default='majority_voting', choices=[
        'majority_voting', 'reasoning_consistency', 'llm_selector', 'rag_consistency'
    ])
    parser.add_argument("--enable_critique", action="store_true", help="")
    parser.add_argument("--enable_doc_generation", action="store_true", help="")
    parser.add_argument("--verbose", action="store_true", help="extra login")
    parser.add_argument("--mcts_discount_factor", type=float, default=1.0)
    parser.add_argument("--mcts_exploration_weight", type=float, default=2.0)
    parser.add_argument("--mcts_weight_scheduler", choices=["exp", "lin", "const"], default="const")
    parser.add_argument("--save_tree", action="store_true")
    parser.add_argument("--num_rollouts", type=int, default=8)
    parser.add_argument("--max_depth_allowed", type=int, default=10)
    parser.add_argument("--num_votes", type=int, default=1)
    parser.add_argument("--mcts_num_last_votes", type=int, default=1)
    parser.add_argument("--enable_potential_score", action="store_true")
    
    # Discrimination ---
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
    # parser.add_argument("--extend_rc_mode", type=str, default="original", choices=["original", "BoN", "majority_vote"])
    parser.add_argument("--best_of", type=int, default=5)
    
    args = parser.parse_args()
    
    # === Files ====================
    model_ = args.model_name_or_path.split('/')[-1]
    args.output_dir = f"run_output/{args.run}/{model_}/{args.dataset}_{args.subsec}/{args.retriever_name}"
    args.generation_trees_results_dir = f'{args.output_dir}/generation_trees'
    args.discrimination_results_file = f"{args.output_dir}/discrimination_results_{args.discriminator_method}.jsonl"
    
    args.reasoning_path_generations_dir = f'{args.output_dir}/paraphrased_paths'
    os.makedirs(args.reasoning_path_generations_dir, exist_ok=True)
    
    # === Prompt files =============
    args.query_decomposition_prompt_file = "run_mcts_two_actions/prompts/query_decomposition_prompt_template.txt"
    args.semantic_equivalence_prompt_file = "run_mcts_two_actions/prompts/semantic_equivalence_prompt_template.txt"
    
    ### === Run Steps ==============
    set_seed(args.seed)
    
    # mcts_discrimination(args)
    # merge_result_files(args)
    mcts_evaluation(args)
    
    # python run_mcts_two_actions/mcts_discriminator.py
    # accelerate launch --multi_gpu run_mcts_two_actions/mcts_discriminator.py
