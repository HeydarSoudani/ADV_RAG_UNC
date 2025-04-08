#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import argparse
from copy import deepcopy
from tqdm import tqdm, trange

from src_mcts.evaluate import Evaluator
from src_mcts.discriminate import Candidate, MajorityVoteDiscriminator, group_candidates_by_answer
from utils.general_utils import set_seed, read_txt, read_json, read_jsonl
from utils.mcts_utils import (
    Node_Type,
    stochastic_find_best_solution,
    print_tree_from_root,
    concat_solution_trace,
    mask_solution_trace,
    rag_mask_solution_trace
)


def mcts_discrimination(args):
    print("\n== MCTS Discrimination ...")
    print(f"""
        Model name:  {args.model_name_or_path}
        Dataset:     {args.dataset}/{args.subsec} ({args.fraction_of_data_to_use})
        Retriever:   {args.retriever_model}
        Seed:        {args.seed}
    """.replace('        ', ''))
    
    # === Output files ==========================
    model_ = args.model_name_or_path.split('/')[-1]
    answer_sheets_dir = f'{args.output_dir}/{model_}/{args.dataset}_{args.subsec}/generation_trees'
    entries = os.listdir(answer_sheets_dir)
    query_ids = [entry for entry in entries if os.path.isdir(os.path.join(answer_sheets_dir, entry))]
    discriminate_results_file = f"{args.output_dir}/{model_}/{args.dataset}_{args.subsec}/discriminate_results.jsonl"
    
    # === Model Definition ======================  
    evaluator = Evaluator()
    discriminator = MajorityVoteDiscriminator(args, evaluator)
    evaluator.model = discriminator.model
    evaluator.eos_token_ids = discriminator.model.config.eos_token_id
    evaluator.tokenizer = discriminator.tokenizer
    evaluator.equiv_prompt = read_txt(args.semantic_equivalence_prompt_file)
    
    # === Main Loop =============================
    num_correct, num_correct_majvote, num_correct_limit, num_tested = 0, 0, 0, 0
    total_num_candidates = 0
    
    
    with open(discriminate_results_file, 'w', encoding='utf-8') as outfile:
        for qid in query_ids:
            final_solutions_file = f"{answer_sheets_dir}/{qid}/final_solutions.jsonl"
            
            
            trace_js = read_jsonl(final_solutions_file)  
            if args.cutoff_rollout > -1:
                trace_js = [s for s in trace_js if s["rollout_id"] <= args.cutoff_rollout]  
            
            user_question = trace_js[0]["trace"]["0"]["user_question"]
            ground_truth = trace_js[0]["trace"]["0"]["ground_truth"]
            evaluator.question = user_question
            
            all_candidates = []
            solution_trace_dic = {}
            for id, s in enumerate(trace_js):
                trace = s["trace"] if "trace" in s else s
                solution_trace, final_step, _, reward = concat_solution_trace(trace) # TODO
                if solution_trace in solution_trace_dic:
                    solution_trace_dic[solution_trace]["freq"] = solution_trace_dic[solution_trace]["freq"] + 1
                    solution_trace_dic[solution_trace]["reward"] = (
                        solution_trace_dic[solution_trace]["reward"] + reward
                    )
                    if len(solution_trace_dic[solution_trace]["final_step"]) < len(final_step):
                        solution_trace_dic[solution_trace]["final_step"] = final_step
                else:
                    solution_trace_dic[solution_trace] = {"freq": 1, "reward": reward, "final_step": final_step}

            for solution_trace in solution_trace_dic.keys():
                final_step = solution_trace_dic[solution_trace]["final_step"]
                trace_freq = solution_trace_dic[solution_trace]["freq"]
                trace_reward = solution_trace_dic[solution_trace]["reward"]

                masked_solution_trace_list = rag_mask_solution_trace( # TODO
                    solution_trace,
                    num_return=args.num_masked_solution_traces,
                    left_boundary=args.mask_left_boundary,
                    right_boundary=args.mask_right_boundary,
                )
                final_answer = final_step.strip()
                # print(f"{final_answer} -> {trace_reward}")
                # print(masked_solution_trace_list)


                candidate = Candidate(
                    solution_trace,
                    deepcopy(masked_solution_trace_list),
                    final_step,
                    final_answer,
                    id,
                    trace_freq,
                    trace_reward,
                )
                all_candidates.append(candidate)
            
            # for can in all_candidates:
            #     print(can.to_dict())
            
            answer2candidates, answer2confidence, _ = group_candidates_by_answer(
                all_candidates, evaluator, args.rc_criteria
            )
            most_confident_answer = max(answer2candidates.keys(), key=lambda x: answer2confidence[x])
            highest_confidence = answer2confidence[most_confident_answer]
            assert highest_confidence > 0
            # -------------------------------------------------------------------------

            candidates = all_candidates  #! exhaustive
            total_num_candidates += len(candidates)

            # ------ Get winner answer ------
            if highest_confidence > args.threshold:
                print("You are very confident. Skipping...")
                winner_answer = most_confident_answer
            else:
                winner_candidate = discriminator.select(
                    user_question,
                    candidates,
                    gt_answer=ground_truth,
                    # aux={"file_idx": file_idx, "problem_id": qid},
                )
                if winner_candidate is not None:
                    winner_answer = winner_candidate.final_answer
                else:
                    winner_answer = most_confident_answer
            
            
            item = {
                "qid": qid,
                "query": user_question,
                "gt_answers": ground_truth,
                "pred_answer": winner_answer,
                "conf": answer2confidence[winner_answer]
            }
            outfile.write(json.dumps(item) + '\n')
            
            # print(winner_answer)

            # 
            # with open(discriminate_results_file, "w") as f:
            #     json.dump(temp_recording, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--dataset', type=str, default='wikimultihopqa', choices=[
        'wikimultihopqa', 'hotpotqa', 'musique', 'iirc', 'multihop_rag',
        'nqgold', 'trivia', 'popqa',
        'factscore'
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--retriever_model', type=str, default='bm25', choices=[
        'positive', 'negative', 'bm25', 'contriever', 'rerank', 'bge_m3', 'sgpt'
    ])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=0.008)
    parser.add_argument('--fewshot', type=int, default=6)
    parser.add_argument("--bm25_k1", type=float, default=0.9)
    parser.add_argument("--bm25_b", type=float, default=0.4)
    parser.add_argument('--retrieve_topk', type=int, default=3)
    parser.add_argument('--generate_max_length', type=int, default=64)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--run', type=str, default='run_1 (+a4)')
    parser.add_argument("--seed", type=int, default=10)
    
    # MCTS ---
    parser.add_argument("--verbose", action="store_true", help="extra login")
    parser.add_argument("--mcts_discount_factor", type=float, default=1.0)
    parser.add_argument("--mcts_exploration_weight", type=float, default=2.0)
    parser.add_argument("--mcts_weight_scheduler", choices=["exp", "lin", "const"], default="const")
    parser.add_argument("--save_tree", action="store_true")
    parser.add_argument("--num_rollouts", type=int, default=12)
    parser.add_argument("--max_depth_allowed", type=int, default=5)
    parser.add_argument("--num_votes", type=int, default=1)
    parser.add_argument("--mcts_num_last_votes", type=int, default=10)
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
    
    # === Prompt files =============
    args.semantic_equivalence_prompt_file = "prompts_mcts/semantic_equivalence_prompt_template.txt"
    
    # === Define CUDA device =======
    args.output_dir = f"run_output/{args.run}" 
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
    
    
    # python run_mcts/mcts_discrimination.py
    