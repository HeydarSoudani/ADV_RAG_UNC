#!/usr/bin/env python3

import re
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import random
import argparse
import numpy as np
import transformers
from accelerate import Accelerator
from accelerate.utils import gather_object
import glob

from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Tuple
from run_mcts.src.generate_node import Generator
from copy import deepcopy

from utils.general_utils import set_seed, read_jsonl, read_txt
from run_searchr1.retrieval_local import BM25Retriever, ContrieverRetriever, RerankRetriever, DenseRetriever
from run_rag_methods.src.correctness import em_score, f1_score
from run_searchr1.inference import get_think, get_query, get_answer, _passages2string, StopOnSequence
from run_mcts.discriminators.searchr1_as_disc import SemanticEquivalenceGenerator
from run_mcts.src.generate_paraphrase import SearchQueryGenerator, ThinkGenerator, get_paraphrased_query, get_paraphrased_think
from run_mcts.sr1_critique_discrimination import _filter_long, _filter_none, _filter_specific_words, _filter_white_space


# ==== Functions 
class Candidate:
    def __init__(
        self,
        trace_obj,
        final_answer,
        trace_id,
        masked_trace_retrieval_list=None,
        trace_reward=1.0,
        trace_freq=1,
        c_type="default",
    ):
        self.trace_obj = trace_obj
        self.masked_trace_retrieval_list = masked_trace_retrieval_list
        self.final_answer = final_answer
        self.trace_id = trace_id
        self.trace_reward = trace_reward
        self.trace_freq = trace_freq
        self.c_type = c_type
        # self.rag_confidence = self.get_rag_confidence()

    def __str__(self):
        return f"Candidate {self.trace_id}: {self.final_answer} | {self.rag_confidence}"

    def get_rag_confidence(self):
        sorted_keys = sorted(self.trace_obj.keys(), key=int)
        preds = [item[sorted_keys[-1]]['think_answer']['answer'] for item in self.masked_trace_retrieval_list]
        self.rag_confidence = sum(1 for item in preds if item is not None and em_score(item, self.final_answer)) / len(preds)
        return self.rag_confidence

    def to_dict(self):
        return {
            "trace_id": self.trace_id,
            "trace_reward": self.trace_reward,
            "trace_freq": self.trace_freq,
            "final_answer": self.final_answer,
            "trace_obj": self.trace_obj,
            "masked_trace_retrieval_list": self.masked_trace_retrieval_list
        }

    def to_search_queries(self):
        sorted_keys = sorted(self.trace_obj.keys(), key=int)
        if len(sorted_keys) == 2:
            org = self.trace_obj[sorted_keys[-1]]['think_answer']['think']
            para = [item[sorted_keys[-1]]['think_answer']['think'] for item in self.masked_trace_retrieval_list]    
        else:
            org = self.trace_obj[sorted_keys[-2]]['think_search']['search_query']
            para = [item[sorted_keys[-2]]['think_search']['search_query'] for item in self.masked_trace_retrieval_list]
        return f"Candidate {self.trace_id}:\nOriginal: {org}\n{para}"
        
    def to_prediction(self):
        sorted_keys = sorted(self.trace_obj.keys(), key=int)
        preds = [item[sorted_keys[-1]]['think_answer']['answer'] for item in self.masked_trace_retrieval_list]
        return f"Candidate {self.trace_id}: {self.final_answer} | {preds}"

def ragc_discrimination(args):
    # === MultiGPU setup =======================
    accelerator = Accelerator()
    device = accelerator.device
    
    if accelerator.is_main_process:
        print("\n== RAG-Consistency Discrimination ...")
        print(f"""
            Model name:  {args.model_name_or_path}
            Dataset:     {args.dataset} / {args.subsec} ({args.fraction_of_data_to_use})
            Retriever:   {args.retriever_name} / ({args.retrieval_model_path})
            Seed:        {args.seed}
            Run:         {args.run}
        """.replace('        ', ''))

        # === Define CUDA device =======
        # args.device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"Number of available GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA is not available. No GPUs detected.")
        print('\n')    

    # === Load files ===========================
    entries = os.listdir(args.generation_trees_results_dir)
    query_ids = [entry for entry in entries if os.path.isdir(os.path.join(args.generation_trees_results_dir, entry))]
    sorted_query_ids = sorted(query_ids, key=lambda x: int(x.split('_')[1]))
    # sorted_query_ids = ['dev_101', 'dev_110', 'dev_207', 'dev_351', 'dev_999', 'dev_1053', 'dev_4480', 'dev_5311', 'dev_5474', 'dev_5710', 'dev_6273', 'dev_6276', 'dev_6523', 'dev_6692', 'dev_6719']
    # sorted_query_ids = ['dev_4480', 'dev_5311', 'dev_5474', 'dev_5710', 'dev_6273']
    # sorted_query_ids = ['dev_101', 'dev_110', 'dev_207', 'dev_351']
    
    # === Read existing data ===================
    generated_qids = []
    generated_em = []
    if os.path.exists(args.discriminate_results_file):
        with open(args.discriminate_results_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'qid' in data:
                    generated_qids.append(data['qid'])
                    generated_em.append(data['em'])
    generated_qids = set(generated_qids)
    filtered_list = [item for item in sorted_query_ids if item not in generated_qids]

    # === generator Model ======================
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    generator = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16).to(device)
    target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
    stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])
    curr_eos = [151645, 151643] # for Qwen2.5 series models
    curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

    # === Static Retriever =====================
    if args.retriever_name == 'bm25':
        retriever = BM25Retriever(args)  
    elif args.retriever_name == 'contriever':
        retriever = ContrieverRetriever(args)
    elif args.retriever_name in ['rerank_l6', 'rerank_l12']:
        retriever = RerankRetriever(args)
    elif args.retriever_name in ['e5', 'bge']:
        retriever = DenseRetriever(args)
        
    # === Other models =========================
    se_model = SemanticEquivalenceGenerator(args, generator, tokenizer)
    node_generator = Generator(args, retriever, generator, tokenizer)
    sq_generator = SearchQueryGenerator(args, generator, tokenizer)
    think_generator = ThinkGenerator(args, generator, tokenizer)

    # === Functions ============================
    def group_candidates_by_answer(se_model, question:str, candidates: list[Candidate], criteria="freq"):
        """Return answer2candidates, answer2confidence, answer2cnt."""
        answer2candidates = {}
        answer2confidence = defaultdict(float)
        answer2cnt = defaultdict(int)

        for c in candidates:
            has_existed = False
            for existing_answer in answer2candidates.keys():
                if se_model.check_answers_equiv(question, c.final_answer, existing_answer):
                    has_existed = True
                    answer2candidates[str(existing_answer)].extend([c] * c.trace_freq)
                    answer2confidence[str(existing_answer)] += c.trace_reward if criteria == "reward" else c.trace_freq
                    answer2cnt[str(existing_answer)] += c.trace_freq
                    break

            if not has_existed:
                if str(c.final_answer) in answer2candidates:
                    answer2candidates[str(c.final_answer)].extend([c] * c.trace_freq)
                else:
                    answer2candidates[str(c.final_answer)] = [c] * c.trace_freq
                answer2confidence[str(c.final_answer)] += c.trace_reward if criteria == "reward" else c.trace_freq
                answer2cnt[str(c.final_answer)] += c.trace_freq

        assert all(answer2cnt[ans] == len(answer2candidates[ans]) for ans in answer2cnt.keys())
        # assert float(sum([candidate.trace_reward for candidate in candidates])) == float(
        #     sum([answer2confidence[ans] for ans in answer2confidence.keys()])
        # )

        candidates_count = sum([candidate.trace_freq for candidate in candidates])
        for ans in answer2confidence.keys():
            answer2confidence[ans] /= candidates_count

        return answer2candidates, answer2confidence, answer2cnt

    def rag_mask_solution_trace(trace: Dict[int, Dict[str, str]], num_return: int = 5) -> List[Dict[int, Dict[str, str]]]:
        
        # --- Functions
        def get_sorted_keys(trace):
            return sorted(trace.keys(), key=int)

        def build_think_answer_entry(think, answer, value):
            return {
                "think_answer": {
                    "think": think,
                    "answer": answer,
                    "value": value
                }
            }

        def process_with_search(sorted_keys, final_value):
            # TODO: more randomness in this function
            
            print(trace)
            print('-------')
            think_search_indices = [k for k, v in trace.items() if "think_search" in v]
            think_answer_index = list(trace.keys())[-1]
            print(think_search_indices)
            print(think_answer_index)
            print('-------')
            
            paraphrased_traces = []
            selected_indices = random.choices(think_search_indices, k=5)  # change to random.sample(...) if needed
            selected_indices_group = [(x, selected_indices.count(x)) for x in sorted(set(selected_indices))]

            print(selected_indices)
            print('-------')
            
            
            for (selected_index, repeat) in selected_indices_group:
                selected_int = int(selected_index)
                original_sq = trace[selected_int]["think_search"].get('search_query', '')
                print(original_sq)
                print('-------')
                
                if original_sq:
                    sq_prompt = sq_generator.get_instruction(original_sq, n=repeat)
                    sq_output = sq_generator.generate(sq_prompt, temperature=1.0)[0]
                    paraphrased_queries = get_paraphrased_query(sq_output) # TODO: check if it's not None

                    print(paraphrased_queries)
                    print('-------')  
                
                for paraphrased_query in paraphrased_queries:
                    new_trace = {}
   
                    # Keep steps before and including the selected one
                    for i in range(selected_int):
                        new_trace[i] = deepcopy(trace[i])
            
                    retrieved_docs = retriever.search(paraphrased_query) if paraphrased_query else []    
                    new_trace[selected_int] = {
                        "think_search": {
                            "think": trace[selected_int]["think_search"].get('think', ''),
                            "search_query": paraphrased_query,
                            "retrieved_documents": retrieved_docs,
                        }
                    }
                
                    # Next think_search steps
                    for i in range(selected_int+1, think_answer_index):
                        thinks, search_query, ret_docs = node_generator.generate_think_search(new_trace)
                        new_trace[i] = {
                            "think_search": {
                                "think": thinks,
                                "search_query": search_query,
                                "retrieved_documents": ret_docs
                            }
                        }
                    
                    # Last step
                    think, most_likely_answer, reward, _ = node_generator.generate_think_answer(new_trace)
                    new_trace[think_answer_index] = {
                        "think_answer": {
                            "think": think,
                            "answer": most_likely_answer,
                            "node_reward": reward,
                            "scores": (0.0, 0.0, 0.0)
                        }
                    }
                    paraphrased_traces.append(new_trace)
                
            
            return paraphrased_traces

        def process_without_search(final_value):
            original_think = trace[1]["think_answer"].get('think', '')
            paraphrased_thinks = []
            if original_think:
                think_prompt = think_generator.get_instruction(original_think, n=num_return)
                think_output = think_generator.generate(think_prompt)[0]
                paraphrased_thinks = get_paraphrased_think(think_output)

            input_text = node_generator.get_prompt_text('think_answer', {0: trace[0]})
            result = []
            for pt in paraphrased_thinks:
                input_text_pt = input_text + f"<think> {pt} </think>\n"
                output = node_generator.generate_(input_text_pt, node_generator.answer_stopping_criteria)[0]
                result.append({
                    0: trace[0],
                    1: build_think_answer_entry(pt, get_answer(output), final_value)
                })
            return result

        # --- Main execution
        sorted_keys = get_sorted_keys(trace)
        final_value = trace[sorted_keys[-1]]['think_answer']["value"]

        # Check if there is any think_search
        has_search = any("think_search" in trace[k] for k in sorted_keys)
        return process_with_search(sorted_keys, final_value) if has_search else process_without_search(final_value)

    # === Inference ============================
    filter_words = {"unknown", "n/a", "none", "not enough information provided"}
    em_evaluation = generated_em
    generator.eval()
    accelerator.wait_for_everyone() # sync GPUs
    with accelerator.split_between_processes(filtered_list) as sorted_query_ids_shard:
        
        # results = []
        ranked_discriminate_results_file = f"{args.discriminate_results_dir}/ragc_discriminate_results_rank{accelerator.process_index}.jsonl"
        with open(ranked_discriminate_results_file, "w") as ranked_f:
            for idx, qid in enumerate(tqdm(sorted_query_ids_shard, desc=f"[Rank {accelerator.process_index}]")):
                final_solutions_file = f"{args.generation_trees_results_dir}/{qid}/final_solutions.jsonl"
                all_traces = read_jsonl(final_solutions_file)  
                gt_answers = all_traces[0]["trace"]["0"]["ground_truth"]
                question = all_traces[0]["trace"]["0"]["user_question"]
                question = question.strip()
                if question[-1] != '?':
                    question += '?'
                
                if idx == 6:
                    break
                
                all_candidates = []
                for trace_id, trace in enumerate(all_traces):
                    trace_ = trace["trace"]
                    trace_ = {int(key): val for key, val in  trace_.items()}
                    last_depth_key = list(trace_.keys())[-1]
                    last_node_type = list(trace_[last_depth_key].keys())[0] 
                    final_answer_reward = trace_[last_depth_key][last_node_type]["value"]
                    final_answer = trace_[last_depth_key][last_node_type]["answer"]
                    
                    if final_answer is None:
                        continue
                    elif len(final_answer) >= 80:
                        continue
                    elif final_answer.strip() == '':
                        continue
                    elif final_answer.strip().lower() in filter_words:
                        continue
                    
                    candidate = Candidate(trace_, final_answer, trace_id, trace_reward=final_answer_reward)
                    all_candidates.append(candidate)
                    # print(candidate)
                    # print(candidate.to_search_queries())
                
                if len(all_candidates) > 0:
                    answer2candidates, answer2confidence, _ = group_candidates_by_answer(
                        se_model, question, all_candidates, args.rc_criteria
                    )
                    most_confident_answer = max(answer2candidates.keys(), key=lambda x: answer2confidence[x])
                    highest_confidence = answer2confidence[most_confident_answer]
                    assert highest_confidence > 0
                
                    item = {
                        "qid": qid,
                        "query": question,
                        "gt_answers": gt_answers
                    }
                
                    if highest_confidence > args.threshold:
                        # print("You are very confident. Skipping...")
                        winner_answer = most_confident_answer if most_confident_answer != None else ''
                        winner_confidence = highest_confidence
                        item["pred_answers"] = [(c.final_answer, winner_confidence) for c in all_candidates]
                    else:
                        for i, c in enumerate(all_candidates):
                            # if i == 1:
                            #     break
                            print('\n\n===============')
                            c.masked_trace_retrieval_list = deepcopy(rag_mask_solution_trace(
                                c.trace_obj,
                                num_return=args.num_masked_solution_traces,
                            ))
                            c.get_rag_confidence()
                            print(c.to_dict())
                            print(f"[Rank {accelerator.process_index}] {c.to_prediction()}")
                        winner_answer, winner_confidence = max(
                            ((c.final_answer, c.rag_confidence) for c in all_candidates),
                            key=lambda x: x[1]
                        )
                        item["pred_answers"] = [(c.final_answer, c.rag_confidence) for c in all_candidates]
                        
                    correctness_em = em_score(winner_answer, gt_answers)
                    em_evaluation.append(correctness_em)
                    
                    item["em"] = correctness_em
                    item["winner_answer"] = winner_answer
                    item["rag_conf"] = winner_confidence
                
                else:
                    correctness_em = 0
                    item = {
                        "qid": qid,
                        "query": question,
                        "gt_answers": gt_answers,
                        "pred_answers": [],
                        "em": 0,
                        "winner_answer": "",
                        "rag_conf": 0.0
                    }
                    
                ranked_f.write(json.dumps(item) + "\n")
                em_evaluation.append(correctness_em)

    # results_gathered = gather_object(results)
    em_evaluation_gathered = gather_object(em_evaluation)
    
    if accelerator.is_main_process:
        # with open(args.discriminate_results_file, "a", encoding="utf-8") as inf_f:
        #     for item in results_gathered:
        #         inf_f.write(json.dumps(item) + "\n")
        
        # --- Print results
        print("\nEvaluation Result:")
        print(em_evaluation_gathered)
        print(f"EM: {np.mean(em_evaluation_gathered)*100}")


def merge_result_files(args):
    shard_files = f"{args.discriminate_results_dir}/ragc_discriminate_results_rank*.jsonl"
    output_file = args.discriminate_results_file

    shard_files = sorted(glob.glob(shard_files))
    with open(output_file, "a") as fout:
        for shard_file in shard_files:
            if shard_file == output_file:
                continue  # Don't append the output file to itself
            with open(shard_file, "r") as fin:
                for line in fin:
                    fout.write(line)
            
            os.remove(shard_file)
            print(f"Deleted shard file: {shard_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
        # Model
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--max_new_tokens', type=int, default=512)
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='bamboogle', choices=[
        'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'
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
    parser.add_argument('--run', type=str, default='run_5 (edited_prompt_roll4)')
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument('--use_counter', action='store_false')
    
    # MCTS ---
    parser.add_argument("--enable_critique", action="store_true", help="")
    parser.add_argument("--enable_doc_generation", action="store_true", help="")
    parser.add_argument("--verbose", action="store_true", help="extra login")
    parser.add_argument("--mcts_discount_factor", type=float, default=1.0)
    parser.add_argument("--mcts_exploration_weight", type=float, default=2.0)
    parser.add_argument("--mcts_weight_scheduler", choices=["exp", "lin", "const"], default="const")
    parser.add_argument("--save_tree", action="store_true")
    parser.add_argument("--num_rollouts", type=int, default=4)
    parser.add_argument("--max_depth_allowed", type=int, default=4)
    parser.add_argument("--num_votes", type=int, default=2)
    parser.add_argument("--mcts_num_last_votes", type=int, default=5)
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
    parser.add_argument("--extend_rc_mode", type=str, default="majority_vote", choices=["original", "BoN", "majority_vote"])
    parser.add_argument("--best_of", type=int, default=5)
    
    args = parser.parse_args()
    
    # === Files ====================
    model_ = args.model_name_or_path.split('/')[-1]
    output_dir = f"run_output/{args.run}/{model_}/{args.dataset}_{args.subsec}/{args.retriever_name}"
    args.generation_trees_results_dir = f'{output_dir}/generation_trees'
    args.discriminate_results_dir = f"{output_dir}"
    args.discriminate_results_file = f"{output_dir}/ragc_discriminate_results.jsonl"
    os.makedirs(args.generation_trees_results_dir, exist_ok=True)
    
    args.semantic_equivalence_prompt_file = "prompts_mcts/semantic_equivalence_prompt_template.txt"
    
    ### === Run Steps =============
    set_seed(args.seed)
    ragc_discrimination(args)
    # merge_result_files(args)
    
    # python run_mcts/ragc_discrimination.py
    # accelerate launch --multi_gpu --num_processes 2 run_mcts/ragc_discrimination.py
    # accelerate launch --multi_gpu run_mcts/ragc_discrimination.py
















            # last_search_index, last_think_search = None, None
            # for key in sorted_keys:
            #     if "think_search" in trace[key]:
            #         last_search_index = key
            #         last_think_search = trace[key]["think_search"]

            # if last_think_search is None:
            #     return []

            # original_sq = last_think_search.get('search_query', '')
            # before_think_search = {k: trace[k] for k in sorted_keys if k < last_search_index}
            # input_text = node_generator.get_prompt_text('think_answer', before_think_search)
            # input_text += f"<think> {last_think_search['think']} </think>\n"

            # paraphrased_queries = []
            # if original_sq:
            #     sq_prompt = sq_generator.get_instruction(original_sq, n=num_return)
            #     sq_output = sq_generator.generate(sq_prompt)[0]
            #     paraphrased_queries = get_paraphrased_query(sq_output)

            # result = []
            # for pq in paraphrased_queries:
            #     retrieved_docs = retriever.search(pq)
            #     input_text_pq = input_text + f"<search> {pq} </search>\n"
            #     input_text_pq += f"<information> {_passages2string(retrieved_docs)}</information>\n"
            #     output = node_generator.generate_(input_text_pq, node_generator.answer_stopping_criteria)[0]

            #     trace_copy = deepcopy(before_think_search)
            #     trace_copy[last_search_index] = {
            #         "think_search": {
            #             "think": last_think_search['think'],
            #             "search_query": pq,
            #             "retrieved_documents": retrieved_docs
            #         }
            #     }
            #     trace_copy[last_search_index + 1] = build_think_answer_entry(get_think(output), get_answer(output), final_value)
            #     result.append(trace_copy)



# === Define CUDA device =======
# args.device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
# if torch.cuda.is_available():
#     print(f"Number of available GPUs: {torch.cuda.device_count()}")
#     for i in range(torch.cuda.device_count()):
#         print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
# else:
#     print("CUDA is not available. No GPUs detected.")

# def merge_results():
#     def extract_qid_number(qid):
#         match = re.search(r'\d+', qid)
#         return int(match.group()) if match else -1

#     input_files = glob.glob(f"{args.discriminate_results_dir}/ragc_discriminate_results_*.jsonl")
#     output_file = f"{args.discriminate_results_dir}/ragc_discriminate_results.jsonl"
    
#     all_rows = []
#     for file in input_files:
#         with open(file, "r") as f:
#             for line in f:
#                 item = json.loads(line)
#                 all_rows.append(item)
#     all_rows.sort(key=lambda x: extract_qid_number(x.get("qid", "")))

#     with open(output_file, "w") as f:
#         for item in all_rows:
#             f.write(json.dumps(item) + "\n")

    # challenging_samples = []
    # Split to multi-GPUs
    # rank = accelerator.process_index
    # world_size = accelerator.num_processes
    # local_sorted_query_ids = sorted_query_ids[rank::world_size]
    
    # with open(f"{args.discriminate_results_dir}/ragc_discriminate_results_{rank}.jsonl", 'w', encoding='utf-8') as inf_file:
   
    # def rag_mask_solution_trace(trace: Dict[int, Dict[str, str]], num_return:int=5):
    #     sorted_keys = sorted(trace.keys(), key=int)
    #     final_value = trace[sorted_keys[-1]]['think_answer']["value"]
    #     masked_traces_list = []
        
    #     # --- Find the last think_search
    #     last_search_index = None
    #     last_think_search = None
    #     for key in sorted(trace.keys(), key=int):
    #         if "think_search" in trace[key]:
    #             last_search_index = key
    #             last_think_search = trace[key]["think_search"]
        
    #     # ---
    #     if last_search_index:
    #         original_sq = last_think_search['search_query']
            
    #         before_think_search = {}
    #         for key in sorted_keys[:int(last_search_index)]:
    #             before_think_search[int(key)] = trace[key]

    #         input_text = node_generator.get_prompt_text('think_answer', before_think_search)
    #         input_text += f"<think> {last_think_search['think']} </think>\n"
            
    #         if original_sq:
    #             sq_prompt = sq_generator.get_instruction(original_sq, n=num_return)
    #             sq_output = sq_generator.generate(sq_prompt)[0]
    #             # print(sq_output)
    #             paraphrased_queries = get_paraphrased_query(sq_output)
    #             # print(paraphrased_queries)
    #         else:
    #             paraphrased_queries = []

    #         for pq in paraphrased_queries:
    #             retrieved_docs = retriever.search(pq)
    #             input_text_pq = input_text + f"<search> {pq} </search>\n"
    #             input_text_pq += f"<information> {_passages2string(retrieved_docs)}</information>\n"
    #             output = node_generator.generate_(input_text_pq, node_generator.answer_stopping_criteria)[0]
                
    #             tmp = deepcopy(before_think_search)
    #             tmp[last_search_index] = {
    #                 "think_search": {
    #                     "think": last_think_search['think'],
    #                     "search_query": pq,
    #                     "retrieved_documents": retrieved_docs
    #                 }
    #             }
    #             tmp[last_search_index+1] = {
    #                 "think_answer": {
    #                     "think": get_think(output),
    #                     "answer": get_answer(output),
    #                     "value": final_value
    #                 }
                    
    #             }
    #             masked_traces_list.append(tmp)
            
    #     # If there is no retirval in the trace
    #     else:
    #         original_think = trace[1]["think_answer"]['think']
    #         if original_think:
    #             think_prompt = think_generator.get_instruction(original_think, n=num_return)
    #             think_output = think_generator.generate(think_prompt)[0]
    #             paraphrased_thinks = get_paraphrased_think(think_output)
    #         else:
    #             paraphrased_thinks = []
                
    #         input_text = node_generator.get_prompt_text('think_answer', {0: trace[0]})
    #         for pt in paraphrased_thinks:
    #             input_text_pt = input_text + f"<think> {pt} </think>\n"
    #             output = node_generator.generate_(input_text_pt, node_generator.answer_stopping_criteria)[0]
    #             masked_traces_list.append({
    #                 0: trace[0],
    #                 1: {
    #                     "think_answer": {
    #                         "think": pt,
    #                         "answer": get_answer(output),
    #                         "value": final_value
    #                     }
    #                 }
    #             })
                
    #     return masked_traces_list
                
   