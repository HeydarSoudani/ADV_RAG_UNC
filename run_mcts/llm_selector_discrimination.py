#!/usr/bin/env python3

import re
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import logging
import argparse
import numpy as np
import transformers
from accelerate import Accelerator
from accelerate.utils import gather_object
import glob
import random

from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Tuple
from src_mcts.generate_node import Generator
from copy import deepcopy

from utils.general_utils import set_seed, read_jsonl, read_txt
from run_searchr1.correctness import normalize_answer, em_score, f1_score
from run_searchr1.inference import get_think, get_query, get_answer, _passages2string, StopOnSequence
from searchr1_discrimination import SemanticEquivalenceGenerator
from run_mcts.sr1_critique_discrimination import _filter_long, _filter_none, _filter_specific_words, _filter_white_space



class CandidateSelector:
    """Select one Candidate"""
    def __init__(self, args, generator, tokenizer) -> None:
        self.args = args
        self.generator = generator
        self.tokenizer = tokenizer
        self.eos_token_ids = self.generator.config.eos_token_id
        self.semantic_equivalence_prompt = read_txt(self.args.semantic_equivalence_prompt_file)

    def generate(self,
        input_text,
        max_new_tokens=1024,
        num_return:int = 1,
        temperature:float = 0.7,
        do_sample:bool = True
    ):
        if self.tokenizer.chat_template:
            input_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": input_text}],
                add_generation_prompt=True,
                tokenize=False
            )
        
        input_ids = self.tokenizer.encode(input_prompt, return_tensors='pt').to(self.generator.device)
        attention_mask = torch.ones_like(input_ids)
        
        generated_texts = []
        for i in range(num_return):
            with torch.no_grad():
                outputs = self.generator.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    temperature=temperature,
                    do_sample=do_sample,
                )
                generated_tokens = outputs[0][input_ids.shape[1]:]
                output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                generated_texts.append(output_text)
        
        return generated_texts

    def get_input_prompt(self, question, docs, candidates):
        # input_text = ''
        # input_text += 'You are an answer selector in a question-answering task. '
        # input_text += 'Your goal is to select the correct answer from a list of candidates.\n'
        # input_text += 'The process consists of two steps:\n'
        # input_text += '1. Carefully analyze and reason over the retrieved documents and your internal knowledge.\n'
        # input_text += '2. Select the final answer from the provided candidates.\n'
        # input_text += 'You must think deeply and consider all retrieved documents thoroughly before deciding.\n'
        # input_text += 'Retrieved documents (if any) will be provided between <information> and </information> tags.\n'
        # input_text += 'Each answer candidate will appear between <candidate> and </candidate> tags.\n'
        # input_text += 'Your reasoning must be enclosed in exactly one pair of <think> and </think> tags.\n'
        # input_text += 'You MUST choose one of the given candidates as the final answer. Do NOT generate a new answer.\n'
        # input_text += 'Your output must include ONLY the following tags, in this exact order:\n'
        # input_text += '<think> your complete, thoughtful reasoning here </think>\n'
        # input_text += '<answer> selected answer here </answer>\n'
        # input_text += 'Do NOT include any additional text, explanations, or formatting outside of these tags.\n'

        # input_text = ''
        # input_text += 'You are an answer selector for a question-answering task.\n'
        # input_text += 'Your job has two steps:\n'
        # # input_text += '1. Deeply reason over the retrieved documents and your internal knowledge.\n'
        # input_text += '1. Carefully analyze each retrieved document one by one, along with your internal knowledge.\n'
        # input_text += '2. Select the correct answer from the provided candidates.\n'
        # input_text += 'Retrieved documents (if any) are enclosed in <information> tags.\n'
        # input_text += 'Answer candidates are enclosed in <candidate> tags.\n'
        # input_text += 'All reasoning must be inside a single <think>...</think> block.\n'
        # input_text += 'You MUST choose exactly one answer from the candidates. Do NOT generate new answers.\n'
        # input_text += 'Choose the most precise and complete candidate.\n'
        # input_text += 'Output ONLY the following tags in this exact order:\n'
        # input_text += '<think> your reasoning here </think>\n'
        # input_text += '<answer> your selected answer here </answer>\n'
        # input_text += 'Do NOT include anything outside these tags. No introductions, no formatting, no extra text.\n'

        input_text = ''
        input_text += 'You are an answer selector for a question-answering task.\n'
        input_text += 'Your task has two steps:\n'
        # input_text += '1. Analyze each retrieved document carefully, along with your internal knowledge.\n'
        input_text += '1. Deeply analyze each retrieved document one by one, along with your internal knowledge.\n'
        input_text += '2. Select the most precise answer from the provided candidates.\n'
        input_text += 'Retrieved documents (if any) are enclosed in <information> tags.\n'
        input_text += 'Answer candidates are enclosed in <candidate> tags.\n'
        input_text += 'All reasoning must be placed inside a single <think>...</think> block.\n'
        input_text += 'You MUST select exactly one of the provided candidates. Do NOT generate new answers.\n'
        input_text += 'Your output must contain ONLY the following tags, in this exact order:\n'
        input_text += '<think> your reasoning here </think>\n'
        input_text += '<answer> your selected answer here </answer>\n'
        input_text += 'Do NOT include any additional text, introductions, or formatting outside these tags.\n'



        if len(docs) > 0:
            input_text += f"\n<information>\n"
            for idx, (doc_id, doc_contents) in enumerate(docs.items()):
                title = doc_contents.split("\n")[0]
                text = "\n".join(doc_contents.split("\n")[1:])
                input_text += f"Doc {idx+1} (Title: {title}) {text}\n"
            input_text += f"</information>\n"
        
        input_text += f'\nQuestion: {question}\n'
        for c in candidates:
            input_text += f'<candidate> {c} </candidate>\n'
        
        return input_text

    def select(self, qid, question, docs, candidates):
        input_prompt = self.get_input_prompt(question, docs, candidates)
        # print(input_prompt)
        initial_output_list = self.generate(input_prompt, num_return=1, temperature=1.0)
        
        think_list, answer_list = [], []
        for initial_output in initial_output_list:
            think = get_think(initial_output)
            final_answer = get_answer(initial_output) 
        
            if think == '':
                print(f"Think is not provided for query {qid}")
                for i in range(self.args.retry):
                    print(f"Think, try {i+1} ...")
                    new_output = self.generate(input_prompt, num_return=1, temperature=1.0)[0]
                    think = get_think(new_output)
                    if think != '':
                        final_answer = get_answer(new_output) 
                        break
            think_list.append(think)
            answer_list.append(final_answer)
        
        most_likely_answer, conf_value = self._get_most_likely_answer(user_query=question, output_list=answer_list)
        
        return most_likely_answer, conf_value

    def _get_most_likely_answer(self, user_query: str, output_list: List[str]):
        assert len(output_list) > 0

        def cluster_by_meaning(user_query, output_list):
            cluster = []

            for i, answer in enumerate(output_list):
                if i == 0:
                    cluster.append([answer])
                else:
                    prompt = self.semantic_equivalence_prompt
                    prompt += f'\n\nWe are evaluating answers to the question: {user_query}\n'
                    prompt += 'Here are two possible answers:\n'

                    for j, c in enumerate(cluster):
                        tmp_prompt = prompt + f'Possible Answer 1: {answer}\n'
                        tmp_prompt += f'Possible Answer 2: {c[0]}\n'
                        tmp_prompt += 'For this question, is Possible Answer 1 semantically equivalent to Possible Answer 2? Respond with Yes or No.\n'
                        tmp_prompt += 'Response: '
                        
                        response = self.generate(
                            tmp_prompt,
                            max_new_tokens=1,
                            num_return=1,
                            # temperature=0.01,
                        )[0]
                        if 'Yes' in response:
                            c.append(answer)
                            break
                        elif j == len(cluster) - 1:
                            cluster.append([answer])
                            break

            return cluster

        if len(output_list) == 1:
            most_confident_answer = output_list[0]
            confidence = 1
        else:
            cluster = cluster_by_meaning(user_query=user_query, output_list=output_list)
            most_confident_cluster = sorted(cluster, key=len, reverse=True)[0]
            most_confident_answer, confidence = most_confident_cluster[0], len(most_confident_cluster)/sum(map(len, cluster))
            assert confidence > 0 and confidence <= 1

        return most_confident_answer, confidence

def llm_selector_discrimination(args):
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

        # --- Define CUDA device
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
    # filtered_list = ['test_4', 'test_24','test_46','test_47','test_51','test_89','test_91','test_93','test_115'] # bamboogle


    # === generator Model ======================
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    generator = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16).to(device)
    se_model = SemanticEquivalenceGenerator(args, generator, tokenizer)
    selector = CandidateSelector(args, generator, tokenizer)
    
    # === Functions ============================
    def group_candidates_by_answer(se_model, question:str, candidates: list[str]):
        """Return answer2candidates"""
        answer2candidates = {}
        for c in candidates:
            has_existed = False
            for existing_answer in answer2candidates.keys():
                if se_model.check_answers_equiv(question, c, existing_answer):
                    has_existed = True
                    answer2candidates[str(existing_answer)].append(c)
                    break

            if not has_existed:
                if str(c) in answer2candidates:
                    answer2candidates[str(c)].append(c)
                else:
                    answer2candidates[str(c)] = [c]
        return answer2candidates
    
    def group_candidates_by_answer_em(candidates: list[str]):
        """Return answer2candidates"""
        answer2candidates = {}
        for c in candidates:
            norm_c = normalize_answer(c)
            has_existed = False
            for existing_answer in answer2candidates.keys():
                if norm_c == existing_answer:
                    has_existed = True
                    answer2candidates[str(existing_answer)].append(norm_c)
                    break

            if not has_existed:
                if str(norm_c) in answer2candidates:
                    answer2candidates[str(norm_c)].append(norm_c)
                else:
                    answer2candidates[str(norm_c)] = [norm_c]
        return answer2candidates

    # === Inference ============================
    em_evaluation = generated_em
    generator.eval()
    accelerator.wait_for_everyone()
    with accelerator.split_between_processes(filtered_list) as sorted_query_ids_shard:
        ranked_discriminate_results_file = f"{args.discriminate_results_dir}/llm_selector_discriminate_rank{accelerator.process_index}.jsonl"
        with open(ranked_discriminate_results_file, "w") as ranked_f:
            for idx, qid in enumerate(tqdm(sorted_query_ids_shard, desc=f"[Rank {accelerator.process_index}]")):
                print('\n------')
                # if idx == 20:
                #     break
                
                final_solutions_file = f"{args.generation_trees_results_dir}/{qid}/final_solutions.jsonl"
                all_traces = read_jsonl(final_solutions_file)  
                gt_answers = all_traces[0]["trace"]["0"]["ground_truth"]
                question = all_traces[0]["trace"]["0"]["user_question"]
                question = question.strip()
                if question[-1] != '?':
                    question += '?'
                print(gt_answers)
                
                docs, answers = {}, []
                for trace_id, trace in enumerate(all_traces):
                    trace_ = trace["trace"]
                    trace_ = {int(key): val for key, val in  trace_.items()}
                    
                    for step_key, step_val in trace_.items():
                        if 'think_search' in step_val:
                            for doc in step_val['think_search']['retrieved_documents']:
                                docs[doc['id']] = doc['contents']
                    
                    last_depth_key = list(trace_.keys())[-1]
                    last_node_type = list(trace_[last_depth_key].keys())[0] 
                    final_answer = trace_[last_depth_key][last_node_type]["answer"]
                    answers.append(final_answer)
                
                filtered_answers = _filter_white_space(_filter_specific_words(_filter_none(_filter_long(answers))))
                if len(filtered_answers) > 0:
                    answer2candidates = group_candidates_by_answer(se_model, question, filtered_answers)  
                    # answer2candidates = group_candidates_by_answer_em(filtered_answers)
                    print(answer2candidates)
                    if len(answer2candidates) == 1:
                        winner_answer = list(answer2candidates.keys())[0].strip()
                        winning_conf = 1.0
                    elif len(answer2candidates) > 1:
                        unq_candidates = [c.strip() for c in list(answer2candidates.keys())]
                        winner_answer, winning_conf = selector.select(qid, question, docs, unq_candidates)
                    else:
                        winner_answer = answer2candidates
                        winning_conf = 1.0
                    
                    correctness_em = em_score(winner_answer, gt_answers)    
                    em_evaluation.append(correctness_em)
                    item = {
                        "qid": qid, "query": question, "gt_answers": gt_answers,
                        "winner_answer": winner_answer, "em": correctness_em,
                        "confidence": winning_conf,
                        "pred_answers": list(answer2candidates.keys())
                    }
                else:
                    item = {
                        "qid": qid, "query": question, "gt_answers": gt_answers,
                        "winner_answer": '', "em": 0, "confidence": 1.0, "pred_answers": []
                    }
                
                ranked_f.write(json.dumps(item) + "\n")
                em_evaluation.append(correctness_em)
    
    em_evaluation_gathered = gather_object(em_evaluation)
    if accelerator.is_main_process:
        print("\nEvaluation Result:")
        # print(em_evaluation_gathered)
        print(f"EM: {np.mean(em_evaluation_gathered)*100}")


def merge_result_files(args):
    shard_files = f"{args.discriminate_results_dir}/llm_selector_discriminate_rank*.jsonl"
    output_file = args.discriminate_results_file

    shard_files = sorted(glob.glob(shard_files))
    with open(output_file, "a") as fout:
        for shard_file in shard_files:
            if shard_file == output_file:
                continue
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
    args.discriminate_results_file = f"{output_dir}/llm_selector_discriminate_results.jsonl"
    os.makedirs(args.generation_trees_results_dir, exist_ok=True)
    
    args.semantic_equivalence_prompt_file = "prompts_mcts/semantic_equivalence_prompt_template.txt"
    
    ### === Run Steps =============
    set_seed(args.seed)
    llm_selector_discrimination(args)
    # merge_result_files(args)
    
    # python run_mcts/llm_selector_discrimination.py
    # accelerate launch --multi_gpu --num_processes 2 run_mcts/llm_selector_discrimination.py
    # accelerate launch --multi_gpu  run_mcts/llm_selector_discrimination.py



    
