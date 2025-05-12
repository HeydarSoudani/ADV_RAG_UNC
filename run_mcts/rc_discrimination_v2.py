#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import math
import torch
import argparse
import numpy as np
from typing import Dict
from copy import deepcopy
from tqdm import tqdm, trange
from collections import defaultdict
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_searchr1.inference import StopOnSequence, get_answer, _passages2string
from run_mcts.searchr1_discrimination import SemanticEquivalenceGenerator
from src_mcts.generate_node import Generator
from run_searchr1.correctness import em_score, normalize_answer
from utils.general_utils import set_seed, read_jsonl
from run_searchr1.retrieval_local import BM25Retriever, ContrieverRetriever, RerankRetriever, DenseRetriever


# ==== Functions 
class Candidate:
    def __init__(
        self,
        trace_text,
        masked_trace_text_list,
        final_answer,
        trace_id,
        trace_reward=1.0,
        trace_freq=1,
        c_type="default",
    ):
        self.trace_text = trace_text
        self.masked_trace_text_list = masked_trace_text_list
        self.final_answer = final_answer
        self.trace_id = trace_id
        self.trace_reward = trace_reward
        self.trace_freq = trace_freq
        self.c_type = c_type

    def __str__(self):
        return f"Candidate {self.trace_id}: {self.final_answer}"

    def to_dict(self):
        return {
            "trace_id": self.trace_id,
            "trace_reward": self.trace_reward,
            "trace_freq": self.trace_freq,
            "final_answer": self.final_answer,
            "trace_text": self.trace_text,
            "masked_trace_text_list": self.masked_trace_text_list
        }

def rag_mask_solution_trace(
    solution_trace_str: str, num_return: int, left_boundary: float, right_boundary: float
) -> list[str]:
    # only mask the reasoning steps behind the retrieved documents
    if num_return == 1:
        interval = 0
    else:
        assert num_return > 1
        assert right_boundary >= left_boundary, f"right_boundary: {right_boundary} < left_boundary: {left_boundary}"
        interval = (right_boundary - left_boundary) / (num_return - 1)

    words_in_solution_trace = solution_trace_str.split(" ")
    
    last_position = next((idx for idx in range(len(words_in_solution_trace) - 1, -1, -1) if "</information>" in words_in_solution_trace[idx]), -1)
    mask_len = len(words_in_solution_trace[last_position+1:])
    
    # Mask the solution trace string from least to most
    masked_solution_traces = []
    for i in range(num_return):
        prefix_part_ratio = left_boundary + i * interval
        prefix_part_num_words = last_position + math.ceil(mask_len * prefix_part_ratio) + 1
        prefix_part_str = " ".join(words_in_solution_trace[:prefix_part_num_words])
        masked_solution_traces.append(prefix_part_str)

    return masked_solution_traces


class Discriminator:
    def __init__(self, args, se_model):
        self.args = args
        self.prompt_instruction = self.get_instruction('think_answer')
        self.se_model = se_model
        answer_target_sequences = ["</answer>", " </answer>", "</answer>\n", " </answer>\n", "</answer>\n\n", " </answer>\n\n"]
        self.answer_stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(answer_target_sequences, self.se_model.tokenizer)])

    def get_instruction(self, node_type):
        input_text = ''
        input_text += 'You are a multi-step reasoner in a question-answering task. '
        input_text += 'Your overall task is to answer the question through a series of intermediate reasoning steps.\n'
        input_text += 'At each step, generate exactly one reasoning step toward answering the question.\n'
        input_text += 'You may use your internal knowledge or retrieved information if needed.\n'
        input_text += 'Retrieved documents, if any, will be provided inside <information> and </information> tags.\n'
        input_text += 'Treat <information> as read-only input. NEVER generate or alter <information> tags yourself.\n'
        input_text += 'NEVER include anything outside the required tags. DO NOT add explanations, introductions, or extra formatting.\n'
        input_text += 'Your output must not contain anything else beyond what is explicitly required.\n\n'

        if node_type == 'think_search':
            input_text += 'You are in the THINK-SEARCH stage.\n'
            input_text += 'Your goal is to identify what specific information is missing and required to move closer to the answer.\n'
            input_text += 'DO NOT attempt to answer the question yet.\n'
            input_text += 'The search query should be precise and focused.\n'
            input_text += 'All reasoning must be enclosed within ONE and ONLY ONE pair of <think> and </think> tags.\n'
            input_text += 'Only include the following tags in this exact order:\n'
            input_text += '<think> one complete reasoning step leading to a search query </think>\n'
            input_text += '<search> search query </search>\n'
        
        elif node_type == 'think_answer':
            input_text += 'You are in the THINK-ANSWER stage.\n'
            input_text += 'Use your internal knowledge and any available <information> content to reason toward the answer.\n'
            input_text += 'Do NOT generate or modify <information> tags in your output.\n'
            input_text += 'Ensure your reasoning is directly connected to the provided information and leads logically to the final answer.\n'
            input_text += 'The final answer must be short, concise, and to the point.\n'
            input_text += 'All reasoning must be enclosed within ONE and ONLY ONE pair of <think> and </think> tags.\n'
            input_text += 'Only include the following tags in this exact order:\n'
            input_text += '<think> one complete reasoning step leading to the final answer </think>\n'
            input_text += '<answer> final answer </answer>\n'
        
        elif node_type == 'critique_search':
            input_text += 'You are in the CRITIQUE-SEARCH stage.\n'
            input_text += 'Your goal is to critically assess both your internal knowledge and the content of the retrieved documents.\n'
            input_text += 'Consider the possibility that these documents may contain inaccuracies, biases, or outdated information.\n'
            input_text += 'Reflect on how these potential issues could affect the reliability of the information provided.\n'
            input_text += 'Based on this critical assessment, formulate a new search query aimed at retrieving alternative or more reliable information sources.\n'
            input_text += 'Formulate a new search query that explores the question from a fresh perspective, utilizing creative strategies like rephrasing, employing synonyms, or considering related concepts.\n'
            input_text += 'All reasoning must be enclosed within ONE and ONLY ONE pair of <critique> and </critique> tags.\n'
            input_text += 'Only include the following tags in this exact order:\n'
            input_text += '<critique> one complete critical assessment and reasoning leading to a new search query </critique>\n'
            input_text += '<search> new search query </search>\n'
        
        elif node_type == 'critique_answer':
            input_text += 'You are in the CRITIQUE-ANSWER stage.\n'
            input_text += 'Your goal is to critically evaluate both your internal knowledge and the content of the retrieved documents.\n'
            input_text += 'Consider the possibility that these documents may contain inaccuracies, biases, or outdated information.\n'
            input_text += 'Reflect on how these potential issues could affect the reliability of the information provided.\n'
            input_text += 'Compare the information from the documents with your internal knowledge to identify any discrepancies or confirmations.\n'
            input_text += 'Based on this critical evaluation, reason carefully toward a new and improved final answer.\n'
            input_text += 'The final answer must be short, concise, and to the point.\n'
            input_text += 'Use <information> content carefully without generating or modifying the tags.\n'
            input_text += 'All reasoning must be enclosed within ONE and ONLY ONE pair of <critique> and </critique> tags.\n'
            input_text += 'Only include the following tags in this exact order:\n'
            input_text += '<critique> one complete critical evaluation and reasoning leading to a new final answer </critique>\n'
            input_text += '<answer> new final answer </answer>\n'
    
        input_text += f'\nQuestion: '
        
        return input_text
        
    def trace2text(self, solution_trace: Dict[int, Dict[str, str]]):
        input_text = ''
        # Path so far
        for item_idx in solution_trace:
            solution_item = solution_trace[item_idx]
            node_keys = list(solution_item.keys())
            node_type = node_keys[0]
            if node_type == 'think_search':
                input_text += f"<think> {solution_item[node_type]['think']} </think>\n"
                input_text += f"<search> {solution_item[node_type]['search_query']} </search>\n"
                docs = solution_item[node_type]['retrieved_documents']
                if len(docs) > 0:
                    input_text += f"<information> {_passages2string(docs)}</information>\n"
            if node_type == 'critique_search':
                input_text += f"<critique> {solution_item[node_type]['critique']} </critique>\n"
                input_text += f"<search> {solution_item[node_type]['search_query']} </search>\n"
                docs = solution_item[node_type]['retrieved_documents']
                if len(docs) > 0:
                    input_text += f"<information> {_passages2string(docs)}</information>\n"
            if node_type == 'think_answer':
                input_text += f"<think> {solution_item[node_type]['think']} </think>\n"
                input_text += f"<answer> {solution_item[node_type]['answer']} </answer>\n"    
            if node_type == 'critique_answer':
                input_text += f"<critique> {solution_item[node_type]['critique']} </critique>\n"
                input_text += f"<answer> {solution_item[node_type]['answer']} </answer>\n"
            
        return input_text  

    def generate(self,
        input_texts:list[str],
        question:str,
        max_new_tokens:int = 1024,
        num_return_sequences:int = 1,
        temperature:float = 1.0,
        top_p:float = 0.95,
        top_k:int = 40,
        repetition_penalty:float = 1.1,
        best_of=None,
    ):
        generated_texts = []
        for input_text in input_texts:
            input_text_ = f"{self.prompt_instruction} {question.strip()}\n{input_text}"
            if self.se_model.tokenizer.chat_template:
                input_prompt = self.se_model.tokenizer.apply_chat_template([{"role": "user", "content": input_text_}], add_generation_prompt=True, tokenize=False)
            else:
                input_prompt = input_text_
            
            input_ids = self.se_model.tokenizer.encode(input_prompt, return_tensors='pt').to(self.se_model.model.device)
            attention_mask = torch.ones_like(input_ids)
            num_beams = best_of if best_of and best_of > num_return_sequences else num_return_sequences
            do_sample = best_of is None or best_of <= num_return_sequences
            
            with torch.no_grad():
                outputs = self.se_model.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=self.se_model.tokenizer.eos_token_id,
                    pad_token_id=self.se_model.tokenizer.eos_token_id,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    num_return_sequences=num_return_sequences,
                    num_beams=num_beams,
                    stopping_criteria=self.answer_stopping_criteria,
                )
                generated_texts.extend(self.se_model.tokenizer.batch_decode(
                    [output[input_ids.shape[1]:] for output in outputs],
                    skip_special_tokens=True
                ))
        del input_ids, outputs
        torch.cuda.empty_cache()
        
        return generated_texts

    def _filter_none(self, candidates: list[Candidate]) -> list[Candidate]:
        candidates = [c for c in candidates if c.final_answer is not None]
        return candidates

    def _filter_long(self, candidates: list[Candidate]) -> list[Candidate]:
        candidates = [c for c in candidates if len(c.final_answer) <= 40]
        return candidates

    def _filter_white_space(self, candidates: list[Candidate]) -> list[Candidate]:
        candidates = [c for c in candidates if c.final_answer.strip()]
        return candidates

    def _filter_specific_words(self, candidates: list[Candidate]) -> list[Candidate]:
        words = ['not enough information', 'not enough information provided', 'unknown', 'more information needed', 'none', 'not specified in the given information', 'information not specified', 'no direct information available in current context', 'no direct information available in the knowledge base.']
        filtered_candidates = []
        for c in candidates:
            normalized_c = normalize_answer(c.final_answer)
            if not any(w in normalized_c for w in words):
                filtered_candidates.append(c)
        return filtered_candidates

    def _filter_reasoning_consistency(self, question: str, candidates: list[Candidate], aux={}) -> list[Candidate]:
        assert all(
            len(c.masked_trace_text_list) == self.args.num_masked_solution_traces
            for c in candidates
            if c.c_type == "default"
        )
        gen_input_list = []
        ground_truth_list = []
        c_completion_num_list = []
        for c in candidates:
            for masked_solution_trace in c.masked_trace_text_list:
                for _ in range(self.args.rc_n_completions):
                    gen_input_list.append(masked_solution_trace)
                    ground_truth_list.append(c.final_answer)
            c_completion_num_list.append(len(c.masked_trace_text_list) * self.args.rc_n_completions)
        """gen_input_list:
        [c1_mask1, c1_mask2, ..., c2_mask1, c2_mask2, ..., ......, ct_mask1, ct_mask2, ...]
        """
        
        # Manually split into batches
        # batch_size = self.args.max_num_seqs // self.args.rc_n_completions // 2
        batch_size = self.args.rc_n_completions
        gen_output_list = []
        for start_idx in range(0, len(gen_input_list), batch_size):
            end_idx = start_idx + batch_size
            sub_gen_input_list = gen_input_list[start_idx:end_idx]
            sub_gen_output_list = self.generate(
                input_texts=sub_gen_input_list,
                question=question,
                max_new_tokens=256,
                num_return_sequences=1,
                temperature=self.args.rc_temperature,
                # best_of=self.args.best_of
            )
            gen_output_list.extend(sub_gen_output_list)

        if all(isinstance(item, list) for item in gen_output_list):
            completion_list = []
            for n_completions in gen_output_list:
                for completion in n_completions:
                    completion_list.append(completion)
            assert len(completion_list) == self.args.rc_n_completions * self.args.num_masked_solution_traces * len(candidates)
            candidate_group_size = self.args.rc_n_completions * self.args.num_masked_solution_traces
        elif all(isinstance(item, str) for item in gen_output_list):
            completion_list = gen_output_list
            candidate_group_size = self.args.num_masked_solution_traces
        
        answer_list = [get_answer(completion) for completion in completion_list]
        count = 0
        completion_group_list = []
        answer_group_list = []
        gt_group_list = []
        for num in c_completion_num_list:
            completion_group_list.append(completion_list[count : count + num])
            answer_group_list.append(answer_list[count : count + num])
            gt_group_list.append(ground_truth_list[count : count + num])
            count += num
        assert count == len(completion_list) == len(answer_list)

        consistent_candidates = []
        for c, completion_group, answer_group, gt_answer in zip(
            candidates, completion_group_list, answer_group_list, gt_group_list
        ):
            candidate_group_size = len(c.masked_trace_text_list)
            num_consistent = 0
            if self.args.rc_mode == "maj":
                answer = self.find_most_confident_answer(question, completion_group)[0]
                if self.se_model.check_answers_equiv(question, gt_answer[-1], answer):
                    consistent_candidates.append(c)
            else:
                for answer, gt_a in zip(answer_group, gt_answer):
                    if self.se_model.check_answers_equiv(question, gt_a, answer):
                        num_consistent += 1
                if self.args.rc_mode == "loose":
                    if num_consistent > 0:
                        consistent_candidates.append(c)
                elif self.args.rc_mode == "mid":
                    if num_consistent >= candidate_group_size // 2:
                        consistent_candidates.append(c)
                elif self.args.rc_mode == "strict":
                    if num_consistent == candidate_group_size:
                        consistent_candidates.append(c)
        return consistent_candidates

    def _BoN_filter(self, question: str, candidates: list[Candidate], aux={}) -> list[Candidate]:
        assert all(
            len(c.masked_trace_text_list) == self.args.num_masked_solution_traces
            for c in candidates
            if c.c_type == "default"
        )
        gen_input_list = []
        ground_truth_list = []
        c_completion_num_list = []
        for c in candidates:
            for masked_solution_trace in c.masked_trace_text_list:
                for _ in range(self.args.rc_n_completions):
                    gen_input_list.append(masked_solution_trace)
                    ground_truth_list.append(c.final_answer)
            c_completion_num_list.append(len(c.masked_trace_text_list) * self.args.rc_n_completions)
        """gen_input_list:
        [c1_mask1, c1_mask2, ..., c2_mask1, c2_mask2, ..., ......, ct_mask1, ct_mask2, ...]
        """
        
        # Manually split into batches
        # batch_size = self.args.max_num_seqs // self.args.rc_n_completions // 2
        batch_size = self.args.rc_n_completions
        gen_output_list = []
        for start_idx in range(0, len(gen_input_list), batch_size):
            end_idx = start_idx + batch_size
            sub_gen_input_list = gen_input_list[start_idx: end_idx]
            sub_gen_output_list = self.generate(
                input_texts=sub_gen_input_list,
                question=question,
                max_new_tokens=256,
                num_return_sequences=1,
                temperature=self.args.rc_temperature,
                best_of=self.args.best_of
            )
            gen_output_list.extend(sub_gen_output_list)
        # """gen_output_list:
        # [[c1_mask1_o1, c1_mask1_o2, ...], [c1_mask2_o1, c1_mask2_o2, ...], ..., [ct_mask1_o1, ct_mask1_o2, ...], [ct_mask2_o1, ct_mask2_o2, ...], ...]
        # """
        
        if all(isinstance(item, list) for item in gen_output_list):
            completion_list = []
            for n_completions in gen_output_list:
                for completion in n_completions:
                    completion_list.append(completion)
            assert len(completion_list) == self.args.rc_n_completions * self.args.num_masked_solution_traces * len(candidates)
            candidate_group_size = self.args.rc_n_completions * self.args.num_masked_solution_traces
        elif all(isinstance(item, str) for item in gen_output_list):
            completion_list = gen_output_list
            candidate_group_size = self.args.num_masked_solution_traces
        
        answer_list = [get_answer(completion) for completion in completion_list]
        
        count = 0
        completion_group_list = []
        answer_group_list = []
        gt_group_list = []
        for num in c_completion_num_list:
            completion_group_list.append(completion_list[count : count + num])
            answer_group_list.append(answer_list[count : count + num])
            gt_group_list.append(ground_truth_list[count : count + num])
            count += num
        assert count == len(completion_list) == len(answer_list)

        consistent_candidates = []
        for c, completion_group, answer_group, gt_answer in zip(
            candidates, completion_group_list, answer_group_list, gt_group_list
        ):
            candidate_group_size = len(c.masked_trace_text_list)
            num_consistent = 0
            if self.args.rc_mode == "maj":
                answer = self.find_most_confident_answer(question, answer_group)[0]
                if self.se_model.check_answers_equiv(question, gt_answer[-1], answer):
                    consistent_candidates.append(c)
            else:
                for answer, gt_a in zip(answer_group, gt_answer):
                    if self.se_model.check_answers_equiv(question, gt_a, answer):
                        num_consistent += 1
                if self.args.rc_mode == "loose":
                    if num_consistent > 0:
                        consistent_candidates.append(c)
                elif self.args.rc_mode == "mid":
                    if num_consistent >= candidate_group_size // 2:
                        consistent_candidates.append(c)
                elif self.args.rc_mode == "strict":
                    if num_consistent == candidate_group_size:
                        consistent_candidates.append(c)
        return consistent_candidates

    def _find_winner_filtered(
        self, question:str, unfiltered_candidates: list[Candidate], filtered_candidates: list[Candidate], gt_answer: str = None
    ) -> Candidate:
        if len(filtered_candidates) == 0:
            answer2candidates, answer2confidence, _ = self.group_candidates_by_answer(
                question, unfiltered_candidates, self.args.rc_criteria
            )
            if answer2confidence:
                most_confident_answer = max(answer2confidence.keys(), key=lambda x: answer2confidence[x], default=None)
                winner = answer2candidates[most_confident_answer][0]
                print(f"==> Winner answer: {most_confident_answer}\n")
            else:
                winner = None
        elif len(filtered_candidates) == 1:
            winner = filtered_candidates[0]
            print(f"==> Winner answer: {winner.final_answer}\n")
        # elif not any(self.se_model.check_answers_equiv(question, c.final_answer, gt_answer[0]) for c in filtered_candidates):
        #     winner = None
        #     print(f"==> Winner answer: None")
        else:
            filtered_answer2score = self._calculate_scores(question, unfiltered_candidates, filtered_candidates)
            winner_answer = max(filtered_answer2score.keys(), key=lambda x: filtered_answer2score[x])
            print(f"==> Winner answer: {winner_answer}")
            winner = next(
                c for c in filtered_candidates if self.se_model.check_answers_equiv(question, c.final_answer, winner_answer)
            )
        return winner

    def _calculate_scores(self, question:str, unfiltered_candidates: list[Candidate], filtered_candidates: list[Candidate]) -> dict:
        _, filtered_answer2confidence, filtered_answer2cnt = self.group_candidates_by_answer(
            question, filtered_candidates, self.args.rc_criteria
        )
        print(f"==> Confidence: {filtered_answer2confidence}")
        _, _, unfiltered_answer2cnt = self.group_candidates_by_answer(
            question, unfiltered_candidates, self.args.rc_criteria
        )

        filtered_answer2survival_rate = {}
        for filtered_ans in filtered_answer2cnt.keys():
            has_existed = False
            for unfiltered_ans in unfiltered_answer2cnt.keys():
                if self.se_model.check_answers_equiv(question, filtered_ans, unfiltered_ans):
                    has_existed = True
                    filtered_answer2survival_rate[filtered_ans] = (
                        filtered_answer2cnt[filtered_ans] / unfiltered_answer2cnt[unfiltered_ans]
                    )
                    break
            if not has_existed:
                filtered_answer2survival_rate[filtered_ans] = 0.0
        print(f"==> Survival rates: {filtered_answer2survival_rate}")

        filtered_answer2score = {}
        for filtered_ans in filtered_answer2confidence.keys():
            has_existed = False
            for unfiltered_ans in unfiltered_answer2cnt.keys():
                if self.se_model.check_answers_equiv(question, filtered_ans, unfiltered_ans):
                    has_existed = True
                    filtered_answer2score[filtered_ans] = (
                        filtered_answer2confidence[filtered_ans] + filtered_answer2survival_rate[filtered_ans]
                    )
                    break
            if not has_existed:
                filtered_answer2score[filtered_ans] = 0.0
        print(f"==> Scores: {filtered_answer2score}")

        return filtered_answer2score

    def group_candidates_by_answer(self, question:str, candidates: list[Candidate], criteria="freq"):
        """Return answer2candidates, answer2confidence, answer2cnt."""
        answer2candidates = {}
        answer2confidence = defaultdict(float)
        answer2cnt = defaultdict(int)

        for c in candidates:
            has_existed = False
            for existing_answer in answer2candidates.keys():
                if self.se_model.check_answers_equiv(question, c.final_answer, existing_answer):
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

    def find_most_confident_answer(self, question:str, completions: list[str], prior_weights: list[float] = None):
        """Returns the most confident answer, its completion, its id in the input list, and its confidence."""
        if completions is None or len(completions) == 0:
            return None, None, None, None

        answer2completions = defaultdict(list)
        answer2ids = defaultdict(list)
        for id, c in enumerate(completions):
            try:
                model_answer = get_answer(c)
                has_existed = False
                for existing_answer in answer2completions.keys():
                    if self.check_answers_equiv(question, model_answer, existing_answer):
                        assert not has_existed
                        has_existed = True
                        answer2completions[existing_answer].append(c)
                        answer2ids[existing_answer].append(id)
                if not has_existed:
                    answer2completions[model_answer].append(c)
                    answer2ids[model_answer].append(id)
            except:
                pass

        assert len(answer2completions.keys()) > 0, "There are no valid completions."
        if prior_weights is not None:
            assert len(completions) == len(prior_weights)
            completion2count = {}
            for answer, answer_completions in answer2completions.items():
                count = len(answer_completions)
                for answer_completion in answer_completions:
                    completion2count[answer_completion] = count

            completion2score = {}
            for id, (completion, count) in enumerate(completion2count.items()):
                prior_weight = prior_weights[id]
                score = prior_weight * (count / len(completions))
                completion2score[completion] = score

            most_confident_completion = max(completion2score.keys(), key=lambda x: completion2score[x])

            return (
                get_answer(most_confident_completion),
                most_confident_completion,
                completions.index(most_confident_completion),
                completion2score[most_confident_completion],
            )
        else:
            most_confident_answer = max(answer2completions.keys(), key=lambda x: len(answer2completions[x]))
            assert (
                len(answer2completions[most_confident_answer]) > 0
            ), "There are no completions for the most confident answer."
            confidence = len(answer2completions[most_confident_answer]) / len(completions)
            assert confidence > 0
            return (
                most_confident_answer,
                answer2completions[most_confident_answer][0],
                answer2ids[most_confident_answer][0],
                confidence,
            )

    def select(self, question: str, candidates: list[Candidate], gt_answer: str = None, aux={}) -> Candidate:
        print(f"==> Ground truth answer: {gt_answer}")
        unfiltered_candidates = candidates
        print(f"==> Unfiltered answers: {[c.final_answer for c in unfiltered_candidates]}")
        prefiltered_candidates = self._filter_none(candidates)
        prefiltered_candidates = self._filter_long(prefiltered_candidates)
        prefiltered_candidates = self._filter_white_space(prefiltered_candidates)
        prefiltered_candidates = self._filter_specific_words(prefiltered_candidates)
        print(f"==> Pre-filtered answers: {[c.final_answer for c in prefiltered_candidates]}")
        
        # select the final trajectory through Reasoning Consistency
        if self.args.extend_rc_mode == 'reasoning_consistency':
            filtered_candidates = self._filter_reasoning_consistency(question, prefiltered_candidates, aux)
            print(f"==> RC-filtered answers: {[c.final_answer for c in filtered_candidates]}")
        elif self.args.extend_rc_mode == 'BoN':
            filtered_candidates = self._BoN_filter(question, prefiltered_candidates, aux)
            print(f"==> BoN-filtered answers: {[c.final_answer for c in filtered_candidates]}")
        elif self.args.extend_rc_mode == 'majority_vote':
            filtered_candidates = []
        else:
            raise NotImplementedError

        return self._find_winner_filtered(question, prefiltered_candidates, filtered_candidates, gt_answer)

def rc_discrimination(args):
    print("\n== MCTS Discrimination ...")
    print(f"""
        Model name:  {args.model_name_or_path}
        Dataset:     {args.dataset} / {args.subsec} ({args.fraction_of_data_to_use})
        Retriever:   {args.retriever_name} / ({args.retrieval_model_path})
        Rollouts:    {args.num_rollouts}
        Seed:        {args.seed}
        Run:         {args.run}
    """.replace('        ', ''))
    
    # === Output files ==========================
    entries = os.listdir(args.generation_trees_results_dir)
    query_ids = [entry for entry in entries if os.path.isdir(os.path.join(args.generation_trees_results_dir, entry))]
    sorted_query_ids = sorted(query_ids, key=lambda x: int(x.split('_')[1]))
    
    # === Model Definition ======================
    se_model = SemanticEquivalenceGenerator(args)
    discriminator = Discriminator(args, se_model)
    
    # === Main Loop =============================
    em_evaluation = []
    with open(args.discriminate_results_file, 'w', encoding='utf-8') as outfile:
        for i, qid in enumerate(tqdm(sorted_query_ids)):
            # if i == 10:
            #     break
            # === Generating answer candidates
            final_solutions_file = f"{args.generation_trees_results_dir}/{qid}/final_solutions.jsonl"
            all_traces = read_jsonl(final_solutions_file)
            gt_answers = all_traces[0]["trace"]["0"]["ground_truth"]
            question = all_traces[0]["trace"]["0"]["user_question"]
            question = question.strip()
            if question[-1] != '?':
                question += '?'
            
            all_candidates = []
            for trace_id, trace in enumerate(all_traces):
                
                trace_ = trace["trace"]
                trace_ = {int(key): val for key, val in  trace_.items()}
                trace_text = discriminator.trace2text(trace_)
                last_depth_key = list(trace_.keys())[-1]
                last_node_type = list(trace_[last_depth_key].keys())[0] 
                final_answer = trace_[last_depth_key][last_node_type]["answer"]
                # final_answer_reward = trace_[last_depth_key][last_node_type]["value"]
                final_answer_reward = trace_[last_depth_key][last_node_type]["node_reward"]
                # final_answer_reward = 10 - trace_[last_depth_key][last_node_type]['scores'][1]['param']['PE']['uncertainty']
                
                masked_trace_text_list = rag_mask_solution_trace(
                    trace_text,
                    num_return=args.num_masked_solution_traces,
                    left_boundary=args.mask_left_boundary,
                    right_boundary=args.mask_right_boundary,
                )
                
                candidate = Candidate(
                    trace_text,
                    deepcopy(masked_trace_text_list),
                    final_answer,
                    trace_id,
                    trace_reward=final_answer_reward,
                )
                # print(candidate.to_dict())
                # print('----')
                all_candidates.append(candidate)
                
            
            # 
            answer2candidates, answer2confidence, _ = discriminator.group_candidates_by_answer(
                question, all_candidates, args.rc_criteria
            )
            most_confident_answer = max(answer2candidates.keys(), key=lambda x: answer2confidence[x])
            highest_confidence = answer2confidence[most_confident_answer]
            assert highest_confidence > 0
            print(answer2confidence)
            # === Decision
            if highest_confidence > args.threshold:
                print("You are very confident. Skipping...")
                winner_answer = most_confident_answer if most_confident_answer != None else ''
            else:
                winner_answer_ = discriminator.select(question, all_candidates, gt_answers)
                winner_answer = winner_answer_.final_answer if winner_answer_ != None else ''
            
            correctness_em = em_score(winner_answer, gt_answers)
            em_evaluation.append(correctness_em)
            item = {
                "qid": qid,
                "query": question,
                "gt_answers": gt_answers,
                "em": correctness_em,
                "winner_answer": winner_answer,
                "pred_answers": [c.final_answer for c in all_candidates],
                "conf": answer2confidence[winner_answer]
            }
            outfile.write(json.dumps(item) + '\n')

    # === Print results ========================
    print("\nEvaluation Result:")
    print(f"EM: {np.mean(em_evaluation)*100}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    # parser.add_argument('--model_name_or_path_disc', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--max_new_token', type=int, default=512)
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='bamboogle', choices=[
        'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
    
    # Retriever
    parser.add_argument('--retriever_name', type=str, default='rerank_l6', choices=[
        'bm25', 'contriever', 'rerank_l6', 'rerank_l12', 'e5', 'bge'
    ])
    parser.add_argument('--corpus_path', type=str, default='data/search_r1_files/wiki-18.jsonl')
    parser.add_argument('--index_path', type=str, default='data/search_r1_files/e5_Flat.index', choices=[
        'data/search_r1_files/bm25',          # For BM25 & Rerank
        'data/search_r1_files/e5_Flat.index', # For E5
    ])
    parser.add_argument("--retrieval_model_path", type=str, default="intfloat/e5-base-v2", choices=[
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
    parser.add_argument('--run', type=str, default='run_7 (new_setup_roll8)')
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument('--use_counter', action='store_false')
    
    # MCTS ---
    parser.add_argument("--num_rollouts", type=int, default=4)
    parser.add_argument("--max_depth_allowed", type=int, default=4)
    parser.add_argument("--num_votes", type=int, default=1)
    parser.add_argument("--mcts_num_last_votes", type=int, default=10)
    parser.add_argument("--verbose", action="store_true", help="extra login")
    parser.add_argument("--mcts_discount_factor", type=float, default=1.0)
    parser.add_argument("--mcts_exploration_weight", type=float, default=2.0)
    parser.add_argument("--mcts_weight_scheduler", choices=["exp", "lin", "const"], default="const")
    parser.add_argument("--save_tree", action="store_true")
    parser.add_argument("--enable_potential_score", action="store_true")
    
    # Discrimination ---
    parser.add_argument("--cutoff_rollout", type=int, default=-1)
    parser.add_argument("--start_idx", type=int, default=-1)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--mask_left_boundary", type=float, default=0.2)
    parser.add_argument("--mask_right_boundary", type=float, default=0.5)
    parser.add_argument("--num_masked_solution_traces", type=int, default=1)
    parser.add_argument("--rc_mode", type=str, default="mid", choices=["loose", "mid", "strict", "maj"])
    parser.add_argument("--rc_temperature", type=float, default=1.0)
    parser.add_argument("--rc_n_completions", type=int, default=1)
    parser.add_argument("--rc_criteria", type=str, default="freq", choices=["freq", "reward"])
    parser.add_argument("--threshold", type=float, default=0.999)
    parser.add_argument("--extend_rc_mode", type=str, default="majority_vote", choices=["reasoning_consistency", "BoN", "majority_vote"])
    parser.add_argument("--best_of", type=int, default=5)
    # parser.add_argument("--max_num_seqs", type=int, default=2)
    
    args = parser.parse_args()
    
    # === Files ====================
    model_ = args.model_name_or_path.split('/')[-1]
    output_dir = f"run_output/{args.run}/{model_}/{args.dataset}_{args.subsec}/{args.retriever_name}"
    args.generation_trees_results_dir = f'{output_dir}/generation_trees'
    args.discriminate_results_file = f"{output_dir}/rc_discriminate_results_v2.jsonl"
    
    # === Prompt files =============
    args.semantic_equivalence_prompt_file = "prompts_mcts/semantic_equivalence_prompt_template.txt"
    args.discriminator_prompt_file = "prompts_mcts/discriminator_prompt_template.txt"
    
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
    rc_discrimination(args)
    
    
    # python run_mcts/rc_discrimination_v2.py
    