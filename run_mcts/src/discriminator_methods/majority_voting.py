#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import math
import torch
import transformers
from typing import Dict
from copy import deepcopy

from run_mcts.src.discriminator_methods.basic_discriminator import BasicDiscriminator, Candidate
from run_rag_methods.src.rag_methods import passages2string
from run_rag_methods.src.generators import StopOnSequence


class MajorityVoting(BasicDiscriminator):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.prompt_instruction = self.get_instruction('think_answer')
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
                    input_text += f"<information> {passages2string(docs)}</information>\n"
            if node_type == 'critique_search':
                input_text += f"<critique> {solution_item[node_type]['critique']} </critique>\n"
                input_text += f"<search> {solution_item[node_type]['search_query']} </search>\n"
                docs = solution_item[node_type]['retrieved_documents']
                if len(docs) > 0:
                    input_text += f"<information> {passages2string(docs)}</information>\n"
            if node_type == 'think_answer':
                input_text += f"<think> {solution_item[node_type]['think']} </think>\n"
                input_text += f"<answer> {solution_item[node_type]['answer']} </answer>\n"    
            if node_type == 'critique_answer':
                input_text += f"<critique> {solution_item[node_type]['critique']} </critique>\n"
                input_text += f"<answer> {solution_item[node_type]['answer']} </answer>\n"
            
        return input_text  
    
    def rag_mask_solution_trace(self, solution_trace_str: str, num_return: int, left_boundary: float, right_boundary: float) -> list[str]:
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
    
    def select(self, question: str, candidates: list[Candidate], gt_answer: str = None, aux={}) -> Candidate:
        print(f"==> Ground truth answer: {gt_answer}")
        unfiltered_candidates = candidates
        print(f"==> Unfiltered answers: {[c.final_answer for c in unfiltered_candidates]}")
        prefiltered_candidates = self._filter_none(candidates)
        prefiltered_candidates = self._filter_long(prefiltered_candidates)
        prefiltered_candidates = self._filter_white_space(prefiltered_candidates)
        prefiltered_candidates = self._filter_specific_words(prefiltered_candidates)
        print(f"==> Pre-filtered answers: {[c.final_answer for c in prefiltered_candidates]}")
        filtered_candidates = []
        winner, filtered_answer2score = self._find_winner_filtered(question, prefiltered_candidates, filtered_candidates, gt_answer)
        return winner, filtered_answer2score
    
    def inference(self, question, gt_answers, paths):
        all_candidates = []
        for trace_id, trace in enumerate(paths):
            trace_ = trace["trace"]
            trace_ = {int(key): val for key, val in  trace_.items()}
            trace_text = self.trace2text(trace_)
            last_depth_key = list(trace_.keys())[-1]
            last_node_type = list(trace_[last_depth_key].keys())[0] 
            final_answer = trace_[last_depth_key][last_node_type]["answer"]
            final_answer_reward = trace_[last_depth_key][last_node_type]["node_reward"]
            
            # masked_trace_text_list = self.rag_mask_solution_trace(
            #     trace_text,
            #     num_return=self.args.num_masked_solution_traces,
            #     left_boundary=self.args.mask_left_boundary,
            #     right_boundary=self.args.mask_right_boundary,
            # )
            # candidate = Candidate(
            #     trace_text, deepcopy(masked_trace_text_list),
            #     final_answer, trace_id, trace_reward=final_answer_reward,
            # )
            candidate = Candidate(
                trace_text, [],
                final_answer, trace_id, trace_reward=final_answer_reward,
            )
            all_candidates.append(candidate)
        
        answer2candidates, answer2confidence, _ = self.group_candidates_by_answer(
            question, all_candidates, self.args.rc_criteria
        )
        most_confident_answer = max(answer2candidates.keys(), key=lambda x: answer2confidence[x])
        highest_confidence = answer2confidence[most_confident_answer]
        assert highest_confidence > 0
        
        # === Decision
        if highest_confidence > self.args.threshold:
            print("You are very confident. Skipping...")
            winner_answer = most_confident_answer if most_confident_answer != None else ''
            answer2score = {winner_answer: 1.0}
        else:
            winner_answer_, answer2score  = self.select(question, all_candidates, gt_answers)
            winner_answer = winner_answer_.final_answer if winner_answer_ != None else ''
        
        return winner_answer, answer2score
