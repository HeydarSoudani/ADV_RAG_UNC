#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import json
import torch
import random
from copy import deepcopy
from typing import List, Dict, Tuple
import transformers

from utils.general_utils import get_answer
from run_rag_methods.src.retrievers_local import BM25Retriever, ContrieverRetriever, RerankRetriever, DenseRetriever
from run_mcts_two_actions.src.discriminator_methods.basic_discriminator import BasicDiscriminator
from run_mcts_two_actions.src.models.generate_paraphrase import (
    SearchQueryGenerator, ThinkGenerator,
    get_paraphrased_query, get_paraphrased_think
)
from run_mcts_two_actions.src.generate_node import Generator as NodeGenerator

class Candidate:
    def __init__(
        self,
        qid,
        trace_obj,
        final_answer,
        trace_id,
        trace_reward=1.0,
        trace_freq=1,
        c_type="default",
        masked_trace_retrieval_list=None,
    ):
        self.qid = qid
        self.trace_obj = trace_obj
        self.final_answer = final_answer
        self.trace_id = trace_id
        self.trace_reward = trace_reward
        self.trace_freq = trace_freq
        self.c_type = c_type
        self.masked_trace_retrieval_list = masked_trace_retrieval_list
        # self.rag_confidence = self.get_rag_confidence()

    def __str__(self):
        return f"Candidate {self.trace_id}: {self.final_answer} | {self.rag_confidence}"

    def to_dict(self):
        return {
            "trace_id": self.trace_id,
            "trace_reward": self.trace_reward,
            "trace_freq": self.trace_freq,
            "final_answer": self.final_answer,
            "trace_obj": self.trace_obj,
            "masked_trace_retrieval_list": self.masked_trace_retrieval_list
        }

    def get_masked_traces(self,
        node_generator, search_query_generator, think_generator,
        retriever, args,
    ) -> List[Dict]:
        masked_traces = []
        
        # has search
        sorted_keys = sorted(self.trace_obj.keys(), key=int)
        has_search = any("think_search" in self.trace_obj[k] for k in sorted_keys)
        
        if has_search:
            think_search_indices = [k for k, v in self.trace_obj.items() if "think_search" in v]
            think_answer_index = list(self.trace_obj.keys())[-1]
            selected_indices = random.choices(think_search_indices, k=args.num_masked_solution_traces)
            selected_indices_group = [(x, selected_indices.count(x)) for x in sorted(set(selected_indices))]
            
            # print('=====')
            # print(self.trace_obj)
            # print('---')
            
            for (selected_index, repeat) in selected_indices_group:
                # print(f"{selected_index} - {repeat}")
                ## Step 1: Generating paraphrased search queries 
                original_sq = self.trace_obj[selected_index]['think_search'].get('search_query', '')
                sq_prompt = search_query_generator.get_instruction(original_sq, n=repeat)
                sq_output = search_query_generator.generate(sq_prompt, temperature=0.7)[0]
                paraphrased_queries = get_paraphrased_query(sq_output)
                
                # check if paraphrased_queries are None
                if paraphrased_queries == None:
                    print(f"Paraphrased queries are not provided for query {self.qid} ...")
                    for i in range(args.retry):
                        print(f"Paraphrased queries, try {i+1} ...")
                        sq_output = search_query_generator.generate(sq_prompt, temperature=1.0)[0]
                        paraphrased_queries = get_paraphrased_query(sq_output)
                        if paraphrased_queries != None:
                            break
                    else:
                        print(f"Failed to generate 'paraphrased queries' after all retries for query {self.qid}!!!")
                        paraphrased_queries = []
                
                # Check if the number of paraphrased_queries is equal to "repeat"
                if paraphrased_queries is None:
                    paraphrased_queries = []
                
                max_iterations = 10
                iteration = 0
                while len(paraphrased_queries) != repeat and iteration < max_iterations:
                    remaining = repeat - len(paraphrased_queries)        
                    extra_prompt = search_query_generator.get_instruction(original_sq, n=remaining)
                    extra_output = search_query_generator.generate(extra_prompt, temperature=1.0)[0]
                    extra_queries = get_paraphrased_query(extra_output)

                    if extra_queries:
                        paraphrased_queries.extend(extra_queries)
                        paraphrased_queries = paraphrased_queries[:repeat]  # trim if over
                    else:
                        print(f"Failed to generate extra queries on iteration {iteration + 1}")
                    iteration += 1
                if len(paraphrased_queries) != repeat:
                    print(f"Warning: Only generated {len(paraphrased_queries)} queries out of {repeat} after {iteration} iterations.")
                
                # print(paraphrased_queries)
                # print('---')
                
                ## Step 2: Generating new masked traces
                for paraphrased_query in paraphrased_queries:
                    paraphrased_query = paraphrased_query.strip()
                    new_trace = {}
                    
                    # Before break point: Keep steps excluding the selected one
                    for i in range(selected_index):
                        new_trace[i] = deepcopy(self.trace_obj[i])
                
                    
                    # On break point
                    retrieved_docs = retriever.search(paraphrased_query) if paraphrased_query else []
                    new_trace[selected_index] = {
                        "think_search": {
                            "think": self.trace_obj[selected_index]["think_search"].get('think', ''),
                            "search_query": paraphrased_query,
                            "retrieved_documents": retrieved_docs,
                        }
                    }
                    
                    # print(new_trace)
                    # print('---')
                    
                    # After break point: next think_search steps
                    for i in range(selected_index+1, think_answer_index):
                        thinks, search_query, ret_docs = node_generator.generate_think_search(new_trace)
                        new_trace[i] = {
                            "think_search": {
                                "think": thinks,
                                "search_query": search_query,
                                "retrieved_documents": ret_docs
                            }
                        }
                    
                    # Last step: think_answer
                    think, most_likely_answer, reward, _ = node_generator.generate_think_answer(new_trace)
                    
                    # print(think)
                    # print(most_likely_answer)
                    
                    new_trace[think_answer_index] = {
                        "think_answer": {
                            "think": think,
                            "answer": most_likely_answer,
                            "node_reward": reward,
                            "scores": (0.0, 0.0, 0.0)
                        }
                    }
                    
                    # print(new_trace)
                    # print('---')
                    
                    # Add new complete path
                    masked_traces.append(new_trace)
                    # ---------------------------------
        
                # print('=======')
        else:
            ## Step 1: Generating paraphrased thinks
            original_think = self.trace_obj[1]["think_answer"].get('think', '')
            think_prompt = think_generator.get_instruction(original_think, n=args.num_masked_solution_traces)
            think_output = think_generator.generate(think_prompt, temperature=0.7)[0]
            paraphrased_thinks = get_paraphrased_think(think_output)
            
            # check if paraphrased_thinks are None
            if paraphrased_thinks == None:
                print(f"Paraphrased thinks are not provided for query {self.qid} ...")
                for i in range(args.retry):
                    print(f"Paraphrased thinks, try {i+1} ...")
                    think_output = think_generator.generate(think_prompt, temperature=1.0)[0]
                    paraphrased_thinks = get_paraphrased_query(think_output)
                    if paraphrased_thinks != None:
                        break
                else:
                    print(f"Failed to generate 'paraphrased thinks' after all retries for query {self.qid}!!!")
                    paraphrased_thinks = []

            ## Step 2: Generating new masked traces
            input_text = node_generator.get_prompt_text('think_answer', {0: self.trace_obj[0]})
            for pt in paraphrased_thinks:
                input_text_pt = input_text + f"<think> {pt} </think>\n"
                output = node_generator.generate_(input_text_pt, node_generator.answer_stopping_criteria)[0]
                masked_traces.append({
                    0: self.trace_obj[0],
                    1: {"think_answer": {"think": pt, "answer": get_answer(output), "value": 0.9}}
                })
        
        self.masked_trace_retrieval_list = masked_traces
        return masked_traces

class RagConsistency(BasicDiscriminator):
    def __init__(self, args, device):
        super().__init__(args, device)
        
        # === Static Retriever =====================
        if args.retriever_name == 'bm25':
            self.retriever = BM25Retriever(args)  
        elif args.retriever_name == 'contriever':
            self.retriever = ContrieverRetriever(args)
        elif args.retriever_name in ['rerank_l6', 'rerank_l12']:
            self.retriever = RerankRetriever(args)
        elif args.retriever_name in ['e5', 'bge']:
            self.retriever = DenseRetriever(args)
            
        self.node_generator = NodeGenerator(args, self.retriever, self.generator, self.tokenizer)
        self.paraphrase_model = transformers.AutoModelForCausalLM.from_pretrained(args.paraphrase_model_name_or_path, torch_dtype=torch.bfloat16).to(device)
        self.paraphrase_tokenizer = transformers.AutoTokenizer.from_pretrained(args.paraphrase_model_name_or_path)
        self.search_query_generator = SearchQueryGenerator(args, self.paraphrase_model, self.paraphrase_tokenizer)
        self.think_generator = ThinkGenerator(args, self.paraphrase_model, self.paraphrase_tokenizer)

    def select(self, question: str, candidates: list[Candidate], gt_answer: str = None, aux={}) -> Candidate:
        print(f"==> Ground truth answer: {gt_answer}")
        unfiltered_candidates = candidates
        print(f"==> Unfiltered answers: {[c.final_answer for c in unfiltered_candidates]}")
        prefiltered_candidates = self._filter_none(candidates)
        prefiltered_candidates = self._filter_long(prefiltered_candidates)
        prefiltered_candidates = self._filter_white_space(prefiltered_candidates)
        prefiltered_candidates = self._filter_specific_words(prefiltered_candidates)
        print(f"==> Pre-filtered answers: {[c.final_answer for c in prefiltered_candidates]}")
        winner, candidates2score = self._find_winner_candidate(question, prefiltered_candidates)
        print(f"==> Winner answer: {winner}")
        print(candidates2score)
        
        return winner, candidates2score

    def _find_winner_candidate(self, question: str, candidates: list[Candidate]):
        assert all(
            len(c.masked_trace_retrieval_list) == self.args.num_masked_solution_traces
            for c in candidates
            if c.c_type == "default"
        )
        completion_list = []
        ground_truth_list = []
        c_completion_num_list = []
        for c in candidates:
            for masked_solution_trace in c.masked_trace_retrieval_list:
                for _ in range(self.args.rc_n_completions):
                    completion_list.append(masked_solution_trace)
                    ground_truth_list.append(c.final_answer)
            c_completion_num_list.append(len(c.masked_trace_retrieval_list) * self.args.rc_n_completions)
        """completion_list:
        [c1_mask1, c1_mask2, ..., c2_mask1, c2_mask2, ..., ......, ct_mask1, ct_mask2, ...]
        """
        
        answer_list = [
            completion[list(completion.keys())[-1]]['think_answer'].get("answer", '')
            for completion in completion_list
        ]
        print(answer_list)
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
        
        candidates2score = {}
        for c, answer_group, gt_answer in zip(candidates, answer_group_list, gt_group_list):
            # guard against empty groups
            num_items = len(answer_group)
            if num_items == 0:
                continue                      # or set score = 0.0

            # consistency count
            num_consistent = sum(
                self.se_model.check_answers_equiv(question, gt, ans)
                for ans, gt in zip(answer_group, gt_answer)
            )
            score = num_consistent / num_items
            # keep best score per final answer
            candidates2score[c.final_answer] = max(score, candidates2score.get(c.final_answer, 0))
        
        winner = max(candidates2score, key=candidates2score.get)

        return winner, candidates2score

    def inference(self, qid, question, gt_answers, paths):
        all_candidates = []
        for trace_id, trace in enumerate(paths):
            trace_ = trace["trace"]
            trace_ = {int(key): val for key, val in  trace_.items()}
            last_depth_key = list(trace_.keys())[-1]
            last_node_type = list(trace_[last_depth_key].keys())[0] 
            final_answer = trace_[last_depth_key][last_node_type]["answer"]
            final_answer_reward = trace_[last_depth_key][last_node_type]["node_reward"]
            
            # skip no-RAG setup # if len(trace_) > 2:
            candidate = Candidate(qid, trace_, final_answer, trace_id, trace_reward=final_answer_reward)
            all_candidates.append(candidate)

        # Group by semantic similarity based on final_answer
        answer2candidates, answer2confidence, _ = self.group_candidates_by_answer(
            question, all_candidates, self.args.rc_criteria
        )
        most_confident_answer = max(answer2candidates.keys(), key=lambda x: answer2confidence[x])
        highest_confidence = answer2confidence[most_confident_answer]
        assert highest_confidence > 0
            
        # Decision based on unique candidates
        if highest_confidence > self.args.threshold:
            print("You are very confident. Skipping...")
            winner_answer = most_confident_answer if most_confident_answer != None else ''
            answer2score = {winner_answer: 1.0}
        else:
            # RAG consistency for all candidates
            for idx, c in enumerate(all_candidates):
                c.get_masked_traces(
                    self.node_generator, self.search_query_generator, self.think_generator,
                    self.retriever, self.args
                )
            
            winner_answer, answer2score = self.select(question, all_candidates, gt_answers)
            # winner_answer = winner_answer_.final_answer if winner_answer_ != None else ''
        
        # print generated paths
        if not (highest_confidence > self.args.threshold):
            with open(f"{self.args.reasoning_path_generations_dir}/{qid}.jsonl", "w") as f:
                for c in all_candidates:
                    item_org = {str(c.trace_id): c.trace_obj}
                    f.write(json.dumps(item_org) + "\n")
                    for mc in c.masked_trace_retrieval_list:
                        item_ph = {str(c.trace_id): mc}
                        f.write(json.dumps(item_ph) + "\n")
        
        
        return winner_answer, answer2score
    




































    # def _filter_rag_consistency(self, question: str, candidates: list[Candidate], aux={}) -> list[Candidate]:
    #     assert all(
    #         len(c.masked_trace_retrieval_list) == self.args.num_masked_solution_traces
    #         for c in candidates
    #         if c.c_type == "default"
    #     )
    #     completion_list = []
    #     ground_truth_list = []
    #     c_completion_num_list = []
    #     for c in candidates:
    #         for masked_solution_trace in c.masked_trace_retrieval_list:
    #             for _ in range(self.args.rc_n_completions):
    #                 completion_list.append(masked_solution_trace)
    #                 ground_truth_list.append(c.final_answer)
    #         c_completion_num_list.append(len(c.masked_trace_retrieval_list) * self.args.rc_n_completions)
    #     """completion_list:
    #     [c1_mask1, c1_mask2, ..., c2_mask1, c2_mask2, ..., ......, ct_mask1, ct_mask2, ...]
    #     """
        
    #     answer_list = [
    #         completion[list(completion.keys())[-1]]['think_answer'].get("answer", '')
    #         for completion in completion_list
    #     ]
    #     print(answer_list)
    #     count = 0
    #     completion_group_list = []
    #     answer_group_list = []
    #     gt_group_list = []
    #     for num in c_completion_num_list:
    #         completion_group_list.append(completion_list[count : count + num])
    #         answer_group_list.append(answer_list[count : count + num])
    #         gt_group_list.append(ground_truth_list[count : count + num])
    #         count += num
    #     assert count == len(completion_list) == len(answer_list)
        
        
    #     consistent_candidates = []
    #     for c, completion_group, answer_group, gt_answer in zip(
    #         candidates, completion_group_list, answer_group_list, gt_group_list
    #     ):
    #         candidate_group_size = len(c.masked_trace_retrieval_list)
    #         num_consistent = 0
    #         if self.args.rc_mode == "maj":
    #             answer = self.find_most_confident_answer(question, completion_group)[0]
    #             if self.se_model.check_answers_equiv(question, gt_answer[-1], answer):
    #                 consistent_candidates.append(c)
    #         else:
    #             for answer, gt_a in zip(answer_group, gt_answer):
    #                 if self.se_model.check_answers_equiv(question, gt_a, answer):
    #                     num_consistent += 1
    #             if self.args.rc_mode == "loose":
    #                 if num_consistent > 0:
    #                     consistent_candidates.append(c)
    #             elif self.args.rc_mode == "mid":
    #                 if num_consistent >= candidate_group_size // 2:
    #                     consistent_candidates.append(c)
    #             elif self.args.rc_mode == "strict":
    #                 if num_consistent == candidate_group_size:
    #                     consistent_candidates.append(c)
        
    #     return consistent_candidates

    # def _find_winner_filtered(
    #     self, question:str, unfiltered_candidates: list[Candidate], filtered_candidates: list[Candidate], gt_answer: str = None
    # ) -> Candidate:
    #     # pass
    #     if len(filtered_candidates) == 0:
    #         answer2candidates, answer2confidence, _ = self.group_candidates_by_answer(
    #             question, unfiltered_candidates, self.args.rc_criteria
    #         )
    #         answer2score = dict(answer2confidence)
    #         if answer2confidence:
    #             most_confident_answer = max(answer2confidence.keys(), key=lambda x: answer2confidence[x], default=None)
    #             winner = answer2candidates[most_confident_answer][0]
    #             print(f"==> Winner answer: {most_confident_answer}\n")
    #         else:
    #             winner = None
    #     elif len(filtered_candidates) == 1:
    #         winner = filtered_candidates[0]
    #         answer2score = {winner.final_answer: 1.0}
    #         print(f"==> Winner answer: {winner.final_answer}\n")
    #     # elif not any(self.se_model.check_answers_equiv(question, c.final_answer, gt_answer[0]) for c in filtered_candidates):
    #     #     winner = None
    #     #     print(f"==> Winner answer: None")
    #     else:
    #         filtered_answer2score = self._calculate_scores(question, unfiltered_candidates, filtered_candidates)
    #         winner_answer = max(filtered_answer2score.keys(), key=lambda x: filtered_answer2score[x])
    #         print(f"==> Winner answer: {winner_answer}")
    #         winner = next(
    #             (c for c in filtered_candidates if self.se_model.check_answers_equiv(question, c.final_answer, winner_answer)), 
    #             None
    #         )
    #         answer2score = filtered_answer2score
    #     return winner, answer2score