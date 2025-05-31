#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import random
from copy import deepcopy
from typing import List, Dict, Tuple
import transformers

from utils.general_utils import get_answer
from run_rag_methods.src.retrievers_local import BM25Retriever, ContrieverRetriever, RerankRetriever, DenseRetriever
from run_mcts.src.discriminator_methods.basic_discriminator import BasicDiscriminator
from run_mcts.src.models.generate_paraphrase import (
    SearchQueryGenerator, ThinkGenerator,
    get_paraphrased_query, get_paraphrased_think
)
from run_mcts.src.generate_node import Generator as NodeGenerator

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
        
            for (selected_index, repeat) in selected_indices_group:
                ## Step 1: Generating paraphrased search queries 
                original_sq = self.trace_obj[selected_index].get('search_query', '')
                sq_prompt = search_query_generator.get_instruction(original_sq, n=repeat)
                sq_output = search_query_generator.generate(sq_prompt, temperature=1.0)[0]
                paraphrased_queries = get_paraphrased_query(sq_output)
                
                # check if paraphrased_queries are None
                if paraphrased_queries == None:
                    print(f"Paraphrased queries are not provided for query {self.qid} ...")
                    for i in range(args.retry):
                        print(f"Think, try {i+1} ...")
                        sq_output = search_query_generator.generate(sq_prompt, temperature=1.3)[0]
                        paraphrased_queries = get_paraphrased_query(sq_output)
                        if paraphrased_queries != None:
                            break
                    else:
                        print(f"Failed to generate 'paraphrased queries' after all retries for query {self.qid}!!!")
                        paraphrased_queries = []
                
                ## Step 2: Generating new masked traces
                for paraphrased_query in paraphrased_queries:
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
                    new_trace[think_answer_index] = {
                        "think_answer": {
                            "think": think,
                            "answer": most_likely_answer,
                            "node_reward": reward,
                            "scores": (0.0, 0.0, 0.0)
                        }
                    }
                    
                    # Add new complete path
                    masked_traces.append(new_trace)
                    # ---------------------------------
        
        else:
            original_think = self.trace_obj[1]["think_answer"].get('think', '')
            paraphrased_thinks = []
            if original_think:
                think_prompt = think_generator.get_instruction(original_think, n=args.num_masked_solution_traces)
                think_output = think_generator.generate(think_prompt)[0]
                paraphrased_thinks = get_paraphrased_think(think_output)

            input_text = node_generator.get_prompt_text('think_answer', {0: self.trace_obj[0]})
            for pt in paraphrased_thinks:
                input_text_pt = input_text + f"<think> {pt} </think>\n"
                output = node_generator.generate_(input_text_pt, node_generator.answer_stopping_criteria)[0]
                masked_traces.append({
                    0: self.trace_obj[0],
                    1: {"think_answer": {"think": pt, "answer": get_answer(output), "value": 0.9}}
                })
            
        return masked_traces
    
    # def get_rag_confidence(self):
    #     sorted_keys = sorted(self.trace_obj.keys(), key=int)
    #     preds = [item[sorted_keys[-1]]['think_answer']['answer'] for item in self.masked_trace_retrieval_list]
    #     self.rag_confidence = sum(1 for item in preds if item is not None and em_score(item, self.final_answer)) / len(preds)
    #     return self.rag_confidence


    # def to_search_queries(self):
    #     sorted_keys = sorted(self.trace_obj.keys(), key=int)
    #     if len(sorted_keys) == 2:
    #         org = self.trace_obj[sorted_keys[-1]]['think_answer']['think']
    #         para = [item[sorted_keys[-1]]['think_answer']['think'] for item in self.masked_trace_retrieval_list]    
    #     else:
    #         org = self.trace_obj[sorted_keys[-2]]['think_search']['search_query']
    #         para = [item[sorted_keys[-2]]['think_search']['search_query'] for item in self.masked_trace_retrieval_list]
    #     return f"Candidate {self.trace_id}:\nOriginal: {org}\n{para}"
        
    # def to_prediction(self):
    #     sorted_keys = sorted(self.trace_obj.keys(), key=int)
    #     preds = [item[sorted_keys[-1]]['think_answer']['answer'] for item in self.masked_trace_retrieval_list]
    #     return f"Candidate {self.trace_id}: {self.final_answer} | {preds}"


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

    def _filter_rag_consistency(self, question: str, candidates: list[Candidate], aux={}) -> list[Candidate]:
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

    def select(self, question: str, candidates: list[Candidate], gt_answer: str = None, aux={}) -> Candidate:
        print(f"==> Ground truth answer: {gt_answer}")
        unfiltered_candidates = candidates
        print(f"==> Unfiltered answers: {[c.final_answer for c in unfiltered_candidates]}")
        prefiltered_candidates = self._filter_none(candidates)
        prefiltered_candidates = self._filter_long(prefiltered_candidates)
        prefiltered_candidates = self._filter_white_space(prefiltered_candidates)
        prefiltered_candidates = self._filter_specific_words(prefiltered_candidates)
        print(f"==> Pre-filtered answers: {[c.final_answer for c in prefiltered_candidates]}")
        # select the final trajectory through RAG Consistency
        filtered_candidates = self._filter_rag_consistency(question, prefiltered_candidates, aux)
        print(f"==> RAGC-filtered answers: {[c.final_answer for c in filtered_candidates]}")
        winner, filtered_answer2score = self._find_winner_filtered(question, prefiltered_candidates, filtered_candidates, gt_answer)
        return winner, filtered_answer2score

    def inference(self, qid, question, gt_answers, paths):
        all_candidates = []
        for trace_id, trace in enumerate(paths):
            trace_ = trace["trace"]
            trace_ = {int(key): val for key, val in  trace_.items()}
            last_depth_key = list(trace_.keys())[-1]
            last_node_type = list(trace_[last_depth_key].keys())[0] 
            final_answer = trace_[last_depth_key][last_node_type]["answer"]
            final_answer_reward = trace_[last_depth_key][last_node_type]["node_reward"]
            
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
            for c in all_candidates:
                c.get_masked_traces(
                    self.node_generator, self.search_query_generator, self.think_generator,
                    self.retriever, self.args
                )
            
            winner_answer_, answer2score = self.select(question, all_candidates, gt_answers)
            winner_answer = winner_answer_.final_answer if winner_answer_ != None else ''
        
        return winner_answer, answer2score
        