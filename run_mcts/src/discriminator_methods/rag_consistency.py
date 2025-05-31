#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from copy import deepcopy

from run_mcts.src.discriminator_methods.basic_discriminator import BasicDiscriminator, Candidate

class RagConsistency(BasicDiscriminator):
    def __init__(self, args, device):
        super().__init__(args, device)

    def _filter_rag_consistency(self, question: str, candidates: list[Candidate], aux={}) -> list[Candidate]:
        pass

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
        print(f"==> RC-filtered answers: {[c.final_answer for c in filtered_candidates]}")
        winner, filtered_answer2score = self._find_winner_filtered(question, prefiltered_candidates, filtered_candidates, gt_answer)
        return winner, filtered_answer2score

    def inference(self, qid, question, gt_answers, paths):
        all_candidates = []
        for trace_id, trace in enumerate(paths):
            trace_ = trace["trace"]
            trace_ = {int(key): val for key, val in  trace_.items()}
            trace_text = self.trace2text(trace_)
            last_depth_key = list(trace_.keys())[-1]
            last_node_type = list(trace_[last_depth_key].keys())[0] 
            final_answer = trace_[last_depth_key][last_node_type]["answer"]
            final_answer_reward = trace_[last_depth_key][last_node_type]["node_reward"]
            
            # TODO:
            
            
            candidate = Candidate(
                trace_text, deepcopy(masked_trace_text_list),
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
        