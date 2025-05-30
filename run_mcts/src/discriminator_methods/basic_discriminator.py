#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import torch
import transformers
from collections import defaultdict

from run_mcts.src.models.semantic_equivalence import SemanticEquivalenceGenerator
# from run_mcts.src.discriminator_methods.reasoning_consistency import Candidate
from run_rag_methods.src.correctness import normalize_answer


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


class BasicDiscriminator:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.generator = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16).to(device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.se_model = SemanticEquivalenceGenerator(args, device, self.generator, self.tokenizer)
    
    def _filter_none(self, candidates: list[Candidate]) -> list[Candidate]:
        candidates = [c for c in candidates if c.final_answer is not None]
        return candidates

    def _filter_long(self, candidates: list[Candidate]) -> list[Candidate]:
        candidates = [c for c in candidates if len(c.final_answer) <= 80]
        return candidates

    def _filter_white_space(self, candidates: list[Candidate]) -> list[Candidate]:
        candidates = [c for c in candidates if c.final_answer.strip()]
        return candidates

    def _filter_specific_words(self, candidates: list[Candidate]) -> list[Candidate]:
        words = ['can not answer', 'CAN NOT ANSWER', 'not enough information', 'not enough information provided', 'unknown', 'more information needed', 'none', 'not specified in the given information', 'information not specified', 'no direct information available in current context', 'no direct information available in the knowledge base.']
        filtered_candidates = []
        for c in candidates:
            normalized_c = normalize_answer(c.final_answer)
            if not any(w in normalized_c for w in words):
                filtered_candidates.append(c)
        return filtered_candidates
    
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
                model_answer = self.get_answer_searchr1(c)
                has_existed = False
                for existing_answer in answer2completions.keys():
                    if self.se_model.check_answers_equiv(question, model_answer, existing_answer):
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
                self.get_answer_searchr1(most_confident_completion),
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

    def _find_winner_filtered(
        self, question:str, unfiltered_candidates: list[Candidate], filtered_candidates: list[Candidate], gt_answer: str = None
    ) -> Candidate:
        if len(filtered_candidates) == 0:
            answer2candidates, answer2confidence, _ = self.group_candidates_by_answer(
                question, unfiltered_candidates, self.args.rc_criteria
            )
            answer2score = dict(answer2confidence)
            if answer2confidence:
                most_confident_answer = max(answer2confidence.keys(), key=lambda x: answer2confidence[x], default=None)
                winner = answer2candidates[most_confident_answer][0]
                print(f"==> Winner answer: {most_confident_answer}\n")
            else:
                winner = None
        elif len(filtered_candidates) == 1:
            winner = filtered_candidates[0]
            answer2score = {winner.final_answer: 1.0}
            print(f"==> Winner answer: {winner.final_answer}\n")
        # elif not any(self.se_model.check_answers_equiv(question, c.final_answer, gt_answer[0]) for c in filtered_candidates):
        #     winner = None
        #     print(f"==> Winner answer: None")
        else:
            filtered_answer2score = self._calculate_scores(question, unfiltered_candidates, filtered_candidates)
            winner_answer = max(filtered_answer2score.keys(), key=lambda x: filtered_answer2score[x])
            print(f"==> Winner answer: {winner_answer}")
            winner = next(
                (c for c in filtered_candidates if self.se_model.check_answers_equiv(question, c.final_answer, winner_answer)), 
                None
            )
            answer2score = filtered_answer2score
        return winner, answer2score

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

    def get_answer_searchr1(self, text):
        pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            return matches[-1]
        else:
            return None
