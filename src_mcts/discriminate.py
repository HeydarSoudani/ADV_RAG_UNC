
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.general_utils import read_txt, read_json

class Candidate:
    def __init__(
        self,
        solution_trace,
        masked_solution_trace_list,
        final_step,
        final_answer,
        id,
        freq=1,
        trace_reward=1.0,
        c_type="default",
    ):
        self.solution_trace = solution_trace
        self.masked_solution_trace_list = masked_solution_trace_list
        self.final_step = final_step
        self.final_answer = final_answer
        self.id = id
        self.freq = freq
        self.trace_reward = trace_reward
        self.c_type = c_type

    def __str__(self):
        return f"Candidate {self.id}: {self.final_answer}"

    def to_dict(self):
        return {
            "solution_trace": self.solution_trace,
            "masked_solution_trace_list": self.masked_solution_trace_list,
            "final_step": self.final_step,
            "final_answer": self.final_answer,
            "id": self.id,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            solution_trace=data["solution_trace"],
            masked_solution_trace_list=data["masked_solution_trace_list"],
            final_step=data["final_step"],
            final_answer=data["final_answer"],
            id=data["id"],
        )

def group_candidates_by_answer(candidates: list[Candidate], evaluator, criteria="freq"):
    """Return answer2candidates, answer2confidence, answer2cnt."""
    answer2candidates = {}
    answer2confidence = defaultdict(float)
    answer2cnt = defaultdict(int)

    for c in candidates:
        has_existed = False
        for existing_answer in answer2candidates.keys():
            if evaluator.check_answers_equiv(c.final_answer, existing_answer):
                has_existed = True
                answer2candidates[str(existing_answer)].extend([c] * c.freq)
                answer2confidence[str(existing_answer)] += c.trace_reward if criteria == "reward" else c.freq
                answer2cnt[str(existing_answer)] += c.freq
                break

        if not has_existed:
            if str(c.final_answer) in answer2candidates:
                answer2candidates[str(c.final_answer)].extend([c] * c.freq)
            else:
                answer2candidates[str(c.final_answer)] = [c] * c.freq
            answer2confidence[str(c.final_answer)] += c.trace_reward if criteria == "reward" else c.freq
            answer2cnt[str(c.final_answer)] += c.freq

    assert all(answer2cnt[ans] == len(answer2candidates[ans]) for ans in answer2cnt.keys())
    assert float(sum([candidate.trace_reward for candidate in candidates])) == float(
        sum([answer2confidence[ans] for ans in answer2confidence.keys()])
    )

    candidates_count = sum([candidate.freq for candidate in candidates])
    for ans in answer2confidence.keys():
        answer2confidence[ans] /= candidates_count

    return answer2candidates, answer2confidence, answer2cnt

class Discriminator:
    def __init__(self, args, evaluator):
        self.args = args
        self.evaluator = evaluator

        self.fewshot_config = None
        self.fewshot_template = None
        self.fewshot_prompt = None
        self.stop_tokens = ['\n</Answer>']        

    def _filter_none(self, candidates: list[Candidate]) -> list[Candidate]:
        candidates = [c for c in candidates if c.final_answer is not None]
        return candidates

    def _filter_long(self, candidates: list[Candidate]) -> list[Candidate]:
        candidates = [c for c in candidates if len(c.final_answer) <= 100]
        return candidates

    def _filter_reasoning_consistency(
        self, gen_model, problem: str, candidates: list[Candidate], aux={}
    ) -> list[Candidate]:
        # problem_id = aux["problem_id"]
        # file_idx = aux["file_idx"]

        prompt_template = self.fewshot_template
        fewshot_examples = self.fewshot_prompt
        stop_tokens = self.stop_tokens

        assert all(
            len(c.masked_solution_trace_list) == self.args.num_masked_solution_traces
            for c in candidates
            if c.c_type == "default"
        )
        gen_input_list = []
        ground_truth_list = []
        c_completion_num_list = []
        for c in candidates:
            for masked_solution_trace in c.masked_solution_trace_list:
                for _ in range(self.args.rc_n_completions):
                    gen_input_list.append(
                        prompt_template.format(examples=fewshot_examples, instruction=problem) + masked_solution_trace
                    )
                    ground_truth_list.append(c.final_answer)
            c_completion_num_list.append(len(c.masked_solution_trace_list) * self.args.rc_n_completions)
        """gen_input_list:
        [c1_mask1, c1_mask2, ..., c2_mask1, c2_mask2, ..., ......, ct_mask1, ct_mask2, ...]
        """

        # Manually split into batches
        batch_size = self.args.max_num_seqs // self.args.rc_n_completions // 2
        gen_output_list = []
        for start_idx in range(0, len(gen_input_list), batch_size):
            end_idx = start_idx + batch_size
            sub_gen_input_list = gen_input_list[start_idx:end_idx]
            sub_gen_output_list = self._gen_func(
                gen_model=gen_model,
                gen_input=sub_gen_input_list,
                temperature=self.args.rc_temperature,
                n=1,
                max_tokens=512,
                stop_tokens=stop_tokens + ["\n"],
            )
            gen_output_list.extend(sub_gen_output_list)

        # return gen_output_list
        # with open(os.path.join(self.args.discriminate_results_dir, f"problem-{problem_id}.json"), "w") as f:
        #     js = {"problem_id": problem_id, "file_idx": file_idx, "gen_output_list": gen_output_list}
        #     json.dump(js, f)

        # """gen_output_list:
        # [[c1_mask1_o1, c1_mask1_o2, ...], [c1_mask2_o1, c1_mask2_o2, ...], ..., [ct_mask1_o1, ct_mask1_o2, ...], [ct_mask2_o1, ct_mask2_o2, ...], ...]
        # """

        if all(isinstance(item, list) for item in gen_output_list):
            completion_list = []
            for n_completions in gen_output_list:
                for completion in n_completions:
                    completion_list.append(completion)
            assert len(completion_list) == self.args.rc_n_completions * self.args.num_masked_solution_traces * len(
                candidates
            )
            candidate_group_size = self.args.rc_n_completions * self.args.num_masked_solution_traces
        elif all(isinstance(item, str) for item in gen_output_list):
            completion_list = gen_output_list
            candidate_group_size = self.args.num_masked_solution_traces

        answer_list = [
            self.evaluator.extract_answer_from_model_completion(completion) for completion in completion_list
        ]

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
            candidate_group_size = len(c.masked_solution_trace_list)
            num_consistent = 0
            if self.args.rc_mode == "maj":
                answer = self.evaluator.find_most_confident_answer(completion_group)[0]
                if self.evaluator.check_answers_equiv(gt_answer[-1], answer):
                    consistent_candidates.append(c)
            else:
                for answer, gt_a in zip(answer_group, gt_answer):
                    if self.evaluator.check_answers_equiv(gt_a, answer):
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

    def _gen_func(self, gen_model, gen_input, temperature: float, n: int = 1, max_tokens: int = 768, stop_tokens=None):
        if temperature == 0.0:
            n = 1

        response = generate_with_vLLM_model(
            model=gen_model, input=gen_input, temperature=temperature, n=n, max_tokens=max_tokens, stop=stop_tokens
        )
        if n == 1:
            if isinstance(gen_input, str):
                return response[0].outputs[0].text
            elif isinstance(gen_input, list):
                return [r.outputs[0].text for r in response]
        elif n > 1:
            if isinstance(gen_input, str):
                return [o.text for o in response[0].outputs]
            elif isinstance(gen_input, list):
                return [[o.text for o in r.outputs] for r in response]

    def _calculate_scores(self, unfiltered_candidates: list[Candidate], filtered_candidates: list[Candidate]) -> dict:
        _, filtered_answer2confidence, filtered_answer2cnt = group_candidates_by_answer(
            filtered_candidates, self.evaluator, self.args.rc_criteria
        )
        print(f"==> Confidence: {filtered_answer2confidence}")
        _, _, unfiltered_answer2cnt = group_candidates_by_answer(
            unfiltered_candidates, self.evaluator, self.args.rc_criteria
        )

        filtered_answer2survival_rate = {}
        for filtered_ans in filtered_answer2cnt.keys():
            has_existed = False
            for unfiltered_ans in unfiltered_answer2cnt.keys():
                if self.evaluator.check_answers_equiv(filtered_ans, unfiltered_ans):
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
                if self.evaluator.check_answers_equiv(filtered_ans, unfiltered_ans):
                    has_existed = True
                    filtered_answer2score[filtered_ans] = (
                        filtered_answer2confidence[filtered_ans] + filtered_answer2survival_rate[filtered_ans]
                    )
                    break
            if not has_existed:
                filtered_answer2score[filtered_ans] = 0.0

        print(f"==> Scores: {filtered_answer2score}")

        return filtered_answer2score

    def _find_winner_filtered(
        self, unfiltered_candidates: list[Candidate], filtered_candidates: list[Candidate], gt_answer: str = None
    ) -> Candidate:
        if len(filtered_candidates) == 0:
            answer2candidates, answer2confidence, _ = group_candidates_by_answer(
                unfiltered_candidates, self.evaluator, self.args.rc_criteria
            )
            most_confident_answer = max(answer2confidence.keys(), key=lambda x: answer2confidence[x])
            winner = answer2candidates[most_confident_answer][0]
            print(f"==> Winner answer: {most_confident_answer}\n")
        elif len(filtered_candidates) == 1:
            winner = filtered_candidates[0]
            print(f"==> Winner answer: {winner.final_answer}\n")
        elif not any(self.evaluator.check_answers_equiv(c.final_answer, gt_answer) for c in filtered_candidates):
            winner = None
            print(f"==> Winner answer: None")
        else:
            filtered_answer2score = self._calculate_scores(unfiltered_candidates, filtered_candidates)
            winner_answer = max(filtered_answer2score.keys(), key=lambda x: filtered_answer2score[x])
            print(f"==> Winner answer: {winner_answer}")
            winner = next(
                c for c in filtered_candidates if self.evaluator.check_answers_equiv(c.final_answer, winner_answer)
            )

        return winner

class MajorityVoteDiscriminator(Discriminator):
    def __init__(self, args, evaluator):
        super().__init__(args, evaluator)
        self.args = args
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=False
        )

    def select(self, problem: str, candidates: list[Candidate], gt_answer: str = None, aux={}) -> Candidate:
        print(f"==> Ground truth answer: {gt_answer}")

        unfiltered_candidates = candidates
        print(f"==> Unfiltered answers: {[c.final_answer for c in unfiltered_candidates]}")
        # candidate: [1, 2, 3, 4, 5, None, paosdifjpsod]
        prefiltered_candidates = self._filter_none(candidates)
        prefiltered_candidates = self._filter_long(prefiltered_candidates)
        # prefiltered_candidates: [1, 2, 3, 4, 5]
        print(f"==> Pre-filtered answers: {[c.final_answer for c in prefiltered_candidates]}")
        
        # select the final trajectory through Reasoning Consistency
        if self.args.extend_rc_mode == 'original':
            filtered_candidates = self._filter_reasoning_consistency(self.model, problem, prefiltered_candidates, aux)
            print(f"==> RC-filtered answers: {[c.final_answer for c in filtered_candidates]}")
        elif self.args.extend_rc_mode == 'BoN':
            filtered_candidates = self._BoN_filter(self.model, problem, prefiltered_candidates, self.best_of, aux)
            print(f"==> BoN-filtered answers: {[c.final_answer for c in filtered_candidates]}")
        elif self.args.extend_rc_mode == 'majority_vote':
            filtered_candidates = []
        else:
            raise NotImplementedError
        # filtered_candidates: [1, 2, 3]
        
        return self._find_winner_filtered(prefiltered_candidates, filtered_candidates, gt_answer)

