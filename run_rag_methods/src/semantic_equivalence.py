#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import transformers
from utils.general_utils import read_txt


class SemanticEquivalenceGenerator:
    def __init__(self, args, device, generator=None, tokenizer=None):
        self.args = args
        self.prompt = read_txt(args.semantic_equivalence_prompt_file)
        if generator is None and tokenizer is None:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16).to(device)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
        else:
            self.model = generator
            self.tokenizer = tokenizer

        self.eos_token_ids = self.model.config.eos_token_id

    def _filter_none(self, candidates):
        candidates = [c for c in candidates if c is not None]
        return candidates

    def _filter_long(self, candidates):
        candidates = [c for c in candidates if len(c) <= 80]
        return candidates

    def _filter_white_space(self, candidates):
        candidates = [s for s in candidates if s.strip()]
        return candidates

    def _filter_stop_words(self, candidates):
        filter_words = {"unknown", "n/a", "none", "not enough information provided"}  # use lowercase, no spaces
        candidates = [s for s in candidates if s.strip() and s.strip().lower() not in filter_words]
        return candidates

    def generate(self,
        input_text,
        max_new_tokens=1024,
        temperature:float = 0.7,
        do_sample:bool = True
    ):
        if self.tokenizer.chat_template:
            input_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": input_text}],
                add_generation_prompt=True,
                tokenize=False
            )
        input_ids = self.tokenizer.encode(input_prompt, return_tensors='pt').to(self.model.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=do_sample,
                temperature=temperature
            )
            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return output_text

    def cluster_by_meaning(self, user_query, output_list):
        cluster = []
        for i, answer in enumerate(output_list):
            if i == 0:
                cluster.append([answer])
            else:
                prompt = self.prompt
                prompt += f'\n\nWe are evaluating answers to the question: {user_query}\n'
                prompt += 'Here are two possible answers:\n'

                for j, c in enumerate(cluster):
                    tmp_prompt = prompt + f'Possible Answer 1: {answer}\n'
                    tmp_prompt += f'Possible Answer 2: {c[0]}\n'
                    tmp_prompt += 'For this question, is Possible Answer 1 semantically equivalent to Possible Answer 2? Respond with Yes or No.\n'
                    tmp_prompt += 'Response: '
                    response = self.generate(tmp_prompt, max_new_tokens=1)
                    if 'Yes' in response:
                        c.append(answer)
                        break
                    elif j == len(cluster) - 1:
                        cluster.append([answer])
                        break

        return cluster

    def check_answers_equiv(self, question:str, answer_a: str, answer_b: str):
        # raise NotImplementedError
        if answer_a is None or answer_b is None:
            return False
        assert isinstance(answer_a, str) and isinstance(answer_b, str)
        assert self.model is not None and self.prompt is not None and question is not None

        equiv_prompt = self.prompt

        # return answer_a == answer_a or fuzz.token_sort_ratio(answer_a, answer_b) >= 90
        equiv_prompt += f'\n\nWe are evaluating answers to the question: {question}\n'
        equiv_prompt += 'Here are two possible answers:\n'

        equiv_prompt += f'Possible Answer 1: {answer_a}'
        equiv_prompt += f'Possible Answer 2: {answer_b}'
        equiv_prompt += 'For this question, is Possible Answer 1 semantically equivalent to Possible Answer 2? Respond with Yes or No.\n'
        equiv_prompt += 'Response: '
        response = self.generate(equiv_prompt, max_new_tokens=1)
        return True if 'Yes' in response else False

    def _get_most_likely_answer(self, user_query: str, output_list: list[str]):
        assert len(output_list) > 0

        if len(output_list) == 1:
            most_confident_answer = output_list[0]
            confidence = 1
        else:
            cluster = self.cluster_by_meaning(user_query=user_query, output_list=output_list)
            most_confident_cluster = sorted(cluster, key=len, reverse=True)[0]
            most_confident_answer, confidence = most_confident_cluster[0], len(most_confident_cluster)/sum(map(len, cluster))
            assert confidence > 0 and confidence <= 1

        return most_confident_answer, confidence
