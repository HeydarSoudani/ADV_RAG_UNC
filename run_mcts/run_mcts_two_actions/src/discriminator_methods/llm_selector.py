#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch

from run_mcts_two_actions.src.discriminator_methods.basic_discriminator import BasicDiscriminator, Candidate
from utils.general_utils import get_think, get_answer


class LlmSelector(BasicDiscriminator):
    def __init__(self, args, device):
        super().__init__(args, device)
    
    def get_prefiltered_candidates(self, question: str, candidates: list[Candidate], gt_answer: str = None, aux={}) -> Candidate:
        print(f"==> Ground truth answer: {gt_answer}")
        unfiltered_candidates = candidates
        print(f"==> Unfiltered answers: {[c.final_answer for c in unfiltered_candidates]}")
        prefiltered_candidates = self._filter_none(candidates)
        prefiltered_candidates = self._filter_long(prefiltered_candidates)
        prefiltered_candidates = self._filter_white_space(prefiltered_candidates)
        prefiltered_candidates = self._filter_specific_words(prefiltered_candidates)
        print(f"==> Pre-filtered answers: {[c.final_answer for c in prefiltered_candidates]}")
        return prefiltered_candidates
    
    def get_unique_docs(self, traces):
        docs = {}
        for trace_id, trace in enumerate(traces):
            trace_ = trace["trace"]
            trace_ = {int(key): val for key, val in  trace_.items()}
            for step_key, step_val in trace_.items():
                if 'think_search' in step_val:
                    for doc in step_val['think_search']['retrieved_documents']:
                        docs[doc['id']] = doc['contents']
        return docs
    
    def get_input_prompt(self, question, docs, candidates):
        input_text = ''
        input_text += 'You are an answer selector for a question-answering task.\n'
        input_text += 'Your task has two steps:\n'
        input_text += '1. Deeply analyze each retrieved document one by one, along with your internal knowledge.\n'
        input_text += '2. Select the most accurate answer from the provided candidates.\n'
        input_text += 'Retrieved documents (if any) are enclosed in <information> tags.\n'
        input_text += 'Answer candidates are enclosed in <candidate> tags.\n'
        input_text += 'All reasoning must be placed inside a single <think>...</think> block.\n'
        input_text += 'You MUST select exactly one of the provided candidates. Do NOT generate new answers. Do NOT say none apply.\n'
        input_text += 'Your output must contain ONLY the following tags, in this exact order:\n'
        input_text += '<think> your deep reasoning here </think>\n'
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
    
    # def _get_most_likely_answer(self, user_query: str, output_list: list[str]):
    #     assert len(output_list) > 0

    #     def cluster_by_meaning(user_query, output_list):
    #         cluster = []

    #         for i, answer in enumerate(output_list):
    #             if i == 0:
    #                 cluster.append([answer])
    #             else:
    #                 prompt = self.semantic_equivalence_prompt
    #                 prompt += f'\n\nWe are evaluating answers to the question: {user_query}\n'
    #                 prompt += 'Here are two possible answers:\n'

    #                 for j, c in enumerate(cluster):
    #                     tmp_prompt = prompt + f'Possible Answer 1: {answer}\n'
    #                     tmp_prompt += f'Possible Answer 2: {c[0]}\n'
    #                     tmp_prompt += 'For this question, is Possible Answer 1 semantically equivalent to Possible Answer 2? Respond with Yes or No.\n'
    #                     tmp_prompt += 'Response: '
                        
    #                     response = self.generate(
    #                         tmp_prompt,
    #                         max_new_tokens=1,
    #                         num_return=1,
    #                         # temperature=0.01,
    #                     )[0]
    #                     if 'Yes' in response:
    #                         c.append(answer)
    #                         break
    #                     elif j == len(cluster) - 1:
    #                         cluster.append([answer])
    #                         break

    #         return cluster

    #     if len(output_list) == 1:
    #         most_confident_answer = output_list[0]
    #         confidence = 1
    #     else:
    #         cluster = self.se_model.cluster_by_meaning(user_query=user_query, output_list=output_list)
    #         most_confident_cluster = sorted(cluster, key=len, reverse=True)[0]
    #         most_confident_answer, confidence = most_confident_cluster[0], len(most_confident_cluster)/sum(map(len, cluster))
    #         assert confidence > 0 and confidence <= 1

    #     return most_confident_answer, confidence
    
    def llm_selector(self, qid: str, question: str, candidates: list[str], docs, aux={}) -> list[Candidate]:
        input_prompt = self.get_input_prompt(question, docs, candidates)
        # print(input_prompt)
        # print('-')
        
        initial_output_list = self.generate(input_prompt, num_return=5, temperature=1.0)
        # print(initial_output_list)
        # print('------')
        
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
        
        most_likely_answer, conf_value = self.se_model._get_most_likely_answer(user_query=question, output_list=answer_list)
        print(f"==> Winner answer: {most_likely_answer} | {conf_value}\n")
        
        return most_likely_answer, conf_value
    
    def inference(self, qid, question, gt_answers, paths):
        all_candidates = []
        unique_docs = self.get_unique_docs(paths)
        
        for trace_id, trace in enumerate(paths):
            trace_ = trace["trace"]
            trace_ = {int(key): val for key, val in  trace_.items()}
            # trace_text = self.trace2text(trace_)
            last_depth_key = list(trace_.keys())[-1]
            last_node_type = list(trace_[last_depth_key].keys())[0] 
            final_answer = trace_[last_depth_key][last_node_type]["answer"]
            final_answer_reward = trace_[last_depth_key][last_node_type]["node_reward"]
            
            candidate = Candidate('', [], final_answer, trace_id, trace_reward=final_answer_reward)
            all_candidates.append(candidate)
        
        prefiltered_candidates = self.get_prefiltered_candidates(question, all_candidates, gt_answers)
        
        if len(prefiltered_candidates) > 0:
            answer2candidates, answer2confidence, _ = self.group_candidates_by_answer(
                question, prefiltered_candidates, self.args.rc_criteria
            )
            most_confident_answer = max(answer2candidates.keys(), key=lambda x: answer2confidence[x])
            highest_confidence = answer2confidence[most_confident_answer]
            assert highest_confidence > 0
            
            unique_answers = list(answer2candidates.keys())
            print(f"==> Unique answers: {unique_answers}")
            
            # === Decision
            if highest_confidence > self.args.threshold:
                print("You are very confident. Skipping...")
                winner_answer = most_confident_answer if most_confident_answer != None else ''
                answer2score = {winner_answer: 1.0}
            else:
                winner_answer, conf = self.llm_selector(qid, question, unique_answers, unique_docs)
                answer2score = {winner_answer: conf}
        else:
            winner_answer = ''
            answer2score = {winner_answer: 1.0}
        
        return winner_answer, answer2score
