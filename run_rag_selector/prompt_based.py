#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import re
import ast
import json
import torch
import argparse
import numpy as np
import transformers
from tqdm import tqdm
from accelerate import Accelerator

from utils.general_utils import set_seed
from run_rag_methods.src.correctness import em_score, subem_score, f1_score
from applications.rag_selector.confidence_in_input.listwise_run import data_creation


def rag_selector(args):
    # === MultiGPU setup ==========
    accelerator = Accelerator()
    device = accelerator.device
    
    # === Read Models =============
    generation_model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path, dtype=torch.bfloat16).to(device)
    generation_tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # === RAGSelector =============
    class RAGSelector:
        def __init__(self, generation_model, generation_tokenizer, device, args):
            self.system_prompt = 'You are a helpful assistant.'
            self.generator = generation_model
            self.tokenizer = generation_tokenizer
            self.device = device
            self.args = args

        def generate(self,
            messages,
            stopping_criteria=None,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
        ):
            if self.tokenizer.chat_template:
                input_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
            
            input_ids = self.tokenizer.encode(input_prompt, return_tensors='pt').to(self.device)
            attention_mask = torch.ones_like(input_ids)
            outputs = self.generator.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=do_sample,
                temperature=temperature
            )
            output_ = outputs[0]
            generated_tokens = output_[input_ids.shape[1]:]
            output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return output_, output_text
        
        def get_think(self, text):
            pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
            matches = pattern.findall(text)
            if matches:
                return matches[0]
            else:
                return None
        
        def get_prediction(self, text):
            pattern = re.compile(r"<prediction>(.*?)</prediction>", re.DOTALL)
            matches = pattern.findall(text)
            if matches:
                return matches[0]
            else:
                return None
        
        def get_instruction(self, question, candidates):
            input_text = ""
            input_text += "You are an expert in the final answer selection task.\n"
            input_text += "Your task is to identify and select the best answer to the question from a list of candidates, each with a given confidence score.\n"
            input_text += "You must select the final answer from the candidates provided. Do not generate a new one.\n\n"            
            input_text += "Your output must include:\n"
            input_text += "- Exactly one reasoning step explaining your choice, wrapped in a single pair of <think> and </think> tags.\n"
            input_text += "- The selected final answer, wrapped in a single pair of <prediction> and </prediction> tags.\n\n"
            
            input_text += f"<question>{question}</question>\n"
            input_text += "<candidates>\n"
            for i, c in enumerate(candidates):
                input_text += f"[{i+1}] {c[0]}, with confidence {c[1]}\n"    
            input_text += "</candidates>\n"
            
            return input_text
        
        def inference(self, question, candidates, generation_temp=0.7):
            input_prompt = self.get_instruction(question, candidates)
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": input_prompt}
            ]
            _, output_text = self.generate(messages, temperature=generation_temp)
            think = self.get_think(output_text)
            prediction = self.get_prediction(output_text)
            
            return think, prediction
    
    # === Load dataset ============
    RAG_METHODS = ['self_ask', 'react', 'search_o1', 'research', 'search_r1']
    dataset = data_creation(args)
    print('---')
    print(f"Train sample: {dataset['train'][0]}")
    print('---')
    print(f"Test sample:  {dataset['test'][0]}")
    print('---')
    
    # === Inference ... ===========
    em_evaluation = []
    rag_selector = RAGSelector(generation_model, generation_tokenizer, device, args)
    with open(args.result_file, "w", encoding="utf-8") as fout:
        for idx, sample in enumerate(tqdm(dataset["test"])):
            # if idx == 5:
            #     break
            
            qid, query = sample['qid'], sample['query']
            candidates, ground_truth = [], []
            for key, val in sample.items():
                if key in RAG_METHODS:
                    parsed = ast.literal_eval(val)
                    candidates.append((parsed[0], parsed[2], parsed[1]))
                    if parsed[1] == 1:
                        ground_truth.append(parsed[0])
                        
            think, prediction = rag_selector.inference(query, candidates)

            if len(ground_truth) > 0:
                correctness_em = em_score(prediction, ground_truth)
                correctness_f1 = f1_score(prediction, ground_truth)
            else:
                correctness_em = 0
                correctness_f1 = {'f1': 0, 'precision': 0, 'recall': 0}

            item = {
                'qid': qid, 'query': query,
                'prediction': prediction, 'think': think,
                'em': correctness_em, 'f1': correctness_f1,
                'candidates': candidates
            }
            fout.write(json.dumps(item) + "\n")
            em_evaluation.append(correctness_em)
    
    print("\nEvaluation Result:")
    print(f"EM: {np.mean(em_evaluation)*100}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    # Dataset
    parser.add_argument('--dataset', type=str, default='hotpotqa', choices=['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'])
    parser.add_argument('--subsec', type=str, default='dev', choices=['train', 'dev', 'test', 'validation'])
    # Retriever
    parser.add_argument('--retriever_name', type=str, default='rerank_l6', choices=[
        'bm25', 'contriever', 'rerank_l6', 'rerank_l12', 'e5', 'bge', 'reasonir'
    ])    
    # Consistency Generation Methods (answer list)
    parser.add_argument('--consistency_method', type=str, default='rag_consistency', choices=[
        'fa_consistency', 'rrr_consistency', 'reasoning_consistency', 'self_consistency', 'rag_consistency'
    ])
    # Others
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--run', type=str, default='run_3 (rag_methods_500)')
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument('--use_counter', action='store_false')
    args = parser.parse_args()
    
    args.result_file = f"run_output/{args.run}/rag_selection_reward_modeling/{args.dataset}_{args.retriever_name}_{args.consistency_method}/{args.subsec}_inference_results_prompt_based.jsonl"
    
    # === Run Steps ================
    set_seed(args.seed)
    rag_selector(args)
    
    # python applications/rag_selector/prompt_based.py
    
    
