#!/usr/bin/env python3

import re
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import random
import requests
import datasets
import argparse
import jsonlines
import numpy as np
import transformers
from tqdm import tqdm, trange
from collections import defaultdict
from typing import List, Dict, Tuple

from utils.general_utils import set_seed, read_jsonl, read_txt
from run_searchr1.retrieval_local import BM25Retriever, ContrieverRetriever, RerankRetriever, DenseRetriever
from run_searchr1.correctness import em_score, f1_score
from run_searchr1.inference import get_think, get_query, get_answer, _passages2string, StopOnSequence


def options2string_(options_list):
    return '\n'.join(f"{chr(65 + i)}. {option}" for i, option in enumerate(options_list))


class SemanticEquivalenceGenerator:
    def __init__(self, args, model=None, tokenizer=None):
        self.args = args
        self.prompt = read_txt(args.semantic_equivalence_prompt_file)
        if model == None and tokenizer==None:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16, device_map='auto')
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
        else: 
            self.model = model
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

        def cluster_by_meaning(user_query, output_list):
            cluster = []

            for i, answer in enumerate(output_list):
                if i == 0:
                    cluster.append([answer])
                else:
                    prompt = self.semantic_equivalence_prompt
                    prompt += f'\n\nWe are evaluating answers to the question: {user_query}\n'
                    prompt += 'Here are two possible answers:\n'

                    for j, c in enumerate(cluster):
                        tmp_prompt = prompt + f'Possible Answer 1: {answer}\n'
                        tmp_prompt += f'Possible Answer 2: {c[0]}\n'
                        tmp_prompt += 'For this question, is Possible Answer 1 semantically equivalent to Possible Answer 2? Respond with Yes or No.\n'
                        tmp_prompt += 'Response: '
                        
                        response = self.generate(
                            tmp_prompt,
                            max_new_tokens=1,
                            num_return=1,
                            # temperature=0.01,
                        )[0]
                        if 'Yes' in response:
                            c.append(answer)
                            break
                        elif j == len(cluster) - 1:
                            cluster.append([answer])
                            break

            return cluster

        if len(output_list) == 1:
            most_confident_answer = output_list[0]
            confidence = 1
        else:
            cluster = cluster_by_meaning(user_query=user_query, output_list=output_list)
            most_confident_cluster = sorted(cluster, key=len, reverse=True)[0]
            most_confident_answer, confidence = most_confident_cluster[0], len(most_confident_cluster)/sum(map(len, cluster))
            assert confidence > 0 and confidence <= 1

        return most_confident_answer, confidence


def searchr1_discrimination(args):
    print("\n== Search-R1 Discrimination ...")
    print(f"""
        Model name:  {args.model_name_or_path}
        Dataset:     {args.dataset} / {args.subsec} ({args.fraction_of_data_to_use})
        Retriever:   {args.retriever_name} / ({args.retrieval_model_path})
        Seed:        {args.seed}
        Run:         {args.run}
    """.replace('        ', ''))
    
    # === Output files =========================
    entries = os.listdir(args.generation_trees_results_dir)
    query_ids = [entry for entry in entries if os.path.isdir(os.path.join(args.generation_trees_results_dir, entry))]
    sorted_query_ids = sorted(query_ids, key=lambda x: int(x.split('_')[1]))
    
    
    # === generator Model ======================
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path_sr1)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path_sr1, torch_dtype=torch.bfloat16, device_map="auto")
    target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
    stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])
    curr_eos = [151645, 151643] # for Qwen2.5 series models
    curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'
    
    
    # === Static Retriever ===================== 
    if args.retriever_name == 'bm25':
        retriever = BM25Retriever(args)  
    elif args.retriever_name == 'contriever':
        retriever = ContrieverRetriever(args)
    elif args.retriever_name == 'rerank':
        retriever = RerankRetriever(args)
    elif args.retriever_name in ['e5', 'bge']:
        retriever = DenseRetriever(args)
    
    
    # === Semantic clustering ================== 
    se_model = SemanticEquivalenceGenerator(args)
   
   
    # === Prompt ===============================
#     prompt = """Select the best answer from the provided candidates.\
# You must conduct reasoning inside <think> and </think> first every time you get new information.\
# After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>.\
# You can search as many times as your want.\
# If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations.\
# For example, <answer> Beijing </answer>. Question: {question}\nOptions:\n{options}"""


    prompt = """Select the best answer from the provided candidates.
You must conduct reasoning inside <think> and </think> first every time you get new information.\
During reasoning, analyze each answer option to determine why it may be correct or incorrect, considering relevant facts, logic, and context.\
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>.\
You can search as many times as you want.\
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations.\
For example, <answer> Beijing </answer>. Question: {question}\nOptions:\n{options}"""



    # === Inference ============================
    em_evaluation = []
    with open(args.discriminate_results_file, 'w', encoding='utf-8') as inf_file:
        for idx, qid in enumerate(tqdm(sorted_query_ids)):
            if idx == 5:
                break
            
            final_solutions_file = f"{args.generation_trees_results_dir}/{qid}/final_solutions.jsonl"
            trace_js = read_jsonl(final_solutions_file)  
            question = trace_js[0]["trace"]["0"]["user_question"]
            question = question.strip()
            if question[-1] != '?':
                question += '?'
                
            gt_answers = trace_js[0]["trace"]["0"]["ground_truth"]
            options = [s["trace"][list(s["trace"].keys())[-1]]['think_answer']["answer"] for s in trace_js]
            cls_options = se_model.cluster_by_meaning(question, options)
            
            if len(cls_options) == 0:
                pred_answer = ''
            elif len(cls_options) == 1:
                pred_answer = random.choice(cls_options[0])
            else:
                unq_options = [random.choice(cls_) for cls_ in cls_options]
                options_str = options2string_(unq_options)            
                input_prompt = prompt.format(question=question, options=options_str)
            
                if tokenizer.chat_template:
                    input_prompt = tokenizer.apply_chat_template([{"role": "user", "content": input_prompt}], add_generation_prompt=True, tokenize=False)
                
                cnt = 0
                while True:
                    input_ids = tokenizer.encode(input_prompt, return_tensors='pt').to(args.device)
                    attention_mask = torch.ones_like(input_ids)
                    
                    # Generate text with the stopping criteria
                    outputs = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=args.max_new_token,
                        stopping_criteria=stopping_criteria,
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=True,
                        temperature=0.7
                    )

                    if outputs[0][-1].item() in curr_eos:
                        generated_tokens = outputs[0][input_ids.shape[1]:]
                        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                        break

                    generated_tokens = outputs[0][input_ids.shape[1]:]
                    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                            
                    print(output_text)
                    print('---')
                                        
                    tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
                    if tmp_query:
                        search_docs = retriever.search(tmp_query)
                        search_results = _passages2string(search_docs)
                    else:
                        search_docs = []
                        search_results = ''

                    # path.append({
                    #     'think': get_think(output_text),
                    #     'search_query': tmp_query,
                    #     'docs': search_docs
                    # })

                    search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
                    input_prompt += search_text
                    cnt += 1
                
                print(output_text)
                pred_answer = get_answer(output_text)
                
            item = {
                "qid": qid,
                "query": question,
                "gt_answers": gt_answers,
                "final_answer": pred_answer,
                "cluster_options": cls_options,
            }
            inf_file.write(json.dumps(item) + '\n')
            
            correctness_em = em_score(pred_answer, gt_answers)
            em_evaluation.append(correctness_em)
        
        
    # === Print results ========================
    print("\nEvaluation Result:")
    print(f"EM: {np.mean(em_evaluation)*100}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--model_name_or_path_sr1', type=str, default='PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo')
    parser.add_argument('--max_new_token', type=int, default=512)
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='bamboogle', choices=[
        'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
    
    # Retriever
    parser.add_argument('--retriever_name', type=str, default='rerank', choices=[
        'bm25', 'contriever', 'rerank', 'e5'
    ])
    parser.add_argument('--corpus_path', type=str, default='data/search_r1_files/wiki-18.jsonl')
    parser.add_argument('--index_path', type=str, default='data/search_r1_files/bm25', choices=[
        'data/search_r1_files/bm25',          # For BM25 & Rerank
        'data/search_r1_files/e5_Flat.index', # For E5
    ])
    parser.add_argument("--retrieval_model_path", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2", choices=[
        "intfloat/e5-base-v2" # For E5
        "cross-encoder/ms-marco-MiniLM-L12-v2" # For Rerank | cross-encoder/ms-marco-MiniLM-L-6-v2
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
    parser.add_argument('--run', type=str, default='run_5 (edited_prompt_roll4)')
    parser.add_argument("--seed", type=int, default=10)
    
    args = parser.parse_args()
    
    # === Files ====================
    model_ = args.model_name_or_path.split('/')[-1]
    output_dir = f"run_output/{args.run}/{model_}/{args.dataset}_{args.subsec}/{args.retriever_name}"
    args.generation_trees_results_dir = f'{output_dir}/generation_trees'
    args.discriminate_results_file = f"{output_dir}/sr1_discriminate_results_v2.jsonl"
    os.makedirs(args.generation_trees_results_dir, exist_ok=True)
    
    args.semantic_equivalence_prompt_file = "prompts_mcts/semantic_equivalence_prompt_template.txt"
    
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
    searchr1_discrimination(args)
    
    
    # python run_mcts/searchr1_discrimination.py

