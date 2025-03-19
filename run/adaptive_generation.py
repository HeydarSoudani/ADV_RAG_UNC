#!/usr/bin/env python3

import re
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import random
import argparse
from tqdm import tqdm

from src.dataset import BaseDataset
from src.rag import *
from src.evaluate import CorrectnessEval
from utils.utils import set_seed


def adaptive_generation(args):
    print("\n== Adaptive Generation ...")
    print(f"""
        Model name:  {args.model_name_or_path}
        Dataset:     {args.dataset}/{args.subsec} ({args.fraction_of_data_to_use})
        RAG Method:  {args.rag_method} 
        Retriever:   {args.retriever_model}
        Hallu_thre:  {args.hallucination_threshold}
        Seed:        {args.seed}
    """.replace('        ', ''))
    
    
    # === Output files ==========================
    model_ = args.model_name_or_path.split('/')[-1]
    ret_method_ = '' if args.rag_method == 'no_retrieval' else f"{args.retriever_model}_"
    generations_output_file = f'{args.output_dir}/{model_}/{args.dataset}_{args.subsec}/{args.rag_method}/{ret_method_}generations.jsonl'
    generation_path_output_file = f'{args.output_dir}/{model_}/{args.dataset}_{args.subsec}/{args.rag_method}/{ret_method_}generation_path.jsonl'
    results_output_file = f'{args.output_dir}/{model_}/{args.dataset}_{args.subsec}/{args.rag_method}/{ret_method_}results.json'
    

    # === Dataset & Metric Setup ================
    correctness = CorrectnessEval()
    dataset_ = BaseDataset(args.dataset, args.subsec, args.fraction_of_data_to_use)
    dataset = dataset_.dataset
    # fewshot_examplers = dataset_.examplers[:args.fewshot] if len(dataset_.examplers) > 0 else []
    fewshot_examplers = (
        random.sample(dataset_.examplers, args.fewshot)
        if len(dataset_.examplers) >= args.fewshot
        else dataset_.examplers
    )
    
    sample_index = 0
    print(f"Dataset example {sample_index}:")
    print(f"Id:             {dataset[sample_index]['qid']}")
    print(f"Question:       {dataset[sample_index]['question']}")
    print(f"Answers:        {dataset[sample_index]['ground_truths']}")
    print(f"Gold Context: \n{dataset[sample_index]['positive_ctxs'][0]}\n\n")
 
 
    # === Select RAG Method =====================
    if args.rag_method == "no_retrieval":
        model = NoRAG(args)
    elif args.rag_method == "single_retrieval":
        model = SingleRAG(args)
    elif args.rag_method in ["fix_length_retrieval", "fix_sentence_retrieval"]:
        model = FixLengthRAG(args)
    elif args.rag_method == 'flare':
        model = FLARE_RAG(args)
    elif args.rag_method == 'dragin':
        model = DRAGIN_RAG(args)
    else:
        raise NotImplementedError
    
    
    # === Generation ============================
    def get_answer(text):
        parts = text.split("the answer is", 1)  # Split at the first occurrence
        pred = parts[1].strip() if len(parts) > 1 else ""
        pattern = r"\.?</s>"
        pred = re.sub(pattern, "", pred)
        return pred
    
    correctness_res = {
        'EM': [],
        'F1': [],
        'Recall': [],
        'Precision': [],
    }
    os.makedirs(os.path.dirname(generations_output_file), exist_ok=True)
    with open(generations_output_file, 'w', encoding='utf-8') as g_file, open(generation_path_output_file, 'w', encoding='utf-8') as gp_file:
        for i in tqdm(range(len(dataset))):
            batch = dataset[i]
            pos_contexts = batch["positive_ctxs"]
            neg_contexts = batch["negative_ctxs"]
            cot, n_halluc, gen_path = model.inference(batch["question"], batch["qid"], fewshot_examplers, pos_contexts, neg_contexts)
            cot = cot.strip()
            
            final_ans = ""
            if "the answer is" not in cot:
                tmp = model.reinference(batch["question"], fewshot_examplers, cot).strip()  
                final_ans = get_answer(tmp) if "the answer is" in tmp else tmp
            else:
                final_ans = get_answer(cot)
            em_socre = correctness.exact_match_score(final_ans, batch["ground_truths"])
            f1_score = correctness.f1_score(final_ans, batch["ground_truths"])
                        
            g_item = {
                "qid": batch["qid"],
                "question": batch["question"],
                "em_score": em_socre['correct'],
                "f1_score": f1_score,
                "gold_answer": batch["ground_truths"],
                "pred": final_ans,
                "cot": cot,
                
            }
            g_file.write(json.dumps(g_item, ensure_ascii=False) + '\n')
            
            if args.rag_method in ['flare', 'dragin']:
                gp_item = {
                    "qid": batch["qid"],
                    "question": batch["question"],
                    "gold_answer": batch["ground_truths"],
                    "pred": final_ans,
                    "n_hallucination": n_halluc,
                    "generation_path": gen_path
                    
                }
                gp_file.write(json.dumps(gp_item, ensure_ascii=False) + '\n')
            
            correctness_res['EM'].append(em_socre['correct'])
            correctness_res['F1'].append(f1_score['f1'])
            correctness_res['Recall'].append(f1_score['recall'])
            correctness_res['Precision'].append(f1_score['precision'])


    # === Save results ==========================
    reuslts_dict = {
        'EM': np.mean(correctness_res['EM'])*100,
        'F1': np.mean(correctness_res['F1'])*100,
        'Recall': np.mean(correctness_res['Recall'])*100,
        'Precision': np.mean(correctness_res['Precision'])*100,
        'retrieve_count': model.counter.retrieve / len(dataset),
        'generate_count': model.counter.generate / len(dataset),
        'hallucinated_count': model.counter.hallucinated / len(dataset),
        'token_count': model.counter.token / len(dataset),
        'sentence_count': model.counter.sentence / len(dataset),
    }
    with open(results_output_file, 'w') as file:
        json.dump(reuslts_dict, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--dataset', type=str, default='wikimultihopqa', choices=[
        'wikimultihopqa', 'hotpotqa', 'musique', 'iirc', 'multihop_rag',
        'nqgold', 'trivia', 'popqa',
        'factscore'
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--rag_method', type=str, default='dragin', choices=[
        'no_retrieval', 'single_retrieval',
        'fix_length_retrieval', 'fix_sentence_retrieval',
        'flare', 'dragin'
    ])
    parser.add_argument('--retriever_model', type=str, default='bm25', choices=[
        'positive', 'negative', 'bm25', 'contriever', 'rerank', 'bge_m3', 'sgpt' # https://github.com/Muennighoff/sgpt
    ])
    parser.add_argument("--bm25_k1", type=float, default=0.9)
    parser.add_argument("--bm25_b", type=float, default=0.4)
    parser.add_argument('--use_counter', action='store_false')
    parser.add_argument('--fraction_of_data_to_use', type=float, default=0.06)
    parser.add_argument('--fewshot', type=int, default=6)
    parser.add_argument('--generate_max_length', type=int, default=64)
    parser.add_argument('--retrieve_topk', type=int, default=3)
    parser.add_argument('--retrieve_max_query_length', type=int, default=64)
    parser.add_argument('--generate_fix_length', type=int, default=25)
    
    parser.add_argument('--modifier_method', type=str, default='token', choices=['token', 'entity'])          # for FLARE
    parser.add_argument('--query_formulation', type=str, default='real_words', choices=[                      # for FLARE & DRAGIN
        'direct', 'forward_all',
        'real_words', 'current', 'current_wo_wrong', 'last_sentence', 'last_n_tokens',
    ])
    parser.add_argument('--sentence_solver', type=str, default='avg', choices=['avg', 'max', 'min'])          # for FLARE
    parser.add_argument('--hallucination_threshold', type=float, default=1.2)                                # for FLARE & DRAGIN
    parser.add_argument('--retrieve_keep_top_k', type=int, default=25)                                        # for DRAGIN
    parser.add_argument('--check_real_words', action='store_false')                                           # for DRAGIN
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--run', type=str, default='run_1 (300s-ct)')
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()
    
    ### === Define CUDA device =================== 
    args.output_dir = f"run_output/{args.run}" 
    args.device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. No GPUs detected.")
        
    
    ### === Run Steps ============================
    set_seed(args.seed)
    adaptive_generation(args)
    
    
    # python run/adaptive_generation.py

