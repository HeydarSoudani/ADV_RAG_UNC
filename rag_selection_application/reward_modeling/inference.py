#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import json
import torch
import argparse
import numpy as np
import pandas as pd
from datasets import Dataset
from accelerate import Accelerator
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

from utils.general_utils import set_seed

def make_serializable(obj):
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj

def gen_proccessor(tokenizer, max_input_length):
    def process(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_input_length, padding="max_length")
    return process

def jsonl_to_df(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    df = pd.DataFrame(data)
    return df

def inference(args):
    # === MultiGPU setup ============
    accelerator = Accelerator()
    device = accelerator.device
    
    inference_data_path = f"run_output/{args.run}/rag_selection_reward_modeling/{args.dataset}_{args.retriever_name}_{args.consistency_method}/{args.subsec}_inference_data.jsonl"
    candidates_df = jsonl_to_df(inference_data_path)

    output_file_path = f"run_output/{args.run}/rag_selection_reward_modeling/{args.dataset}_{args.retriever_name}_{args.consistency_method}/{args.subsec}_inference_results_{args.prompt_format}.jsonl"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    # === Prompt format
    if args.prompt_format == 'o_c':
        prompt_template = 'The answer is {answer}, with confidence score {sep_token} {conf_score}'
    elif args.prompt_format == 'p_o_c':
        prompt_template = '{path} {sep_token} the answer is {answer}, with confidence score {sep_token} {conf_score}'
    elif args.prompt_format == 'x_o_c':
        # prompt_template = '{query} {sep_token} the answer is {answer}, with confidence score {sep_token} {conf_score}'
        prompt_template = '{query} {sep_token} {answer} {sep_token} {conf_score}'
    elif args.prompt_format == 'x_o_mc':
        prompt_template = '{query} {sep_token} {answer} {sep_token} {rag_method} {conf_score}'
    elif args.prompt_format == 'x_p_o_c':
        prompt_template = '{query} {sep_token} {path} {sep_token} the answer is {answer}, with confidence score {sep_token} {conf_score}'
    elif args.prompt_format == 'x_g_o_c':
        prompt_template = '{query} {sep_token} {generations} {sep_token} {answer} {sep_token} {conf_score}'
    elif args.prompt_format == 'x_p_o':
        prompt_template = '{query} {sep_token} {path} {sep_token} the answer is {answer}'
    elif args.prompt_format == 'x_o':
        prompt_template = '{query} {sep_token} {answer}'

    # --- Prepare input prompts -----
    def prepare_prompts(candidates_df, sep_token):
        inputs, qids, mids = [], [], []
        for idx, row in enumerate(candidates_df.itertuples(index=False)):
            # if idx == 50:
            #     break
            
            inputs.append({
                "text": prompt_template.format(
                    query=row.query,
                    sep_token=sep_token,
                    answer=row.pred_answer,
                    conf_score=row.confidence,
                    rag_method=row.method,
                    path=' '.join(str(q) for q in row.search_queries if q),
                    generations=' '.join(str(g) for g in row.generations if g),
                )
            })
            qids.append(row.qid)
            mids.append(row.method)
        
        return Dataset.from_list(inputs), qids, mids
    
    # --- Load models ---------------
    tokenizer = AutoTokenizer.from_pretrained(args.saved_model_name_or_path, cache_dir=args.cache_dir)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(args.saved_model_name_or_path, num_labels=1, cache_dir=args.cache_dir, reference_compile=False).to(device)
    except:
        model = AutoModelForSequenceClassification.from_pretrained(args.saved_model_name_or_path, num_labels=1, cache_dir=args.cache_dir).to(device)
    
    inputs, qids, mids = prepare_prompts(candidates_df, tokenizer.sep_token)
    inputs = inputs.map(gen_proccessor(tokenizer, args.max_tokens), batched=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=args.max_tokens, pad_to_multiple_of=512)
    arguments = TrainingArguments(
        output_dir="temp",
        per_device_eval_batch_size=128,
        eval_accumulation_steps=1,
        do_predict=True,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        data_collator=collator,
        tokenizer=tokenizer
    )
    scores = trainer.predict(inputs).predictions.squeeze().tolist()
    
    # --- Inference loop -------------
    all_grouped_scores, final_outputs = {}, {}
    ordered_methods = ['self_ask', 'react', 'search_o1', 'research', 'search_r1']    

    for qid, mid, score in zip(qids, mids, scores):
        if qid not in all_grouped_scores:
            all_grouped_scores[qid] = []
        all_grouped_scores[qid].append({"method": mid, "score": score})
    
    with open(output_file_path, 'w') as res_f:
        for idx, (qid, scores) in enumerate(all_grouped_scores.items()):
            all_scores = [round(item['score'], 2) for item in all_grouped_scores[qid]]
            
            all_grouped_scores[qid] = sorted(scores, key=lambda x: x["score"], reverse=True)
            selected_answer = candidates_df[
                (candidates_df["qid"] == qid) &
                (candidates_df["method"] == all_grouped_scores[qid][0]["method"])
            ]["pred_answer"].iloc[0]
            em = candidates_df[
                (candidates_df["qid"] == qid) &
                (candidates_df["method"] == all_grouped_scores[qid][0]["method"])
            ]["em"].iloc[0]
            
            # 
            sub = candidates_df[candidates_df["qid"] == qid].set_index("method")
            sub = sub.reindex(ordered_methods)
            all_pred_answers = sub["pred_answer"].tolist()
            all_ems = sub["em"].tolist()
            all_confs = sub["confidence"].tolist()
            
            item = {
                "qid": qid,
                "all_scores": all_scores,
                "all_correctness": all_ems,
                "all_confidences": all_confs,
                "all_pred_answers": all_pred_answers,
                "selected_method": all_grouped_scores[qid][0]["method"],
                "selected_answer": selected_answer,
                "em": str(em),
            }
            res_f.write(json.dumps(item) + '\n')

    # print(all_grouped_scores)
    # print(final_outputs)
    # with open(args.output_addr, 'w') as f:
    #     json.dump(final_outputs, f, indent=4)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model_name_or_path', type=str, default='answerdotai/ModernBERT-base')
    parser.add_argument('--saved_model_name_or_path', type=str, default='models/rag_selection_reward_modeling/ModernBERT-large/x_o_c/checkpoint-830')
    parser.add_argument('--cache_dir', type=str, default='./cache/')
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--max_len_input', type=int, default=128)
    parser.add_argument("--max_tokens", type=int, default=4096)

    # Dataset
    parser.add_argument('--dataset', type=str, default='hotpotqa', choices=[
        'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'
    ])
    parser.add_argument('--subsec', type=str, default='dev', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
    parser.add_argument("--enable_fewshot_examples", action="store_true", help="")
    parser.add_argument('--prompt_format', type=str, default='x_o_c', choices=['x_o_c', 'o_c', 'x_g_o_c', 'x_p_o_c', 'p_o_c', 'x_p_o', 'x_o_mc'])
    
    # Retriever
    parser.add_argument('--retriever_name', type=str, default='rerank_l6', choices=[
        'bm25', 'contriever', 'rerank_l6', 'rerank_l12', 'e5', 'bge', 'reasonir'
    ])
    parser.add_argument('--corpus_path', type=str, default='data/search_r1_files/wiki-18.jsonl')
    parser.add_argument('--index_path', type=str, default='data/search_r1_files/bm25', choices=[
        'data/search_r1_files/bm25',          # For BM25 & Rerank
        'data/search_r1_files/e5_Flat.index', # For E5
        'data/search_r1_files/reasonir_Flat.index', # For ReasonIR
    ])
    parser.add_argument("--retrieval_model_path", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2", choices=[
        "cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-MiniLM-L12-v2", # For Rerank
        "intfloat/e5-base-v2",  # For E5
        "reasonir/ReasonIR-8B", # For ReasonIR
    ])
    parser.add_argument('--retrieval_topk', type=int, default=3)
    parser.add_argument('--faiss_gpu', action='store_false', help='Use GPU for computation')
    parser.add_argument('--retrieval_pooling_method', type=str, default="mean")
    parser.add_argument('--retrieval_query_max_length', type=int, default=256)
    parser.add_argument('--retrieval_use_fp16', action='store_false', help='')
    parser.add_argument('--retrieval_batch_size', type=int, default=512)
    parser.add_argument("--bm25_k1", type=float, default=0.9)
    parser.add_argument("--bm25_b", type=float, default=0.4)
    
    # Consistency Generation Methods (answer list) ---
    parser.add_argument('--consistency_method', type=str, default='rag_consistency', choices=[
        'self_consistency', 'reasoning_consistency', 'rag_consistency'
    ])
    parser.add_argument("--n_generations", type=int, default=10)
    parser.add_argument("--mask_left_boundary", type=float, default=0.1)
    parser.add_argument("--mask_right_boundary", type=float, default=0.4)
    parser.add_argument("--consistency_temperature", type=float, default=1.0)
    
    # Others
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--run', type=str, default='run_3 (rag_methods_500)')
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument('--use_counter', action='store_false')
    
    args = parser.parse_args()
    
    
    ### === Run Steps =============
    set_seed(args.seed)
    inference(args)
    
    
    # python rag_selection_application/reward_modeling/inference.py
    # accelerate launch --multi_gpu rag_selection_application/reward_modeling/inference.py

