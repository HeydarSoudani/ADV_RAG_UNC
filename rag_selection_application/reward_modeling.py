#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import math
import torch
import datasets
import argparse
import pandas as pd
import torch.nn as nn
from accelerate import Accelerator
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    DataCollatorWithPadding
)
from datasets import Dataset

from utils.general_utils import set_seed


## -----------------------------------------------
## Src: https://github.com/alirezasalemi7/PR-RAG
## ---
# Step 1) Create training data
# Step 2) Training
# Step 3) inference
## -----------------------------------------------

class RewardTrainer(Trainer):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.reward_loss_fn = nn.BCEWithLogitsLoss()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=1):
        pos_input_ids = inputs["pos_input_ids"]
        pos_attention_mask = inputs["pos_attention_mask"]
        neg_input_ids = inputs["neg_input_ids"]
        neg_attention_mask = inputs["neg_attention_mask"]

        pos_outputs = model(input_ids=pos_input_ids, attention_mask=pos_attention_mask)
        neg_outputs = model(input_ids=neg_input_ids, attention_mask=neg_attention_mask)
        pos_scores = pos_outputs.logits.squeeze(-1)
        neg_scores = neg_outputs.logits.squeeze(-1)

        score_diff = pos_scores - neg_scores
        labels = torch.ones(score_diff.size(), device=pos_scores.device)
        reward_loss = self.reward_loss_fn(score_diff, labels)
        labels.to(torch.device("cpu"))

        return (reward_loss, pos_outputs) if return_outputs else reward_loss

def create_processor(max_len):
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_len)
    return preprocess_function

def get_collator(max_length, tokenizer):
    def collator(batch):
        pos_outputs = [x['pos_output'] for x in batch]
        neg_outputs = [x['neg_output'] for x in batch]
        pos_encoded = tokenizer(
            pos_outputs,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        neg_encoded = tokenizer(
            neg_outputs,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        return {
            "pos_input_ids": pos_encoded["input_ids"],
            "pos_attention_mask": pos_encoded["attention_mask"],
            "neg_input_ids": neg_encoded["input_ids"],
            "neg_attention_mask": neg_encoded["attention_mask"]
        }
    return collator

def gen_proccessor(tokenizer, max_input_length):
    def process(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_input_length, padding="max_length")
    return process


### === Main Functions =========================== 

def training(args):
    # === MultiGPU setup =========================
    accelerator = Accelerator()
    device = accelerator.device
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=1,
        cache_dir=args.cache_dir,
        reference_compile=False
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    processor = create_processor(args.max_len_input)
    collator = get_collator(args.max_len_input, tokenizer)
    
    training_args = TrainingArguments(
        output_dir="models/rag_selector",      # REQUIRED
        do_train=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        weight_decay=0.0,
        num_train_epochs=10,
        lr_scheduler_type="linear",             # or SchedulerType.LINEAR
        warmup_ratio=0.05,
        save_strategy="epoch",                  # "steps" or "epoch"
        remove_unused_columns=False,            # you were setting this later; set it here
        logging_steps=50,                       # optional extras
        save_total_limit=2,                     # optional
        seed=42,                                # optional
    )
    
    # --- Create training dataset
    dataset_list = [] # {"pos_output", "neg_output"}
    if os.path.exists('rag_selection_application/datasets/training_preference_dataset.jsonl'):
        with open('rag_selection_application/datasets/training_preference_dataset.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line)
                dataset_list.append({
                    "pos_output": f'{sample["query"]}{tokenizer.sep_token}{sample['positive_sample'][0]}{tokenizer.sep_token}{sample['positive_sample'][2]}',
                    "neg_output": f'{sample["query"]}{tokenizer.sep_token}{sample['negative_sample'][0]}{tokenizer.sep_token}{sample['negative_sample'][2]}'
                })

    train_dataset = datasets.Dataset.from_list(dataset_list)
    
    trainer = RewardTrainer(
        model = model,
        args = training_args,
        data_collator = collator,
        train_dataset = train_dataset,
        tokenizer = tokenizer
    )
    trainer.train()

def inference(args):
    # === MultiGPU setup ============
    accelerator = Accelerator()
    device = accelerator.device
    
    candidates_df = create_inference_data(args)

    # --- Prepare input prompts -----
    def prepare_prompts(candidates_df, sep_token):
        inputs, qids, mids = [], [], []
        for idx, row in enumerate(candidates_df.itertuples(index=False)):
            # if idx == 50:
            #     break
            
            inputs.append({"text": f'{row.query} {sep_token} {row.pred_answer} {sep_token} {row.confidence}'})
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
        per_device_eval_batch_size=32,
        eval_accumulation_steps=1,
        do_predict=True,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        data_collator=collator,
        tokenizer=tokenizer
    )
    
    # --- Inference loop -------------
    all_grouped_scores, final_outputs = {}, {}
    scores = trainer.predict(inputs).predictions.squeeze().tolist()

    for qid, mid, score in zip(qids, mids, scores):
        if qid not in all_grouped_scores:
            all_grouped_scores[qid] = []
        all_grouped_scores[qid].append({"method": mid, "score": score})
    
    output_file_path = f"run_output/{args.run}/rag_selection/{args.dataset}_{args.subsec}/{args.retriever_name}/{args.consistency_method}_results.jsonl"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    with open(output_file_path, 'w') as res_f:
        for qid, scores in all_grouped_scores.items():
            all_grouped_scores[qid] = sorted(scores, key=lambda x: x["score"], reverse=True)
            
            selected_answer = candidates_df[
                (candidates_df["qid"] == qid) &
                (candidates_df["method"] == all_grouped_scores[qid][0]["method"])
            ]["pred_answer"].iloc[0]
            em = candidates_df[
                (candidates_df["qid"] == qid) &
                (candidates_df["method"] == all_grouped_scores[qid][0]["method"])
            ]["em"].iloc[0]
            
            item = {
                "qid": qid,
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
    parser.add_argument('--saved_model_name_or_path', type=str, default='models/rag_selector/checkpoint-800')
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
    parser.add_argument('--run', type=str, default='run_4 (rag_methods_500)')
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument('--use_counter', action='store_false')
    
    args = parser.parse_args()
    
    
    ### === Run Steps =============
    set_seed(args.seed)
    # create_training_data(args)
    # create_inference_data(args)
    # training(args)
    inference(args)
    
    
    # python rag_selection_application/reward_modeling.py
    # accelerate launch --multi_gpu rag_selection_application/reward_modeling.py
