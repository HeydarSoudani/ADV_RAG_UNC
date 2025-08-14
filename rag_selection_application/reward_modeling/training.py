#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import json
import torch
import random
import datasets
import argparse
import torch.nn as nn
from accelerate import Accelerator
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from utils.general_utils import set_seed


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

def create_processor(max_len):
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_len)
    return preprocess_function

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


def training(args):
    # === MultiGPU setup =========================
    accelerator = Accelerator()
    device = accelerator.device
    
    # === Model output path
    model_ = args.model_name_or_path.split('/')[-1]
    model_output_dir = f'models/rag_selection_reward_modeling/{model_}/{args.prompt_format}'
    
    # === Prompt format
    if args.prompt_format == 'o_c':
        prompt_template = 'The answer is {answer}, with confidence score {sep_token} {conf_score}'
    elif args.prompt_format == 'x_o':
        prompt_template = '{query} {sep_token} {answer}'
    elif args.prompt_format == 'p_o_c':
        prompt_template = '{path} {sep_token} the answer is {answer}, with confidence score {sep_token} {conf_score}'
    elif args.prompt_format == 'x_o_c':
        # prompt_template = '{query} {sep_token} the answer is {answer}, with confidence score {sep_token} {conf_score}'
        prompt_template = '{query} {sep_token} {answer} {sep_token} {conf_score}'
    elif args.prompt_format == 'x_o_mc':
        prompt_template = '{query} {sep_token} {answer} {sep_token} {rag_method} {conf_score}'
    elif args.prompt_format == 'x_p_o_c':
        # prompt_template = '{query} {sep_token} {path} {sep_token} the answer is {answer}, with confidence score {sep_token} {conf_score}'
        prompt_template = '{query} {sep_token} {path} {sep_token} {answer} {sep_token} {conf_score}'
    elif args.prompt_format == 'x_g_o_c':
        prompt_template = '{query} {sep_token} {generations} {sep_token} {answer} {sep_token} {conf_score}'
    elif args.prompt_format == 'x_p_o':
        prompt_template = '{query} {sep_token} {path} {sep_token} the answer is {answer}'
    
    # === Model 
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
        output_dir=model_output_dir,
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
        logging_steps=100,                       # optional extras
        save_total_limit=2,                     # optional
        seed=args.seed,                                # optional
    )
    
    # --- Prapering training dataset
    MASK_PROB = 0.7          # 70% of pairs will have conf masked
    MASK_TOKEN = "[NO_CONF]" # or "" if you prefer empty
    rng = random.Random(args.seed)  # set seed for reproducibility
    
    training_data_path = f"run_output/{args.run}/rag_selection_reward_modeling/{args.dataset}_{args.retriever_name}_{args.consistency_method}/train_preference_data.jsonl"
    train_dataset_list = [] # {"pos_output", "neg_output"}
    if os.path.exists(training_data_path):
        with open(training_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line)
                
                conf_visible = rng.random() > MASK_PROB

                train_dataset_list.append({
                    "pos_output": prompt_template.format(
                        query=sample["query"],
                        sep_token=tokenizer.sep_token,
                        answer=sample['positive_sample'][0],
                        conf_score=sample['positive_sample'][3], # if conf_visible else MASK_TOKEN
                        rag_method=sample['positive_sample'][2],
                        generations=' '.join(str(q) for q in sample['positive_sample'][4] if q),
                        path=' '.join(str(q) for q in sample['positive_sample'][5] if q)
                    ),
                    "neg_output": prompt_template.format(
                        query=sample["query"],
                        sep_token=tokenizer.sep_token,
                        answer=sample['negative_sample'][0],
                        conf_score=sample['negative_sample'][3], # if conf_visible else MASK_TOKEN
                        rag_method=sample['negative_sample'][2],
                        generations=' '.join(str(q) for q in sample['negative_sample'][4] if q),
                        path=' '.join(str(q) for q in sample['negative_sample'][5] if q)
                    ),
                })

    train_dataset = datasets.Dataset.from_list(train_dataset_list)
    
    trainer = RewardTrainer(
        model = model,
        args = training_args,
        data_collator = collator,
        train_dataset = train_dataset,
        tokenizer = tokenizer
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model_name_or_path', type=str, default='answerdotai/ModernBERT-large')
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
    parser.add_argument('--prompt_format', type=str, default='x_o_c', choices=['x_o', 'x_o_c', 'o_c', 'x_g_o_c', 'x_p_o_c', 'p_o_c', 'x_p_o', 'x_o_mc'])
    
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
    training(args)
    
    # python rag_selection_application/reward_modeling/training.py
    # accelerate launch --multi_gpu rag_selection_application/reward_modeling/training.py

