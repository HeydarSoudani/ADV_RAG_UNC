#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import ast
import json
import torch
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from typing import Optional, Union, Dict, Any, Tuple
from scipy.stats import wilcoxon

import evaluate
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from utils.general_utils import set_seed


def str_to_tuple(s):
    try:
        return eval(s) if s != "None" else None
    except:
        return s

def data_creation(args):
    train_run = 'run_1 (rag_methods_2k)'
    dataset_subsec_train = 'train'
    test_run = 'run_3 (rag_methods_500)'
    dataset_subsec_test = 'dev'
    
    rag_methods = [
        ('Qwen2.5-7B-Instruct', 'self_ask'),
        ('Qwen2.5-7B-Instruct', 'react'),
        ('Qwen2.5-7B-Instruct', 'search_o1'),
        ('ReSearch-Qwen-7B-Instruct', 'research'),
        ('SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo', 'search_r1')
    ]
    
    # === train set ====================
    dfs = []
    for rag_method in rag_methods:
        file_path = f"run_output/{train_run}/{rag_method[0]}/{args.dataset}_{dataset_subsec_train}/{rag_method[1]}_{args.retriever_name}/{args.consistency_method}_results.jsonl"    
        with open(file_path, "r") as f:
            data = [json.loads(line) for line in f]
        df_temp = pd.DataFrame(data)[["qid", "query", "pred_answer", "em", "ue_scores"]]
        
        if args.consistency_method == 'rag_consistency':
            confidences = df_temp["ue_scores"].apply(lambda x: x["majority_voting"]["confidence"])
        else:
            confidences = df_temp["ue_scores"].apply(lambda x: x["majority_voting"]["most_confident_answer"][1])
        
        df_temp[rag_method[1]] = list(zip(df_temp["pred_answer"], df_temp["em"], confidences))
        df = df_temp[["qid", "query", rag_method[1]]]
        dfs.append(df)
    
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(df, on=["qid", "query"], how="outer")
    
    merged_df = merged_df.sort_values(by="qid", key=lambda x: x.str.extract(r"(\d+)").squeeze().astype(int)).reset_index(drop=True)
    merged_df_str = merged_df.astype(str)
    train_ds = Dataset.from_pandas(merged_df_str)
    
    
    # === test set ====================
    dfs = []
    for rag_method in rag_methods:
        file_path = f"run_output/{test_run}/{rag_method[0]}/{args.dataset}_{dataset_subsec_test}/{rag_method[1]}_{args.retriever_name}/{args.consistency_method}_results.jsonl"    
        with open(file_path, "r") as f:
            data = [json.loads(line) for line in f]
        df_temp = pd.DataFrame(data)[["qid", "query", "pred_answer", "em", "ue_scores"]]
        
        if args.consistency_method == 'rag_consistency':
            confidences = df_temp["ue_scores"].apply(lambda x: x["majority_voting"]["confidence"])
        else:
            confidences = df_temp["ue_scores"].apply(lambda x: x["majority_voting"]["most_confident_answer"][1])
        
        df_temp[rag_method[1]] = list(zip(df_temp["pred_answer"], df_temp["em"], confidences))
        df = df_temp[["qid", "query", rag_method[1]]]
        dfs.append(df)
    
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(df, on=["qid", "query"], how="outer")
    
    merged_df = merged_df.sort_values(by="qid", key=lambda x: x.str.extract(r"(\d+)").squeeze().astype(int)).reset_index(drop=True)
    merged_df_str = merged_df.astype(str)
    test_ds = Dataset.from_pandas(merged_df_str)
    
    dataset_dict = DatasetDict({"train": train_ds, "test": test_ds})
    print(dataset_dict)
    
    return dataset_dict
    # df_tuple = df_str.applymap(str_to_tuple)

def training(args):
    # === Load dataset ==========
    dataset = data_creation(args)
    classes = [title for title, value in dataset['train'].features.items() if title not in ['qid', 'query']]
    class2id = {class_:id for id, class_ in enumerate(classes)}
    id2class = {id:class_ for class_, id in class2id.items()}
    
    # === Load model ============
    model_path = 'answerdotai/ModernBERT-large'
    # model_path = 'microsoft/deberta-v3-base'
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=len(classes),
        id2label=id2class, label2id=class2id,
        problem_type = "multi_label_classification"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # === Functions =============
    class RewardTrainer(Trainer):
        def __init__(self, model, *args, **kwargs):
            super().__init__(model, *args, **kwargs)
        
        def compute_loss(self, model, inputs: Dict[str, Any], return_outputs: bool = False, num_items_in_batch=1):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits", outputs[0])
            labels = labels.to(dtype=logits.dtype)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            return (loss, outputs) if return_outputs else loss
    
    def preprocess_function(example):
        labels = []
        text = f'{example['query']}{tokenizer.sep_token}'
        for method in classes:
            method_item_tuple = ast.literal_eval(example[method])
            text += f'{method_item_tuple[0]}, {method_item_tuple[2]}{tokenizer.sep_token}'
            labels.append(method_item_tuple[1])
        
        example = tokenizer(text, truncation=True)
        example['labels'] = labels
        return example
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        top1_idx = np.argmax(logits, axis=1)
        correct = [labels[i, j] == 1 for i, j in enumerate(top1_idx)]
        correct = np.array(correct, dtype=int)  # 1 if correct, 0 if not
        accuracy = correct.mean()
        return {
            "top1_accuracy": accuracy,
            "correct_count": int(correct.sum())
        }
    
    # === Training ... ==========
    model_ = model_path.split('/')[-1]
    model_output_dir = f'models/rag_selection_reward_modeling/mlc/{model_}'
    
    tokenized_dataset = dataset.map(preprocess_function)
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        do_train=True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=10,
        weight_decay=0.01,
        eval_strategy="epoch",   # or "steps"
        save_strategy="epoch",
        # load_best_model_at_end=True,
        logging_steps=100,
        save_total_limit=2,
        seed=args.seed,
        load_best_model_at_end=True,
        # metric_for_best_model="micro_f1",
        greater_is_better=True,
    )
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()

def inference(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        'fa_consistency', 'rrr_consistency', 'reasoning_consistency', 'self_consistency', 'rag_consistency'
    ])
    
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
    # inference(args)
    # rubostness_analysis(args)
    
    # python applications/rag_selector/multi_label_classification/mlc_run.py
    # accelerate launch --multi_gpu applications/rag_selector/multi_label_classification/mlc_run.py
    