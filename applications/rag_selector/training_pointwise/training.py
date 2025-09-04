#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import json
import torch
import random
import datasets
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import collections
from dataclasses import dataclass
from accelerate import Accelerator
from typing import Dict, Any, Optional
import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorWithPadding
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

class ScoreEvalCollator:
    def __init__(self, tokenizer):
        self.pad = DataCollatorWithPadding(tokenizer, padding=True, return_tensors="pt")
    def __call__(self, batch):
        # keep only tokenized fields for padding
        features = [{k: ex[k] for k in ("input_ids","attention_mask","token_type_ids") if k in ex} for ex in batch]
        out = self.pad(features)
        # labels as float tensor
        out["labels"] = torch.tensor([float(ex["labels"]) for ex in batch], dtype=torch.float32)
        return out

# 2) Smart wrapper that chooses which collator to use
class SmartCollator:
    def __init__(self, train_collator, eval_collator):
        self.train_collator = train_collator
        self.eval_collator = eval_collator
    def __call__(self, batch):
        b0 = batch[0]
        # heuristics: fields typical of preference training
        if ("pos_output" in b0 or "pos_input_ids" in b0 or ("chosen" in b0 and "rejected" in b0)):
            return self.train_collator(batch)
        return self.eval_collator(batch)


def gen_proccessor(tokenizer, max_input_length):
    def process(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_input_length, padding="max_length")
    return process

def create_processor(max_len):
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_len)
    return preprocess_function

def sample_by_qid_ds(eval_ds, frac=0.2, seed=42):
    qids = np.array(eval_ds["qid"])
    uniq_qids = np.unique(qids)
    n_keep = max(1, int(np.ceil(len(uniq_qids) * frac)))
    rng = np.random.default_rng(seed)
    keep_qids = set(rng.choice(uniq_qids, size=n_keep, replace=False))
    return eval_ds.filter(lambda ex: ex["qid"] in keep_qids)

def make_grouped_top1_compute(eval_ds):
    qids = np.array(eval_ds["qid"])
    gold = np.array(eval_ds["labels"], dtype=float)
    methods = np.array(eval_ds["method"])

    # Precompute indices per qid
    group_ix = collections.defaultdict(list)
    for i, q in enumerate(qids):
        group_ix[q].append(i)

    def compute_metrics(eval_pred):
        preds = np.asarray(eval_pred.predictions).reshape(-1)
        # Safety: ensure alignment
        if preds.shape[0] != len(qids):
            # This should not happen; return 0 to avoid crashing training
            return {"grouped_top1": 0.0}

        correct = 0
        total = 0
        for q, idxs in group_ix.items():
            p = preds[idxs]
            g = gold[idxs]

            # allow ties: any overlap counts as correct
            pred_best = np.flatnonzero(p == p.max())
            gold_best = np.flatnonzero(g == g.max())
            if np.intersect1d(pred_best, gold_best).size > 0:
                correct += 1
            total += 1

        top1 = correct / total if total else 0.0
        return {"grouped_top1": float(top1)}

    return compute_metrics

# class RewardTrainer(Trainer):
#     def __init__(self, model, *args, **kwargs):
#         super().__init__(model, *args, **kwargs)
#         self.reward_loss_fn = nn.BCEWithLogitsLoss()

#     def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=1):
#         pos_input_ids = inputs["pos_input_ids"]
#         pos_attention_mask = inputs["pos_attention_mask"]
#         neg_input_ids = inputs["neg_input_ids"]
#         neg_attention_mask = inputs["neg_attention_mask"]

#         pos_outputs = model(input_ids=pos_input_ids, attention_mask=pos_attention_mask)
#         neg_outputs = model(input_ids=neg_input_ids, attention_mask=neg_attention_mask)
#         pos_scores = pos_outputs.logits.squeeze(-1)
#         neg_scores = neg_outputs.logits.squeeze(-1)

#         score_diff = pos_scores - neg_scores
#         labels = torch.ones(score_diff.size(), device=pos_scores.device)
#         reward_loss = self.reward_loss_fn(score_diff, labels)
#         labels.to(torch.device("cpu"))

#         return (reward_loss, pos_outputs) if return_outputs else reward_loss

class RewardTrainer(Trainer):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.reward_loss_fn = nn.BCEWithLogitsLoss()
        self._mse = nn.MSELoss()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=1):
        # -------- Pairwise path (training) --------
        if "pos_input_ids" in inputs and "neg_input_ids" in inputs:
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

        # -------- Scoring path (eval) --------
        # single input (optionally with float labels)
        outputs = model(input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"))
        logits = outputs.logits.squeeze(-1)

        labels = inputs.get("labels", None)
        if labels is not None:
            labels = labels.to(logits.device).float().view_as(logits)
            loss = self._mse(logits, labels)   # only for logging; your metrics use labels anyway
        else:
            loss = torch.zeros((), device=logits.device)

        return (loss, outputs) if return_outputs else loss


def compute_scoring_metrics(
    qids: np.ndarray,
    methods: np.ndarray,
    preds: np.ndarray,
    em: Optional[np.ndarray] = None,
    gold: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    - preds: model predicted scores (higher = better).
    - em: optional 0/1 label per (qid, method) -> enables top-1 EM@qid.
    - gold: optional continuous target per (qid, method) -> enables per-qid Spearman.
    Returns a flat dict of metrics.
    """
    metrics: Dict[str, float] = {}

    # Top-1 EM accuracy: pick predicted-best method per qid, check its EM==1
    if em is not None:
        df = pd.DataFrame({"qid": qids, "method": methods, "pred": preds, "em": em})
        top1 = (
            df.sort_values(["qid", "pred"], ascending=[True, False])
              .groupby("qid", as_index=False).first()
        )
        metrics["test_top1_em_acc"] = float((top1["em"] == 1).mean())

    # Per-qid Spearman correlation between preds and gold (if provided)
    if gold is not None:
        try:
            from scipy.stats import spearmanr
            have_scipy = True
        except Exception:
            have_scipy = False

        df = pd.DataFrame({"qid": qids, "pred": preds, "gold": gold})
        rhos = []
        for _, g in df.groupby("qid"):
            if g["gold"].nunique() <= 1:
                continue  # undefined when gold is constant
            if have_scipy:
                rho = spearmanr(g["pred"], g["gold"]).correlation
            else:
                rho = g[["pred", "gold"]].corr(method="spearman").iloc[0, 1]
            if pd.notna(rho):
                rhos.append(rho)
        if rhos:
            metrics["test_spearman_qid_mean"] = float(np.mean(rhos))
            metrics["test_spearman_qid_median"] = float(np.median(rhos))

    # Simple global correlation (optional)
    if gold is not None:
        if have_scipy:
            from scipy.stats import spearmanr
            rho = spearmanr(preds, gold).correlation
        else:
            rho = pd.Series(preds).corr(pd.Series(gold), method="spearman")
        if pd.notna(rho):
            metrics["test_spearman_global"] = float(rho)

    # Count
    metrics["test_examples"] = int(len(preds))
    metrics["test_qids"] = int(len(np.unique(qids)))
    return metrics

@dataclass
class ScoringEvalCallback(TrainerCallback):
    eval_dataset: Any                   # HF datasets.Dataset for scoring
    tokenizer: Any
    eval_data_collator: Any
    eval_batch_size: int = 128

    def on_epoch_end(self, args, state, control, **kwargs):
        trainer = kwargs["trainer"]
        model = trainer.model

        # Build a minimal eval Trainer that uses your scoring collator (single-sample),
        # leaving the RewardTrainer's pairwise collator untouched.
        eval_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_eval_batch_size=self.eval_batch_size,
            do_predict=True,
            report_to=args.report_to,     # keep your loggers (wandb/tensorboard) consistent
            logging_dir=args.logging_dir if hasattr(args, "logging_dir") else None,
            disable_tqdm=True,
        )
        eval_trainer = Trainer(
            model=model,
            args=eval_args,
            data_collator=self.eval_data_collator,
            tokenizer=self.tokenizer,
        )

        preds = eval_trainer.predict(self.eval_dataset).predictions
        preds = np.squeeze(preds)

        qids = np.asarray(self.eval_dataset["qid"])
        mids = np.asarray(self.eval_dataset["method"])
        em = np.asarray(self.eval_dataset["em"]) if "em" in self.eval_dataset.column_names else None
        gold = np.asarray(self.eval_dataset["gold"]) if "gold" in self.eval_dataset.column_names else None

        metrics = compute_scoring_metrics(qids=qids, methods=mids, preds=preds, em=em, gold=gold)

        # Log into the main trainer (shows up in progress bar + callbacks, wandb, etc.)
        trainer.log(metrics)
        # Keep going as usual
        return control


def training(args):
    # === MultiGPU setup =========================
    accelerator = Accelerator()
    device = accelerator.device
    
    # === Model output path
    model_ = args.model_name_or_path.split('/')[-1]
    model_output_dir = f'models/rag_selection_reward_modeling/{model_}/{args.prompt_format}'
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=1,
        cache_dir=args.cache_dir,
        reference_compile=False
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    processor = create_processor(args.max_len_input)
    collator = get_collator(args.max_len_input, tokenizer)
    
    train_collator = collator  # your existing preference collator
    eval_collator  = ScoreEvalCollator(tokenizer)
    smart_collator = SmartCollator(train_collator, eval_collator)


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
    
    # === training & test dataset
    # --- Train setup -------------
    training_data_path = f"run_output/{args.run}/rag_selection_reward_modeling/{args.dataset}_{args.retriever_name}_{args.consistency_method}/train_preference_data.jsonl"
    train_dataset_list = [] # {"pos_output", "neg_output"}
    if os.path.exists(training_data_path):
        with open(training_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line)
                train_dataset_list.append({
                    "pos_output": prompt_template.format(
                        sep_token=tokenizer.sep_token,
                        # V1
                        query=sample["query"],
                        answer=sample['positive_sample'][0],
                        rag_method=sample['positive_sample'][2],
                        conf_score=sample['positive_sample'][3],
                        generations=' '.join(str(q) for q in sample['positive_sample'][4] if q),
                        path=' '.join(str(q) for q in sample['positive_sample'][5] if q),
                        # V2
                        # query=sample["positive_sample"][1],
                        # answer=sample['positive_sample'][2],
                        # rag_method=sample['positive_sample'][4],
                        # conf_score=sample['positive_sample'][5],
                    ),
                    "neg_output": prompt_template.format(
                        sep_token=tokenizer.sep_token,
                        # V1
                        query=sample["query"],
                        answer=sample['negative_sample'][0],
                        rag_method=sample['negative_sample'][2],
                        conf_score=sample['negative_sample'][3],
                        generations=' '.join(str(q) for q in sample['negative_sample'][4] if q),
                        path=' '.join(str(q) for q in sample['negative_sample'][5] if q),
                        # V2
                        # query=sample["negative_sample"][1],
                        # answer=sample['negative_sample'][2],
                        # rag_method=sample['negative_sample'][4],
                        # conf_score=sample['negative_sample'][5],
                    ),
                })
    train_dataset = datasets.Dataset.from_list(train_dataset_list)
    
    # --- Eval setup -------------
    eval_data_path = f"run_output/{args.run}/rag_selection_reward_modeling/{args.dataset}_{args.retriever_name}_{args.consistency_method}/{args.subsec}_inference_data.jsonl"
    eval_dataset_list = []
    if os.path.exists(eval_data_path):
        with open(eval_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line)
                eval_dataset_list.append({
                    "text": prompt_template.format(
                        query=sample["query"],
                        answer=sample["pred_answer"],
                        rag_method=sample["method"],
                        conf_score=sample["confidence"],
                        sep_token=tokenizer.sep_token,
                    ),
                    "qid": sample["qid"],
                    "labels": sample["em"],
                    "method": sample["method"],
                })
    
    eval_dataset = datasets.Dataset.from_list(eval_dataset_list)
    eval_dataset = eval_dataset.map(gen_proccessor(tokenizer, args.max_tokens), batched=True)
    eval_dataset = sample_by_qid_ds(eval_dataset, frac=0.0, seed=42)
    
    compute_metrics = make_grouped_top1_compute(eval_dataset)
    
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        do_train=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        weight_decay=0.0,
        num_train_epochs=5,
        greater_is_better=True,
        # metric_for_best_model="grouped_top1",
        # evaluation_strategy="epoch",
        # load_best_model_at_end=True,          # optional
        lr_scheduler_type="linear",             # or SchedulerType.LINEAR
        warmup_ratio=0.05,
        save_strategy="epoch",                  # "steps" or "epoch"
        remove_unused_columns=False,            # you were setting this later; set it here
        logging_steps=100,                      # optional extras
        save_total_limit=2,                     # optional
        seed=args.seed,                         # optional
    )
    
    reward_trainer = RewardTrainer(
        model = model,
        args = training_args,
        data_collator = smart_collator,
        train_dataset = train_dataset,
        tokenizer = tokenizer,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    
    class EvalEachEpochCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            control.should_evaluate = True
            return control
    
    reward_trainer.add_callback(EvalEachEpochCallback())
    
    # reward_trainer.train(resume_from_checkpoint=True)
    reward_trainer.train()


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
    parser.add_argument('--prompt_format', type=str, default='x_o', choices=['x_o', 'x_o_c', 'o_c', 'x_g_o_c', 'x_p_o_c', 'p_o_c', 'x_p_o', 'x_o_mc'])
    
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

