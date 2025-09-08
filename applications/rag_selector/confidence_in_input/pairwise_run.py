#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import re
import ast
import math
import json
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import unicodedata as ud
import torch.nn.functional as F
from dataclasses import dataclass
from itertools import combinations
from typing import List, Dict, Any, Optional
from datasets import Dataset, DatasetDict
import transformers
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from utils.general_utils import set_seed
from run_mcts_two_actions.src.models.semantic_equivalence import SemanticEquivalenceGenerator

# -- --------------------------------------
def clean_text(text: str) -> str:
    if not text:
        return text
    
    # Characters to remove
    BIDI = {0x202A,0x202B,0x202C,0x202D,0x202E,0x2066,0x2067,0x2068,0x2069}
    ZERO_WIDTH = {0x200B,0x200C,0x200D,0xFEFF}
    SOFT_HYPHEN = {0x00AD}
    PUNCT_MAP = {"“":"\"", "”":"\"", "‘":"'", "’":"'", "—":"-", "–":"-", "…":"..."}
    
    def _is_allowed_char(ch: str) -> bool:
        if ch in "\n\t":  # keep newlines/tabs
            return True
        c = ord(ch)
        if c in BIDI or c in ZERO_WIDTH or c in SOFT_HYPHEN:
            return False
        cat = ud.category(ch)
        if cat in ("Cc","Cf"):  # control/format chars
            return False
        return True
    
    # 1) Normalize Unicode
    s = ud.normalize("NFKC", text)
    # 2) Standardize line endings
    s = re.sub(r"\r\n?","\n", s)
    # 3) Simplify punctuation
    s = s.translate(str.maketrans(PUNCT_MAP))
    # 4) Remove disallowed characters
    s = "".join(ch for ch in s if _is_allowed_char(ch))
    # 5) Collapse odd whitespace
    s = re.sub(r"[ \t\f\v\u00A0\u2000-\u200A\u202F\u205F\u3000]+", " ", s)
    s = re.sub(r"[ \t]*\n[ \t]*", "\n", s)  # trim around newlines
    s = re.sub(r"\n{3,}", "\n\n", s)  # limit blank lines
    # 6) Strip leading/trailing spaces
    return s.strip()

def pack(s):
    return (s["prediction"], s["confidence_score"], s["method"], s["correctness"], s['search_queries'], s['generations'])

def merge_rag_systems_data(args, subsec='train'):
    if subsec == 'train':
        run = 'run_1 (rag_methods_2k)'
        dataset_subsec = 'train'
        correctness_m = 'llm_as_judge'
    else:
        run = 'run_3 (rag_methods_500)'
        dataset_subsec = 'dev'
        correctness_m = 'em'
    
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
        file_path = f"run_output/{run}/{rag_method[0]}/{args.dataset}_{dataset_subsec}/{rag_method[1]}_{args.retriever_name}/{args.consistency_method}_results.jsonl"    
        with open(file_path, "r") as f:
            data = [json.loads(line) for line in f]
        df_temp = pd.DataFrame(data)[["qid", "query", "pred_answer", correctness_m, "final_answer_list", "ue_scores"]]
        
        if args.consistency_method == 'rag_consistency':
            confidences = df_temp["ue_scores"].apply(lambda x: x["majority_voting"]["confidence"])
        else:
            confidences = df_temp["ue_scores"].apply(lambda x: x["majority_voting"]["most_confident_answer"][1])
        
        # - Add path
        search_queries_map = {}
        path_file = f"run_output/{run}/{rag_method[0]}/{args.dataset}_{dataset_subsec}/{rag_method[1]}_{args.retriever_name}/inference_results.jsonl"
        if os.path.exists(path_file):
            with open(path_file, "r") as f:
                corr_data = [json.loads(line) for line in f]
            for item in corr_data:
                if "path" in item and isinstance(item["path"], list):
                    sqs = [d.get("search_query") for d in item["path"] if isinstance(d, dict) and "search_query" in d]
                    search_queries_map[item["qid"]] = sqs
        
        
        # --- Add tuple including search queries
        df_temp[rag_method[1]] = [
            (pred, em, conf, gens, search_queries_map.get(qid, []))
            for pred, em, conf, qid, gens in zip(
                df_temp["pred_answer"], df_temp[correctness_m], confidences, df_temp["qid"], df_temp["final_answer_list"]
            )
        ]
        df = df_temp[["qid", "query", rag_method[1]]]
        dfs.append(df)
    
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(df, on=["qid", "query"], how="outer")
    merged_df = merged_df.sort_values(by="qid", key=lambda x: x.str.extract(r"(\d+)").squeeze().astype(int)).reset_index(drop=True)
    
    return merged_df

def get_prompt_template(prompt_format):
    if prompt_format == 'o_c':
        prompt_template = 'The answer is {answer}, with confidence score {sep_token} {conf_score}'
    elif prompt_format == 'x_o':
        prompt_template = '{query} {sep_token} {answer}'
    elif prompt_format == 'p_o_c':
        prompt_template = '{search_queries} {sep_token} the answer is {answer}, with confidence score {sep_token} {conf_score}'
    elif prompt_format == 'x_o_c':
        prompt_template = '{query} {sep_token} {answer} {sep_token} {conf_score}'
    elif prompt_format == 'x_o_mc':
        prompt_template = '{query} {sep_token} {answer} {sep_token} {rag_method} {conf_score}'
    elif prompt_format == 'x_p_o_c':
        prompt_template = '{query} {sep_token} {search_queries} {sep_token} {answer} {sep_token} {conf_score}'
    elif prompt_format == 'x_g_o_c':
        prompt_template = '{query} {sep_token} {generations} {sep_token} {answer} {sep_token} {conf_score}'
    elif prompt_format == 'x_p_o':
        prompt_template = '{query} {sep_token} {search_queries} {sep_token} the answer is {answer}'
    elif prompt_format == 'x_o_p_g':
        prompt_template = '{query} {sep_token} {answer} {sep_token} {search_queries} {sep_token} {generations}'

    return prompt_template


### ==== Main Functions =================== 
def add_correctness(args):
    rag_methods = [
        ('Qwen2.5-7B-Instruct', 'self_ask'),
        ('Qwen2.5-7B-Instruct', 'react'),
        ('Qwen2.5-7B-Instruct', 'search_o1'),
        ('ReSearch-Qwen-7B-Instruct', 'research'),
        ('SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo', 'search_r1')
    ]
    model = transformers.AutoModelForCausalLM.from_pretrained(args.secondary_model_name_or_path, dtype=torch.bfloat16).to(args.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.secondary_model_name_or_path)
    se_model = SemanticEquivalenceGenerator(args, args.device, model, tokenizer)
    
    for rag_method in rag_methods:
        file_path = f"run_output/{args.run}/{rag_method[0]}/{args.dataset}_{args.subsec}/{rag_method[1]}_{args.retriever_name}/{args.consistency_method}_results.jsonl"
        new_result_file_path = f"run_output/{args.run}/{rag_method[0]}/{args.dataset}_{args.subsec}/{rag_method[1]}_{args.retriever_name}/{args.consistency_method}_results_new.jsonl"
        
        with open(file_path, "r") as fin, open(new_result_file_path, "w") as fout:
            for idx, line in enumerate(tqdm(fin)):

                sample = json.loads(line)
                question, gt_answers, prediction = sample['query'], sample['gt_answers'], sample['pred_answer']
                llm_as_judge = int(any([se_model.check_answers_equiv(question, ga, prediction) for ga in gt_answers]))
                
                item = {
                    **sample,
                    'llm_as_judge': llm_as_judge
                }
                fout.write(json.dumps(item) + "\n")

def data_creation(args):
    train_df = merge_rag_systems_data(args, subsec='train')
    test_df = merge_rag_systems_data(args, subsec='test')

    # --- Train set: create perefrence pairs
    W_EM, W_CONF, MIN_GAP = 0.5, 0.5, 0.4
    rag_methods = [c for c in train_df.columns if c not in ("qid", "query")]
    records = []
    for _, row in train_df.iterrows():
        qid, query, samples = row["qid"], row["query"], []
        for col in rag_methods:
            val = row.get(col, None)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                continue
            try:
                if len(val) == 5:
                    pred, correctness, conf, gens, sqs = val
                elif len(val) == 3:
                    pred, correctness, conf = val
                    sqs, gens = None, None
                else:
                    continue
            except Exception:
                continue
            
            if conf is None or (isinstance(conf, float) and math.isnan(conf)):
                continue
                        
            score = W_EM * float(correctness) + W_CONF * float(conf)
            samples.append({
                "method": col, "prediction": pred,
                "correctness": int(correctness), "confidence_score": float(conf),
                "score": score,
                "search_queries": sqs, "generations": gens
            })
        
        pid = 1
        for a, b in combinations(samples, 2):
            gap = abs(a["score"] - b["score"])
            if gap > MIN_GAP:
                pos, neg = (a, b) if a["score"] > b["score"] else (b, a)
                records.append({
                    "qid": qid, "pid": pid, "query": query,
                    "positive_sample": pack(pos), "negative_sample": pack(neg)
                })
                pid += 1         
    train_preference_df = pd.DataFrame.from_records(records, columns=["qid", "pid", "query", "positive_sample", "negative_sample"])
    
    train_preference_df_str = train_preference_df.astype(str)
    train_preference_ds = Dataset.from_pandas(train_preference_df_str)
    
    # --- Test set: create perefrence pairs
    test_df_str = test_df.astype(str)
    test_ds = Dataset.from_pandas(test_df_str)
    
    dataset_dict = DatasetDict({"train": train_preference_ds, 'test': test_ds})
    return dataset_dict

def training(args):
    # === Load dataset ==========
    prompt_template = get_prompt_template(args.prompt_format)
    RAG_METHODS = ['self_ask', 'react', 'search_o1', 'research', 'search_r1']
    dataset = data_creation(args)
    
    # === Load model ============
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    
    # === Printing Samples ======
    print('---')
    selected_train_sample = dataset['train'][0]
    selected_train_sample_pos_tuple = ast.literal_eval(selected_train_sample['positive_sample'])
    selected_train_sample_pos_str = prompt_template.format(
            sep_token=tokenizer.sep_token,
            query=selected_train_sample["query"],
            answer=selected_train_sample_pos_tuple[0],
            conf_score=selected_train_sample_pos_tuple[1],
            rag_method=selected_train_sample_pos_tuple[2],
            search_queries=' '.join(str(g) for g in selected_train_sample_pos_tuple[4] if g),
            generations=' '.join(str(g) for g in selected_train_sample_pos_tuple[5] if g),
        )
    print(f"Train sample: {selected_train_sample}")
    print(f'Train Prompt:\n{selected_train_sample_pos_str}')
    
    print('\n---')
    selected_test_sample = dataset['test'][0]
    selected_test_sample_rag_tuple = ast.literal_eval(selected_test_sample[RAG_METHODS[0]])
    selected_test_sample_rag_str = prompt_template.format(
            sep_token=tokenizer.sep_token,
            query=selected_test_sample["query"],
            answer=selected_test_sample_rag_tuple[0],
            conf_score=selected_test_sample_rag_tuple[2],
            rag_method=RAG_METHODS[0],
            search_queries=' '.join(str(g) for g in selected_test_sample_rag_tuple[4] if g),
            generations=' '.join(str(g) for g in selected_test_sample_rag_tuple[3] if g),
        )
    print(f"Test sample:  {dataset['test'][0]}")
    print(f'Test Prompt:\n{selected_test_sample_rag_str}')
    print('---\n')
    
    # === Functions =============
    def preprocess_function(example, idx=None, max_length=5000):
        # -------- TRAIN MODE: pairwise pos/neg --------
        if "positive_sample" in example and "negative_sample" in example:
            positive_sample_tuple = ast.literal_eval(example['positive_sample'])
            negative_sample_tuple = ast.literal_eval(example['negative_sample'])
            pos_conf = positive_sample_tuple[1]
            neg_conf = negative_sample_tuple[1]
            
            pos_sample = prompt_template.format(
                sep_token=tokenizer.sep_token,
                query=example["query"],
                answer=positive_sample_tuple[0],
                conf_score=pos_conf,
                rag_method=positive_sample_tuple[2],
                search_queries=' '.join(str(g) for g in positive_sample_tuple[4] if g),
                generations=' '.join(str(g) for g in positive_sample_tuple[5] if g),
            )
            neg_sample = prompt_template.format(
                sep_token=tokenizer.sep_token,
                query=example["query"],
                answer=negative_sample_tuple[0],
                conf_score=neg_conf,
                rag_method=negative_sample_tuple[2],
                search_queries=' '.join(str(g) for g in negative_sample_tuple[4] if g),
                generations=' '.join(str(g) for g in negative_sample_tuple[5] if g),
            )
            pos_encoded = tokenizer(pos_sample, max_length=max_length, padding=False, truncation=True)
            neg_encoded = tokenizer(neg_sample, max_length=max_length, padding=False, truncation=True)
            
            return {
                "mode": "train",
                "pos_input_ids": pos_encoded["input_ids"],
                "pos_attention_mask": pos_encoded["attention_mask"],
                "neg_input_ids": neg_encoded["input_ids"],
                "neg_attention_mask": neg_encoded["attention_mask"],
            }
            
        # -------- EVAL MODE: multi-candidate per query --------
        else:
            cand_ids, cand_masks, cand_methods, cand_answers, cand_is_correct = [], [], [], [], []
            for k in RAG_METHODS:
                if k in example and example[k]:
                    example_k_tuple = ast.literal_eval(example[k])
                    sample = prompt_template.format(
                        sep_token=tokenizer.sep_token,
                        query=example["query"],
                        answer=example_k_tuple[0],
                        conf_score=example_k_tuple[2],
                        rag_method=k,
                        search_queries=' '.join(str(g) for g in example_k_tuple[4] if g),
                        generations=' '.join(str(g) for g in example_k_tuple[3] if g),
                    )
                    sample_encoded = tokenizer(sample, max_length=max_length, padding=False, truncation=True)
                    cand_ids.append(sample_encoded["input_ids"])
                    cand_masks.append(sample_encoded["attention_mask"])
                    cand_is_correct.append(int(example_k_tuple[1]))
                    
            return {
                "mode": "eval",
                "group_id": int(idx),
                "cand_input_ids": cand_ids,
                "cand_attention_mask": cand_masks,
                "cand_methods": cand_methods,
                "cand_answers": cand_answers,      
                "cand_is_correct": cand_is_correct
            }
    
    @dataclass
    class PairwiseDataCollator:
        tokenizer: Any
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None

        def _pad_batch(self, features: List[Dict[str, Any]]):
            return self.tokenizer.pad(
                features, max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of, return_tensors="pt",
            )
        
        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            mode = features[0].get("mode", "train")
            if mode == "train":
                pos_feats = [{"input_ids": f["pos_input_ids"], "attention_mask": f["pos_attention_mask"]} for f in features]
                neg_feats = [{"input_ids": f["neg_input_ids"], "attention_mask": f["neg_attention_mask"]} for f in features]
                pos_batch = self._pad_batch(pos_feats)
                neg_batch = self._pad_batch(neg_feats)
                return {
                    "mode": "train",
                    "pos_input_ids": pos_batch["input_ids"], "pos_attention_mask": pos_batch["attention_mask"],
                    "neg_input_ids": neg_batch["input_ids"], "neg_attention_mask": neg_batch["attention_mask"],
                }
            
            # ---- EVAL ----
            flat, group_ids, is_correct = [], [], []
            for f in features:
                gid = int(f["group_id"])
                ids_list = f.get("cand_input_ids", [])
                msk_list = f.get("cand_attention_mask", [])
                corr     = f.get("cand_is_correct", [0]*len(ids_list))
                for i in range(len(ids_list)):
                    flat.append({"input_ids": ids_list[i], "attention_mask": msk_list[i]})
                    group_ids.append(gid)
                    is_correct.append(int(corr[i]) if i < len(corr) else 0)
            
            if not flat:
                flat = [{"input_ids": [self.tokenizer.pad_token_id], "attention_mask": [0]}]
                group_ids = [group_ids[0] if group_ids else 0]
                is_correct = [0]

            batch = self._pad_batch(flat)
            labels = torch.stack([torch.tensor(group_ids, dtype=torch.long), torch.tensor(is_correct, dtype=torch.long)], dim=1)
            N = batch["input_ids"].size(0)
            assert labels.size(0) == N, f"labels({labels.size(0)}) != inputs({N})"

            return {
                "mode": "eval",
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": labels,                        # [:,0]=group_id, [:,1]=is_correct
            }
    
    class RewardTrainer(Trainer):
        def __init__(self, model, *args, **kwargs):
            super().__init__(model, *args, **kwargs)
            self.reward_loss_fn = nn.BCEWithLogitsLoss()            # train (pairwise)
            self._bce_none = nn.BCEWithLogitsLoss(reduction="none") # eval (per-candidate)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=1):
            # ---------- TRAIN: pairwise BCE over score differences ----------
            if model.training and ("pos_input_ids" in inputs) and ("neg_input_ids" in inputs):
                pos_outputs = model(input_ids=inputs["pos_input_ids"], attention_mask=inputs["pos_attention_mask"])
                neg_outputs = model(input_ids=inputs["neg_input_ids"], attention_mask=inputs["neg_attention_mask"])
                pos_scores = pos_outputs.logits.squeeze(-1)  # [B]
                neg_scores = neg_outputs.logits.squeeze(-1)  # [B]
                score_diff = pos_scores - neg_scores
                labels = torch.ones_like(score_diff, device=score_diff.device)
                loss = self.reward_loss_fn(score_diff, labels)
                # return (loss, {"pos_scores": pos_scores.detach(), "neg_scores": neg_scores.detach()}) if return_outputs else loss
                return (loss, {"logits": score_diff.detach()}) if return_outputs else loss
       
        
            # ---- EVAL: listwise CE if 1 positive; else pointwise BCE averaged per query ----
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
            scores = outputs.logits.squeeze(-1)  # [N_total]

            labels = inputs["labels"]            # [N_total, 2]
            group_ids = labels[:, 0].long()
            y = labels[:, 1].to(dtype=scores.dtype)  # 0/1

            per_query_losses = []
            for g in torch.unique(group_ids):
                m = (group_ids == g)
                s_g = scores[m]      # [K]
                y_g = y[m]           # [K]
                if s_g.numel() == 0:
                    continue
                pos_count = int(torch.sum(y_g).item())
                if pos_count == 1:
                    # Listwise softmax CE
                    log_probs = F.log_softmax(s_g, dim=0)
                    pos_idx = torch.argmax(y_g).item()
                    per_query_losses.append(-log_probs[pos_idx])
                else:
                    # Pointwise BCE averaged per query (your requested behavior)
                    bce = self._bce_none(s_g, y_g)  # [K]
                    per_query_losses.append(bce.mean())

            loss = torch.stack(per_query_losses).mean() if per_query_losses else scores.new_zeros(())

            # Always give logits so Trainer can gather predictions
            return (loss, {"logits": scores.detach()}) if return_outputs else loss
        
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds, labels = np.asarray(preds).squeeze(), np.asarray(labels)

        if preds.ndim == 0:
            preds = np.array([preds], dtype=float)
        if labels.ndim == 1:
            labels = labels.reshape(-1, 2)

        if preds.shape[0] != labels.shape[0]:
            raise ValueError(f"Pred/label length mismatch: preds={preds.shape[0]} labels={labels.shape[0]}")

        group_ids = labels[:, 0].astype(np.int64, copy=False)
        is_correct = labels[:, 1].astype(np.int64, copy=False)

        acc_count, total = 0, 0
        for g in np.unique(group_ids):
            idx = np.nonzero(group_ids == g)[0]
            if idx.size == 0:
                total += 1
                continue
            
            total += 1
            if not np.any(is_correct[idx] == 1):
                continue # skip groups with no gold (or count as 0 if you prefer)
            
            top_i = idx[np.argmax(preds[idx])]
            acc_count += int(is_correct[top_i] == 1)
           
        return {
            "acc@1": (acc_count / total) if total > 0 else 0.0,
            "eval_groups_counted": int(total),
        }
    
    # === Training ... ==========
    model_ = args.model_name_or_path.split('/')[-1]
    model_output_dir = f'models/rag_selection/pairwise_input/{model_}'
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_labels=1,
        cache_dir=args.cache_dir, reference_compile=False
    )
    tokenized_dataset = dataset.map(preprocess_function, with_indices=True)
    data_collator = PairwiseDataCollator(tokenizer)
    
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        weight_decay=0.0,
        num_train_epochs=5,
        greater_is_better=True,
        load_best_model_at_end=True,
        lr_scheduler_type="linear",
        warmup_ratio=0.05,
        eval_strategy="epoch",
        save_strategy="epoch",
        # eval_steps=200,
        logging_steps=150,
        remove_unused_columns=False,
        save_total_limit=2, 
        metric_for_best_model="acc@1",
        report_to=[],  # disable W&B etc. unless you want it
        seed=args.seed
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
    # Model
    parser.add_argument('--model_name_or_path', type=str, default='answerdotai/ModernBERT-base')
    parser.add_argument('--saved_model_name_or_path', type=str, default='models/rag_selector/checkpoint-800')
    parser.add_argument('--secondary_model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--cache_dir', type=str, default='./cache/')
    parser.add_argument("--max_input_tokens", type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=128)

    # Dataset
    parser.add_argument('--dataset', type=str, default='hotpotqa', choices=[
        'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'
    ])
    parser.add_argument('--subsec', type=str, default='train', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
    parser.add_argument("--enable_fewshot_examples", action="store_true", help="")
    parser.add_argument('--prompt_format', type=str, default='x_p_o_c', choices=['o_c', 'x_o_c', 'p_o_c', 'x_p_o_c', 'x_g_o_c', 'x_p_o'])
    
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
    parser.add_argument('--run', type=str, default='run_1 (rag_methods_2k)')
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument('--use_counter', action='store_false')
    
    args = parser.parse_args()
    
    # === Define CUDA device =======
    args.device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. No GPUs detected.")
    
    args.semantic_equivalence_prompt_file = "run_mcts_two_actions/prompts/semantic_equivalence_prompt_template.txt"
    
    ### === Run Steps =============
    set_seed(args.seed)
    
    # add_correctness(args)
    training(args)
    
    # python applications/rag_selector/confidence_in_input/pairwise_run.py
    # accelerate launch --multi_gpu applications/rag_selector/confidence_in_input/pairwise_run.py

