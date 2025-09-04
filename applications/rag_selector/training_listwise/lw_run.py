#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import ast
import json
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict, Any
from transformers import Trainer, TrainingArguments, AutoModel, AutoTokenizer
from datasets import Dataset, DatasetDict, load_dataset

from utils.general_utils import set_seed

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
    rag_methods = [title for title, value in dataset['train'].features.items() if title not in ['qid', 'query']]
    
    
    # === Load model ============
    model_path = 'answerdotai/ModernBERT-base'
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)


    # === Functions =============
    class BertRewardRanker(nn.Module):
        """
        Expects:
        input_ids:      [B, K, L]
        attention_mask: [B, K, L]
        Returns:
        scores:         [B, K]
        """
        def __init__(self, model_name="bert-base-uncased", head_hidden=256, dropout=0.1, device='cuda'):
            super().__init__()
            
            self.device = device
            self.bert = AutoModel.from_pretrained(model_name)
            h = self.bert.config.hidden_size
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(h, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, 1)
            )

        def forward(self, input_ids, attention_mask):
            B, K, L = input_ids.shape
            flat_ids = input_ids.view(B*K, L)
            flat_mask = attention_mask.view(B*K, L)

            out = self.bert(input_ids=flat_ids, attention_mask=flat_mask, return_dict=True)
            # pooled_output if present; otherwise CLS
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                rep = out.pooler_output
            else:
                rep = out.last_hidden_state[:, 0, :]
            scores = self.head(rep).view(B, K).contiguous()
            return scores

    def listnet_loss(scores, labels):
        # labels: [B,K] (0/1 or ratings). Skip lists with no positives.
        pos_counts = labels.sum(dim=-1, keepdim=True)  # [B,1]
        valid = (pos_counts > 0).float()
        P_y = torch.where(labels > 0, 1.0 / pos_counts.clamp(min=1.0), torch.zeros_like(labels))
        P_s = F.softmax(scores, dim=-1)
        ce = -(P_y * torch.log(P_s + 1e-12)).sum(dim=-1)
        return (ce * valid.squeeze(-1)).sum() / valid.sum().clamp(min=1.0)

    def listmle_loss(scores, labels):
        # positives first; ties arbitrary. Skip all-0 or all-1.
        B, K = labels.shape
        perm = torch.argsort(labels, dim=-1, descending=True)  # [B,K]
        s_sorted = torch.gather(scores, 1, perm)
        lse = torch.logcumsumexp(s_sorted.flip(-1), dim=-1).flip(-1)
        ll = (s_sorted - lse).sum(dim=-1)
        valid = ((labels.sum(dim=-1) > 0) & (labels.sum(dim=-1) < K)).float()
        return (-(ll) * valid).sum() / valid.sum().clamp(min=1.0)

    def _dcg_gain(labels):
        # for binary labels, this is labels; for graded labels you can do (2**labels - 1)
        return labels.float()

    def lambda_ndcg_loss(scores, labels, eps=1e-12):
        """
        LambdaRank-style pairwise loss weighted by ΔNDCG.
        Works with:
        - scores: [B, K] or [K]
        - labels: [B, K] or [K]  (0/1 or graded)
        Returns a scalar loss.
        """
        # ---- Normalize shapes to [B, K] ----
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)

        scores = scores.float()
        labels = labels.float()
        B, K = scores.shape

        # If K < 2, no pairwise signal
        if K < 2:
            # create a 0.0 loss that still connects to graph
            return (scores.sum() * 0.0) + 0.0

        # Pairwise score diffs S_ij = s_i - s_j
        S = scores.unsqueeze(-1) - scores.unsqueeze(-2)      # [B, K, K]

        # Mask out diagonal i==j
        eye = torch.eye(K, device=scores.device, dtype=scores.dtype)
        mask_offdiag = (1 - eye).unsqueeze(0)               # [1, K, K]

        # Gains (for binary labels, this is labels; for graded you may prefer (2**labels - 1))
        gains = labels

        # Ideal DCG (for normalization)
        ideal_order = torch.argsort(labels, dim=-1, descending=True)  # [B, K]
        pos_idx = torch.arange(K, device=scores.device).float()       # [K]
        ideal_discounts = 1.0 / torch.log2(2.0 + pos_idx)             # [K]
        ideal_gains = torch.gather(gains, -1, ideal_order)            # [B, K]
        idcg = (ideal_gains * ideal_discounts).sum(dim=-1, keepdim=True) + eps  # [B, 1]

        # Predicted ranks (no torchsort): use current score order as proxy
        pred_order = torch.argsort(scores, dim=-1, descending=True)   # [B, K]
        pred_ranks = torch.empty_like(scores, dtype=torch.float)      # [B, K]
        src = pos_idx.unsqueeze(0).expand(B, K)                       # [B, K] values 0..K-1
        pred_ranks.scatter_(-1, pred_order, src)                      # fill ranks per row

        discounts = 1.0 / torch.log2(2.0 + pred_ranks)                # [B, K]

        # ΔNDCG weights for swapping i and j ≈ |(g_i - g_j) * (disc_j - disc_i)| / IDCG
        gain_i = gains.unsqueeze(-1)            # [B, K, 1]
        gain_j = gains.unsqueeze(-2)            # [B, 1, K]
        disc_i = discounts.unsqueeze(-1)        # [B, K, 1]
        disc_j = discounts.unsqueeze(-2)        # [B, 1, K]
        delta_dcg = (gain_i - gain_j) * (disc_j - disc_i)             # [B, K, K]
        delta_ndcg = torch.abs(delta_dcg) / idcg.unsqueeze(-1)        # [B, K, K]

        # Consider only pairs where rel_i > rel_j
        rel_diff = (labels.unsqueeze(-1) - labels.unsqueeze(-2))      # [B, K, K]
        pos_pairs = (rel_diff > 0).float() * mask_offdiag             # [B, K, K]

        # Pairwise logistic loss weighted by ΔNDCG
        pair_loss = F.softplus(-S) * pos_pairs * delta_ndcg           # [B, K, K]

        # Normalize per list by number of positive pairs present
        denom = pos_pairs.sum(dim=(1, 2)).clamp_min(1.0)              # [B]
        loss_per_list = pair_loss.sum(dim=(1, 2)) / denom             # [B]
        return loss_per_list.mean()

    def preprocess_function(example, max_length=256):
        ctx = example["query"]
        ids_list, attn_list, labels_list = [], [], []
        for method in rag_methods:
            method_item_tuple = ast.literal_eval(example[method])
            resp = f'{method_item_tuple[0]}, {method_item_tuple[2]}'
            enc = tokenizer(
                ctx,
                resp,
                truncation=True,
                max_length=max_length,
                padding=False,  # don't pad here; collator will do dynamic padding
                return_tensors=None,
            )
            ids_list.append(enc["input_ids"])
            attn_list.append(enc["attention_mask"])
            labels_list.append(method_item_tuple[1])
        
        return {
            "input_ids": ids_list,          # list of lists
            "attention_mask": attn_list,    # list of lists
            "labels": labels_list,          # list[float|int]
        }
        
    @dataclass
    class ListwiseDataCollator:
        tokenizer: Any
        pad_to_multiple_of: int = None  # optional

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            # features[i]["input_ids"] is a list[K][Lvar]
            B = len(features)
            K = len(features[0]["input_ids"])
            # Determine max L in this batch
            max_len = 0
            for f in features:
                for seq in f["input_ids"]:
                    max_len = max(max_len, len(seq))
            if self.pad_to_multiple_of:
                # round up to multiple (useful for tensor cores)
                ceil = ((max_len + self.pad_to_multiple_of - 1)//self.pad_to_multiple_of)*self.pad_to_multiple_of
                max_len = ceil

            input_ids = torch.full((B, K, max_len), self.tokenizer.pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((B, K, max_len), dtype=torch.long)
            labels = torch.tensor([f["labels"] for f in features], dtype=torch.float)

            for i, f in enumerate(features):
                for k in range(K):
                    ids = f["input_ids"][k]
                    attn = f["attention_mask"][k]
                    L = len(ids)
                    input_ids[i, k, :L] = torch.tensor(ids, dtype=torch.long)
                    attention_mask[i, k, :L] = torch.tensor(attn, dtype=torch.long)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
        }

    def ndcg_at_k_np(scores, labels, k=None):
        # scores/labels: [B,K] numpy
        B, K = scores.shape
        k = k or K
        ndcgs = []
        for b in range(B):
            order = np.argsort(-scores[b])
            top = order[:k]
            gains = (2**labels[b] - 1.0)
            dcg = np.sum(gains[top] / np.log2(np.arange(2, k+2)))
            ideal_order = np.argsort(-labels[b])[:k]
            idcg = np.sum(gains[ideal_order] / np.log2(np.arange(2, k+2))) + 1e-12
            ndcgs.append(dcg / idcg)
        return float(np.mean(ndcgs))

    def mrr_np(scores, labels):
        # labels are binary or graded; treat any >0 as relevant for MRR
        B, K = scores.shape
        mrrs = []
        for b in range(B):
            order = np.argsort(-scores[b])
            rel = (labels[b] > 0).astype(np.float32)
            rr = 0.0
            for rank, idx in enumerate(order, start=1):
                if rel[idx] > 0:
                    rr = 1.0 / rank
                    break
            mrrs.append(rr)
        return float(np.mean(mrrs))

    def acc1_np(scores, labels):
        pred = np.argmax(scores, axis=1)
        gold = np.argmax(labels, axis=1)
        return float(np.mean(pred == gold))

    def compute_metrics(eval_pred):
        # eval_pred: transformers.EvalPrediction
        # predictions: [B,K], label_ids: [B,K]
        preds = eval_pred.predictions
        if isinstance(preds, tuple):  # just in case
            preds = preds[0]
        labels = eval_pred.label_ids
        return {
            "ndcg@K": ndcg_at_k_np(preds, labels, k=preds.shape[1]),
            "mrr": mrr_np(preds, labels),
            "acc@1": acc1_np(preds, labels),
        }
    
    class RewardTrainer(Trainer):
        def __init__(self, *args, loss_name="listmle", **kwargs):
            super().__init__(*args, **kwargs)
            self.loss_name = loss_name

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=1):
            labels = inputs.pop("labels").to(model.device)
            scores = model(**inputs)  # [B, K]
            if self.loss_name == "listnet":
                loss = listnet_loss(scores, labels)
            elif self.loss_name == "listmle":
                loss = listmle_loss(scores, labels)
            elif self.loss_name == "lambda_ndcg":
                loss = lambda_ndcg_loss(scores, labels)
            else:
                raise ValueError(f"Unknown loss '{self.loss_name}'")
            return (loss, scores) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
            model.eval()
            has_labels = "labels" in inputs
            labels = inputs.pop("labels", None)
            # move to device
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                scores = model(**inputs)  # tensor [B, K]
                loss = None
                if has_labels:
                    labels_t = labels.to(model.device).float()
                    if self.loss_name == "listnet":
                        loss = listnet_loss(scores, labels_t)
                    elif self.loss_name == "listmle":
                        loss = listmle_loss(scores, labels_t)
                    else:
                        loss = lambda_ndcg_loss(scores, labels_t)

            # IMPORTANT: return torch.Tensors (not numpy)
            if has_labels:
                return (loss, scores, labels.to(scores.device))
            else:
                return (loss, scores, None)


    # === Training ... ===========
    model_ = model_path.split('/')[-1]
    model_output_dir = f'models/rag_selection_reward_modeling/list_wise/{model_}'
    model = BertRewardRanker(model_path, device=args.device)
    data_collator = ListwiseDataCollator(tokenizer)
    tokenized_dataset = dataset.map(preprocess_function)
    
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=4,     # note: batch is B (lists), not pairs
        per_device_eval_batch_size=4,
        remove_unused_columns=False,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=10,
        weight_decay=0.01,
        eval_strategy="epoch",   # or "steps"
        save_strategy="epoch",
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        greater_is_better=True,
        save_safetensors=True,
        report_to=[],  # disable W&B etc. unless you want it
        seed=args.seed,
    )
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        loss_name="lambda_ndcg",   # listmle / listnet / soft_ndcg
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
    training(args)
    
# python applications/rag_selector/training_listwise/lw_run.py 





























    # def lambda_ndcg_loss(scores, labels, eps=1e-12):
    #     """
    #     scores: [B, K]
    #     labels: [B, K] (0/1 or graded)
    #     Implements LambdaRank-style pairwise loss, weighted by ΔNDCG for each pair.
    #     """
    #     B, K = scores.shape
    #     # Sort by predicted scores per list
    #     S = scores.unsqueeze(2) - scores.unsqueeze(1)  # [B, K, K] pairwise score diffs
    #     # Only consider i != j
    #     mask_offdiag = 1 - torch.eye(K, device=scores.device).unsqueeze(0)

    #     # Relevance and gains
    #     rel = labels.float()
    #     gains = _dcg_gain(rel)

    #     # Ideal DCG for normalization
    #     ideal_order = torch.argsort(rel, dim=-1, descending=True)
    #     arange = torch.arange(K, device=scores.device).float()
    #     ideal_discounts = 1.0 / torch.log2(2.0 + arange)  # rank 0 -> 1/log2(2)
    #     ideal_gains = torch.gather(gains, 1, ideal_order)
    #     idcg = (ideal_gains * ideal_discounts).sum(dim=-1, keepdim=True) + eps  # [B,1]

    #     # Approx predicted ranks via sorting *indices* (no torchsort): use current order as proxy
    #     pred_order = torch.argsort(scores, dim=-1, descending=True)
    #     pred_ranks = torch.empty_like(pred_order, dtype=torch.float)
    #     pred_ranks.scatter_(1, pred_order, arange)  # rank positions 0..K-1
    #     discounts = 1.0 / torch.log2(2.0 + pred_ranks)  # [B, K]

    #     # ΔNDCG weights for pairs (i,j): |Δ(DCG)/IDCG|
    #     # Approx change in DCG if i and j swapped: (gain_i - gain_j) * (disc_j - disc_i)
    #     gain_i = gains.unsqueeze(2)  # [B,K,1]
    #     gain_j = gains.unsqueeze(1)  # [B,1,K]
    #     disc_i = discounts.unsqueeze(2)  # [B,K,1]
    #     disc_j = discounts.unsqueeze(1)  # [B,1,K]
    #     delta_dcg = (gain_i - gain_j) * (disc_j - disc_i)      # [B,K,K]
    #     delta_ndcg = torch.abs(delta_dcg) / idcg.unsqueeze(-1) # [B,K,K]

    #     # Pairwise logistic loss weighted by ΔNDCG, only for rel_i > rel_j
    #     rel_diff = (rel.unsqueeze(2) - rel.unsqueeze(1))        # [B,K,K]
    #     pos_pairs = (rel_diff > 0).float() * mask_offdiag       # upper triangle of beneficial swaps

    #     # logistic: log(1 + exp(-S)) encourages S = score_i - score_j to be large when rel_i>rel_j
    #     pair_loss = F.softplus(-S) * pos_pairs * delta_ndcg
    #     # Normalize by number of positive pairs per list
    #     denom = pos_pairs.sum(dim=(1,2)).clamp_min(1.0)         # [B]
    #     loss_per_list = pair_loss.sum(dim=(1,2)) / denom
    #     return loss_per_list.mean()