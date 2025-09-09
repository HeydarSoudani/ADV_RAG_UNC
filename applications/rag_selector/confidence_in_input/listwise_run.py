#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import ast
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict, Any
from transformers import Trainer, TrainingArguments, AutoModel, AutoTokenizer
from datasets import Dataset, DatasetDict

from utils.general_utils import set_seed
from applications.rag_selector.confidence_in_input.pairwise_run import (
    merge_rag_systems_data,
    drop_all_same_correctness,
    get_prompt_template
)

def join_skip_none(xs):
    if not xs:
        return ""
    return ", ".join(str(x) for x in xs if x is not None and str(x) != "")

# -- --------------------------------------
def listnet_loss(scores: torch.Tensor,
                 labels: torch.Tensor,
                 target_variant: str = "softmax_labels") -> torch.Tensor:
    """
    scores: [B, K] (higher is better)
    labels: [B, K] (binary or graded)
    target_variant:
        - "uniform_pos"     : uniform over positives (your current behavior)
        - "softmax_labels"  : ListNet top-1 with softmax over labels (classic)
        - "proportional"    : labels / sum(labels) (for nonnegative graded labels)
    """
    # Valid examples: at least one positive (or nonzero sum)
    if target_variant == "uniform_pos":
        pos_counts = (labels > 0).sum(dim=-1, keepdim=True)  # [B,1]
        valid = (pos_counts > 0)
        target = torch.where(labels > 0,
                             1.0 / pos_counts.clamp(min=1.0),
                             torch.zeros_like(labels))
    elif target_variant == "softmax_labels":
        # Works with graded labels (can be any real; consider scaling if very large)
        valid = (labels.sum(dim=-1, keepdim=True) > 0)
        target = F.softmax(labels, dim=-1)
    elif target_variant == "proportional":
        # For nonnegative labels; distribute mass proportional to label value
        lbl = labels.clamp(min=0)
        sums = lbl.sum(dim=-1, keepdim=True)
        valid = (sums > 0)
        target = lbl / sums.clamp(min=1.0)
    else:
        raise ValueError(f"Unknown target_variant: {target_variant}")

    log_probs = F.log_softmax(scores, dim=-1)           # [B, K]
    per_example = -(target * log_probs).sum(dim=-1)     # [B]

    if valid.any():
        return per_example[valid.squeeze(-1)].mean()
    else:
        # No valid lists in batch -> return zero (or torch.tensor(0., device=scores.device))
        return scores.new_tensor(0.0)

def listmle_loss(scores, labels, tie_mode: str = "stable"):
    B, K = labels.shape
    perm = torch.argsort(labels, dim=-1, descending=True)
    s_sorted = torch.gather(scores, 1, perm)               # [B, K]
    lse = torch.logcumsumexp(s_sorted.flip(-1), dim=-1).flip(-1)
    ll = (s_sorted - lse).sum(dim=-1)                      # [B]
    sum_labels = labels.sum(dim=-1)                        # [B]
    valid = (sum_labels > 0) & (sum_labels < K)            # bool [B]

    if valid.any():
        return -(ll[valid]).mean()
    else:
        # Return a zero that's connected to the graph
        return scores.sum() * 0.0

def lambda_ndcg_loss(scores: torch.Tensor,
                     labels: torch.Tensor,
                     sigma: float = 1.0,
                     use_exp_gains: bool = True,
                     eps: float = 1e-12) -> torch.Tensor:
    """
    LambdaRank-style pairwise logistic loss weighted by ΔNDCG.

    scores: [B, K] or [K]
    labels: [B, K] or [K]  (binary or graded; larger = better)
    sigma : slope for pairwise logistic (RankNet)
    use_exp_gains: if True, gains = 2^labels - 1 (standard DCG); else gains = labels
    """
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)

    scores = scores.float()
    labels = labels.float()
    B, K = scores.shape

    if K < 2:
        return scores.sum() * 0.0  # 0, but connected to graph

    # Pairwise diffs S_ij = s_i - s_j (B,K,K)
    S = scores.unsqueeze(-1) - scores.unsqueeze(-2)

    # Off-diagonal mask
    eye = torch.eye(K, device=scores.device, dtype=torch.bool)
    mask_offdiag = ~eye  # (K,K) bool

    # Gains for DCG
    gains = (2.0 ** labels - 1.0) if use_exp_gains else labels.clamp_min(0.0)

    # ----- Compute ΔNDCG weights with no grad -----
    with torch.no_grad():
        # Ideal DCG (IDCG)
        ideal_order = torch.argsort(labels, dim=-1, descending=True)
        pos_idx = torch.arange(K, device=scores.device, dtype=torch.float)
        ideal_discounts = 1.0 / torch.log2(2.0 + pos_idx)              # (K,)
        ideal_gains = torch.gather(gains, -1, ideal_order)             # (B,K)
        idcg = (ideal_gains * ideal_discounts).sum(dim=-1, keepdim=True) + eps  # (B,1)

        # Predicted ranks by current scores (0-based ranks)
        pred_order = torch.argsort(scores, dim=-1, descending=True)    # (B,K)
        pred_ranks = torch.empty_like(scores)                          # float (B,K)
        src = pos_idx.unsqueeze(0).expand(B, K)                        # (B,K)
        pred_ranks.scatter_(-1, pred_order, src)

        discounts = 1.0 / torch.log2(2.0 + pred_ranks)                 # (B,K)

        # ΔDCG for swapping i and j: (g_i - g_j) * (disc_j - disc_i)
        gi = gains.unsqueeze(-1)                                       # (B,K,1)
        gj = gains.unsqueeze(-2)                                       # (B,1,K)
        di = discounts.unsqueeze(-1)                                   # (B,K,1)
        dj = discounts.unsqueeze(-2)                                   # (B,1,K)
        delta_dcg = (gi - gj) * (dj - di)                              # (B,K,K)
        delta_ndcg = (delta_dcg.abs() / idcg.unsqueeze(-1))            # (B,K,K)

        # Only consider pairs where rel_i > rel_j
        rel_diff = labels.unsqueeze(-1) - labels.unsqueeze(-2)         # (B,K,K)
        pos_pairs = (rel_diff > 0) & mask_offdiag                      # bool (B,K,K)

    # Pairwise logistic loss weighted by ΔNDCG
    # softplus(-x) = log(1 + exp(-x)) = -log(sigmoid(x))
    pair_loss = F.softplus(-sigma * S) * delta_ndcg * pos_pairs.float()

    # Normalize per-list by number of positive pairs
    denom = pos_pairs.float().sum(dim=(1, 2)).clamp_min(1.0)           # (B,)
    loss_per_list = pair_loss.sum(dim=(1, 2)) / denom                  # (B,)
    return loss_per_list.mean()

def lambda_ndcg_at1_loss(scores: torch.Tensor,
                         labels: torch.Tensor,
                         sigma: float = 1.0,
                         use_exp_gains: bool = True,
                         eps: float = 1e-12) -> torch.Tensor:
    """
    LambdaRank-style pairwise logistic loss focused on NDCG@1 (i.e., Acc@1).
    scores: [B, K] or [K]
    labels: [B, K] or [K] (binary or graded; larger = better)
    Returns a scalar loss.
    """
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)

    scores = scores.float()
    labels = labels.float()
    B, K = scores.shape
    if K < 2:
        return scores.sum() * 0.0

    # Gains
    gains = (2.0 ** labels - 1.0) if use_exp_gains else labels.clamp_min(0.0)

    # Current predicted top indices and values
    top_idx = torch.argmax(scores, dim=-1)                          # [B]
    top_scores = scores.gather(1, top_idx.unsqueeze(1)).squeeze(1)  # [B]
    top_gains  = gains.gather(1,  top_idx.unsqueeze(1)).squeeze(1)  # [B]

    # Best possible gain at rank 1 (IDCG@1); for binary it’s 1 if any positive else 0
    idcg1 = gains.max(dim=-1, keepdim=False).values + eps           # [B]

    # For each item i, compare against current top
    # Pairwise diffs s_i - s_top
    S_it = scores - top_scores.unsqueeze(1)                         # [B, K]

    # Only include items strictly better than the current top (rel_i > rel_top)
    better_than_top = (gains > top_gains.unsqueeze(1))              # [B, K]

    # ΔNDCG@1 weight for replacing the top with i: (g_i - g_top)/IDCG@1
    delta_ndcg1 = (gains - top_gains.unsqueeze(1)) / idcg1.unsqueeze(1)  # [B, K]
    delta_ndcg1 = torch.clamp(delta_ndcg1, min=0.0)                 # only positive deltas matter

    # Exclude the top item itself from loss (its delta is 0 anyway)
    eye_mask = torch.zeros_like(scores, dtype=torch.bool)
    eye_mask.scatter_(1, top_idx.unsqueeze(1), True)
    mask = better_than_top & (~eye_mask)                            # [B, K]

    # Pairwise logistic loss: softplus(-sigma * (s_i - s_top))
    per_item_loss = F.softplus(-sigma * S_it) * delta_ndcg1 * mask.float()  # [B, K]

    # Normalize per list by count of valid pairs (items better than current top)
    denom = mask.float().sum(dim=-1).clamp_min(1.0)                 # [B]
    loss_per_list = per_item_loss.sum(dim=-1) / denom               # [B]

    # If a list has no positives at all, idcg1≈eps and mask is all False -> contributes 0 (no signal)
    return loss_per_list.mean()


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
    """
    Top-1 accuracy: 1 if the highest-scoring candidate is relevant (label>0), else 0.
    Supports multiple correct or no correct candidates.
    """
    top_pred = np.argmax(scores, axis=1)        # [B]
    hits = labels[np.arange(len(labels)), top_pred] > 0
    return float(np.mean(hits))


### ==== Main Functions =================== 
def data_creation(args, train=True, test=True):
    if train:
        train_df = merge_rag_systems_data(args, subsec='train')
        rag_methods = [c for c in train_df.columns if c not in ("qid", "query", "gt_answers")]
        # 1) Remove sample with all 0 or all 1
        train_df = drop_all_same_correctness(train_df, rag_methods)
        train_df_str = train_df.astype(str)
        train_ds = Dataset.from_pandas(train_df_str)
    
    if test:
        test_df = merge_rag_systems_data(args, subsec='test')
        test_df_str = test_df.astype(str)
        test_ds = Dataset.from_pandas(test_df_str)
    
    if train and test:
        dataset_dict = DatasetDict({"train": train_ds, "test": test_ds})
    elif train and not test:
        dataset_dict = DatasetDict({"train": train_ds})
    elif not train and test:
        dataset_dict = DatasetDict({"test": test_ds})
    
    print(dataset_dict)
    return dataset_dict

def training(args):
    # === Load dataset ==========
    RAG_METHODS = ['self_ask', 'react', 'search_o1', 'research', 'search_r1']
    dataset = data_creation(args)
   
    print('---')
    print(f"Train sample: {dataset['train'][0]}")
    print('---')
    print(f"Test sample:  {dataset['test'][0]}")
    print('---')
    
    # === Load model ============
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)

    # === Functions =============
    def preprocess_function(example, max_length=5000):
        ctx = example["query"]
        ids_list, attn_list, typeids_list, labels_list = [], [], [], []
        for method in RAG_METHODS:
            method_item_tuple = ast.literal_eval(example[method])
            prediction = method_item_tuple[0]
            confidence_score = method_item_tuple[2]
            generations = join_skip_none(method_item_tuple[3]) if len(method_item_tuple) > 3 else ''
            search_queries = join_skip_none(method_item_tuple[4]) if len(method_item_tuple) > 4 else ''
            
            # resp = f"{prediction} {confidence_score}"
            resp = f"{prediction} {confidence_score} {generations} {search_queries}"
        
            enc = tokenizer(ctx, resp, truncation=True, max_length=max_length, padding=False)
            ids_list.append(enc["input_ids"])
            attn_list.append(enc["attention_mask"])
            typeids_list.append(enc.get("token_type_ids", [0] * len(enc["input_ids"])))
            labels_list.append(method_item_tuple[1])
        
        return {
            "input_ids": ids_list,          # list of lists
            "attention_mask": attn_list,    # list of lists
            "token_type_ids": typeids_list,   # List[List[int]] (for BERT)
            "labels": labels_list,          # list[float|int]
            "num_candidates": len(ids_list),
        }
       
    @dataclass
    class ListwiseDataCollator:
        tokenizer: Any
        pad_to_multiple_of: int = None
        dtype_labels: torch.dtype = torch.float  # change to long if your loss needs ints

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            # --- Fixed K sanity check ---
            K = len(features[0]["input_ids"])
            for f in features:
                if len(f["input_ids"]) != K or len(f["attention_mask"]) != K or len(f["labels"]) != K:
                    raise ValueError("All samples must have the same number of candidates (fixed K).")

            # --- Determine max sequence length (L_max) in this batch ---
            L_max = 0
            for f in features:
                for seq in f["input_ids"]:
                    if len(seq) > L_max:
                        L_max = len(seq)
            if self.pad_to_multiple_of:
                L_max = ((L_max + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_id is None:
                raise ValueError("Tokenizer has no pad_token_id; set one before batching.")
            
            # --- Allocate tensors ---
            B = len(features)
            input_ids      = torch.full((B, K, L_max), pad_id, dtype=torch.long)
            attention_mask = torch.zeros((B, K, L_max), dtype=torch.long)
            token_type_ids = torch.zeros((B, K, L_max), dtype=torch.long)  # BERT: segment IDs
            labels         = torch.empty((B, K), dtype=self.dtype_labels)

            # --- Fill tensors ---
            for i, f in enumerate(features):
                # token_type_ids may be per-candidate or absent
                ttis = f.get("token_type_ids", None)
                for k in range(K):
                    ids  = f["input_ids"][k]
                    attn = f["attention_mask"][k]
                    L = len(ids)
                    input_ids[i, k, :L] = torch.as_tensor(ids,  dtype=torch.long)
                    attention_mask[i, k, :L] = torch.as_tensor(attn, dtype=torch.long)

                    if ttis is not None:
                        # Support either a list-of-lists or a single list reused for all candidates
                        if isinstance(ttis, list) and len(ttis) == K:
                            token_type_ids[i, k, :L] = torch.as_tensor(ttis[k], dtype=torch.long)
                        else:
                            token_type_ids[i, k, :L] = torch.as_tensor(ttis, dtype=torch.long)
                    # else: leave zeros

                    labels[i, k] = float(f["labels"][k])

            return {
                "input_ids": input_ids,           # (B, K, L_max)
                "attention_mask": attention_mask, # (B, K, L_max)
                # "token_type_ids": token_type_ids, # (B, K, L_max)
                "labels": labels,                 # (B, K)
            }
      
    class RewardTrainer(Trainer):
        def __init__(self, *args, loss_name: str = "listmle", **kwargs):
            super().__init__(*args, **kwargs)
            self.loss_name = loss_name             

        @staticmethod
        def _get_scores_from_output(output):
            if isinstance(output, torch.Tensor):
                return output
            if hasattr(output, "logits"):
                return output.logits
            if isinstance(output, (list, tuple)) and len(output) > 0:
                return output[0]
            raise ValueError("Model forward did not return logits/scores in a recognizable format.")

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=1):
            inputs = self._prepare_inputs(inputs)
            labels = inputs.pop("labels", None)
            if labels is None:
                raise ValueError("`labels` missing from inputs for listwise loss.")

            output = model(**inputs)         # expect scores with shape [B, K]
            scores = self._get_scores_from_output(output)
            labels = labels.float()

            if self.loss_name == "listnet":
                loss = listnet_loss(scores, labels)
            elif self.loss_name == "listmle":
                loss = listmle_loss(scores, labels)
            elif self.loss_name == "lambda_ndcg":
                loss = lambda_ndcg_loss(scores, labels)
            elif self.loss_name == "lambda_ndcg_at1":
                loss = lambda_ndcg_at1_loss(scores, labels)
            else:
                raise ValueError(f"Unknown loss '{self.loss_name}'")

            return (loss, {"scores": scores}) if return_outputs else loss
    
        def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
            has_labels = "labels" in inputs
            inputs = self._prepare_inputs(inputs)
            labels = inputs.pop("labels", None) if has_labels else None

            with torch.no_grad():
                output = model(**inputs)
                scores = self._get_scores_from_output(output)
                loss = None
                if has_labels:
                    labels_f = labels.float()
                    if self.loss_name == "listnet":
                        loss = listnet_loss(scores, labels_f)
                    elif self.loss_name == "listmle":
                        loss = listmle_loss(scores, labels_f)
                    elif self.loss_name == "lambda_ndcg_at1":
                        loss = lambda_ndcg_at1_loss(scores, labels)
                    else:
                        loss = lambda_ndcg_loss(scores, labels_f)

            if prediction_loss_only:
                return (loss, None, None)

            return (loss, scores, labels if has_labels else None)
    
    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        labels = eval_pred.label_ids

        return {
            "ndcg@1": ndcg_at_k_np(preds, labels, k=1),
            "ndcg@3": ndcg_at_k_np(preds, labels, k=3),
            "ndcg@K": ndcg_at_k_np(preds, labels, k=preds.shape[1]),
            "mrr": mrr_np(preds, labels),
            "acc@1": acc1_np(preds, labels),
        }

    class BertRewardRanker(nn.Module):
        """
        Expects:
        input_ids:       [B, K, L]
        attention_mask:  [B, K, L]
        token_type_ids:  [B, K, L] (optional; used if provided)
        Returns:
        scores:          [B, K]
        """
        def __init__(self, model_name: str = "bert-base-uncased", head_hidden: int = 256, dropout: float = 0.1):
            super().__init__()
            self.bert = AutoModel.from_pretrained(model_name)
            h = self.bert.config.hidden_size
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(h, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, 1),
            )

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor = None,
            **kwargs,
        ) -> torch.Tensor:
            # 
            B, K, L = input_ids.shape
            flat_ids   = input_ids.reshape(B * K, L)
            flat_mask  = attention_mask.reshape(B * K, L)

            bert_kwargs = dict(input_ids=flat_ids, attention_mask=flat_mask, return_dict=True)
            if token_type_ids is not None:
                bert_kwargs["token_type_ids"] = token_type_ids.reshape(B*K, L)

            out = self.bert(**bert_kwargs)
            rep = out.pooler_output if getattr(out, "pooler_output", None) is not None \
                else out.last_hidden_state[:, 0, :]
            scores = self.head(rep).reshape(B, K).contiguous()
            return scores

    # === Training ... ===========
    model_ = args.model_name_or_path.split('/')[-1]
    model_output_dir = f'models/rag_selection/listwise_input/{model_}'
    model = BertRewardRanker(args.model_name_or_path)
    tokenized_dataset = dataset.map(preprocess_function)
    data_collator = ListwiseDataCollator(tokenizer)
    
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",   # or "steps"
        save_strategy="epoch",
        logging_steps=100,
        save_total_limit=2,
        metric_for_best_model="acc@1",
        load_best_model_at_end=True,
        greater_is_better=True,
        save_safetensors=True,
        remove_unused_columns=False,
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
        loss_name="listmle",   # listmle / listnet / lambda_ndcg / lambda_ndcg_at1
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
    parser.add_argument("--max_input_tokens", type=int, default=4096)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='hotpotqa', choices=[
        'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'
    ])
    parser.add_argument('--subsec', type=str, default='dev', choices=['train', 'dev', 'test', 'validation'])
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
    
    
# python applications/rag_selector/confidence_in_input/listwise_run.py
