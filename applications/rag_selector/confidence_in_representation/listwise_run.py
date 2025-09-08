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
from applications.rag_selector.confidence_in_input.listwise_run import (
    join_skip_none, data_creation,
    listnet_loss, listmle_loss, lambda_ndcg_loss, lambda_ndcg_at1_loss,
    ndcg_at_k_np, mrr_np, acc1_np
)

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
        ids_list, attn_list, typeids_list, labels_list, conf_list = [], [], [], [], []
        for method in RAG_METHODS:
            method_item_tuple = ast.literal_eval(example[method])
            prediction = method_item_tuple[0]
            generations = join_skip_none(method_item_tuple[3]) if len(method_item_tuple) > 3 else ''
            search_queries = join_skip_none(method_item_tuple[4]) if len(method_item_tuple) > 4 else ''
            
            resp = f"{prediction} {generations} {search_queries}"
            enc = tokenizer(ctx, resp, truncation=True, max_length=max_length, padding=False)
            
            ids_list.append(enc["input_ids"])
            attn_list.append(enc["attention_mask"])
            typeids_list.append(enc.get("token_type_ids", [0] * len(enc["input_ids"])))
            labels_list.append(method_item_tuple[1])
            conf_list.append(method_item_tuple[2])
        
        return {
            "input_ids": ids_list,          # list of lists
            "attention_mask": attn_list,    # list of lists
            "token_type_ids": typeids_list, # List[List[int]] (for BERT)
            "labels": labels_list,          # list[float|int]
            "confidence": conf_list,        # list[float]  <-- NEW
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
                if "confidence" not in f or len(f["confidence"]) != K:
                    raise ValueError("Each sample must include `confidence` with length K.")

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
            confidence     = torch.empty((B, K), dtype=torch.float)

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
                    confidence[i, k] = float(f["confidence"][k])

            return {
                "input_ids": input_ids,           # (B, K, L_max)
                "attention_mask": attention_mask, # (B, K, L_max)
                "labels": labels,                 # (B, K)
                "confidence": confidence,         # (B, K)
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
            confidence:      [B, K] 
        Returns:
            scores:          [B, K]
        """
        def __init__(
            self,
            model_name: str = "bert-base-uncased",
            head_hidden: int = 256,
            dropout: float = 0.1,
            use_mean_pool: bool = False,
        ):
            super().__init__()
            self.bert = AutoModel.from_pretrained(model_name)
            self.use_mean_pool = use_mean_pool
            h = self.bert.config.hidden_size
            in_features = h + 1
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, 1),
            )

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor = None,
            confidence: torch.Tensor = None,
            **kwargs,
        ) -> torch.Tensor:
            if confidence is None:
                raise ValueError("`confidence` tensor is required (shape [B, K]).")
            
            B, K, L = input_ids.shape
            flat_ids   = input_ids.reshape(B * K, L)
            flat_mask  = attention_mask.reshape(B * K, L)

            bert_kwargs = dict(input_ids=flat_ids, attention_mask=flat_mask, return_dict=True)
            if token_type_ids is not None:
                bert_kwargs["token_type_ids"] = token_type_ids.reshape(B * K, L)

            out = self.bert(**bert_kwargs)
            
            if self.use_mean_pool: # mean pooling over valid tokens
                last_hidden = out.last_hidden_state                    # [B*K, L, H]
                mask = flat_mask.unsqueeze(-1).float()                 # [B*K, L, 1]
                pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            else: # CLS pooler if available, else raw CLS
                pooled = getattr(out, "pooler_output", None)
                if pooled is None:
                    pooled = out.last_hidden_state[:, 0, :]            # [B*K, H]
            
            # Concatenate confidence feature
            conf_flat = confidence.reshape(B * K)                      # [B*K]
            conf_flat = conf_flat.to(dtype=pooled.dtype, device=pooled.device).unsqueeze(-1)  # [B*K, 1]

            x = torch.cat([pooled, conf_flat], dim=-1)                 # [B*K, H+1]
            scores = self.head(x).reshape(B, K).contiguous()           # [B, K]
            return scores
            
    
    # === Training ... ===========
    model_ = args.model_name_or_path.split('/')[-1]
    model_output_dir = f'models/rag_selection/listwise_representation/{model_}'
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
    
    
# python applications/rag_selector/confidence_in_representation/listwise_run.py
