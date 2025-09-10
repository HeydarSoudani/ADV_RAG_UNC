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
from typing import List, Dict, Any, Optional
from transformers import Trainer, TrainingArguments, AutoModel, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from utils.general_utils import set_seed
from applications.rag_selector.confidence_in_input.pairwise_run import get_prompt_template, build_or_load_dataset


def training(args):
    # === Load dataset ==========
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    prompt_template = get_prompt_template(args.prompt_format)
    dataset = build_or_load_dataset(args)
    
    # === Load model ============
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    
    # === Printing Samples ======
    print('------')
    selected_train_sample = dataset['train'][0]
    selected_train_sample_pos_tuple = ast.literal_eval(selected_train_sample['positive_sample'])
    selected_train_sample_pos_str = prompt_template.format(
            sep_token=getattr(tokenizer, "pad_token", " "),
            query=selected_train_sample_pos_tuple[0],
            answer=selected_train_sample_pos_tuple[1],
            conf_score=selected_train_sample_pos_tuple[2],
            search_queries=' '.join(str(g) for g in selected_train_sample_pos_tuple[4] if g),
            thinks=' '.join(str(g) for g in selected_train_sample_pos_tuple[5] if g),
            docs=' '.join(str(g) for g in selected_train_sample_pos_tuple[6][:args.n_docs_prompt] if g),
            generations=' '.join(str(g) for g in selected_train_sample_pos_tuple[7] if g),
        )
    # print(f"Train sample: {selected_train_sample}")
    print(f'Train Prompt:\n{selected_train_sample_pos_str}')
    print('\n---\n')
    selected_test_sample = dataset['test'][0]
    selected_test_sample_rag = ast.literal_eval(selected_test_sample['clusters'])[0]
    selected_test_sample_rag_str = prompt_template.format(
            sep_token = getattr(tokenizer, "pad_token", " "),
            query=selected_test_sample["query"],
            answer=selected_test_sample_rag[0],
            conf_score=selected_test_sample_rag[1],
            search_queries=' '.join(str(g) for g in selected_test_sample_rag[3] if g),
            thinks=' '.join(str(g) for g in selected_test_sample_rag[4] if g),
            docs=' '.join(str(g) for g in selected_test_sample_rag[5][:args.n_docs_prompt] if g),
            generations=' '.join(str(g) for g in selected_test_sample_rag[6] if g),
        )
    # print(f"Test sample:  {dataset['test'][0]}")
    print(f'Test Prompt:\n{selected_test_sample_rag_str}')
    print('------\n')

    # === Functions =============
    def preprocess_function(example, idx=None, max_length=5000):
        # -------- TRAIN MODE: pairwise pos/neg --------
        if "positive_sample" in example and "negative_sample" in example:
            positive_sample_tuple = ast.literal_eval(example['positive_sample'])
            negative_sample_tuple = ast.literal_eval(example['negative_sample'])
            pos_conf = positive_sample_tuple[2]
            neg_conf = negative_sample_tuple[2]
            
            pos_sample = prompt_template.format(
                sep_token=tokenizer.sep_token,
                query=positive_sample_tuple[0],
                answer=positive_sample_tuple[1],
                conf_score=pos_conf,
                search_queries=' '.join(str(g) for g in positive_sample_tuple[4] if g),
                thinks=' '.join(str(g) for g in positive_sample_tuple[5] if g),
                docs=' '.join(str(g) for g in positive_sample_tuple[6][:args.n_docs_prompt] if g),
                generations=' '.join(str(g) for g in positive_sample_tuple[7] if g),
            )
            neg_sample = prompt_template.format(
                sep_token=tokenizer.sep_token,
                query=negative_sample_tuple[0],
                answer=negative_sample_tuple[1],
                conf_score=neg_conf,
                search_queries=' '.join(str(g) for g in negative_sample_tuple[4] if g),
                thinks=' '.join(str(g) for g in negative_sample_tuple[5] if g),
                docs=' '.join(str(g) for g in negative_sample_tuple[6][:args.n_docs_prompt] if g),
                generations=' '.join(str(g) for g in negative_sample_tuple[7] if g),
            )
            pos_encoded = tokenizer(pos_sample, max_length=max_length, padding=False, truncation=True)
            neg_encoded = tokenizer(neg_sample, max_length=max_length, padding=False, truncation=True)
            
            return {
                "mode": "train",
                "pos_input_ids": pos_encoded["input_ids"],
                "pos_attention_mask": pos_encoded["attention_mask"],
                "neg_input_ids": neg_encoded["input_ids"],
                "neg_attention_mask": neg_encoded["attention_mask"],
                "pos_confidence": float(pos_conf),
                "neg_confidence": float(neg_conf)
            }
            
        # -------- EVAL MODE: multi-candidate per query --------
        else:
            cand_ids, cand_masks, cand_is_correct, cand_conf = [], [], [], []
            clusters_list = ast.literal_eval(example['clusters'])
            for cluster in clusters_list:
                sample = prompt_template.format(
                    sep_token=tokenizer.sep_token,
                    query=example["query"],
                    answer=cluster[0],
                    conf_score=cluster[1],
                    search_queries=' '.join(str(g) for g in cluster[3] if g),
                    thinks=' '.join(str(g) for g in cluster[4] if g),
                    docs=' '.join(str(g) for g in cluster[5][:args.n_docs_prompt] if g),
                    generations=' '.join(str(g) for g in cluster[6] if g),
                )
                sample_encoded = tokenizer(sample, max_length=max_length, padding=False, truncation=True)
                cand_ids.append(sample_encoded["input_ids"])
                cand_masks.append(sample_encoded["attention_mask"])
                cand_is_correct.append(int(cluster[2]))    
                cand_conf.append(float(cluster[1]))    
            
            return {
                "mode": "eval",
                "group_id": int(idx),
                "cand_input_ids": cand_ids,
                "cand_attention_mask": cand_masks,
                "cand_is_correct": cand_is_correct,
                "cand_confidence": cand_conf
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
                pos_conf = torch.tensor([float(f.get("pos_confidence", 1.0)) for f in features], dtype=torch.float)
                neg_conf = torch.tensor([float(f.get("neg_confidence", 1.0)) for f in features], dtype=torch.float)
                
                return {
                    "mode": "train",
                    "pos_input_ids": pos_batch["input_ids"], "pos_attention_mask": pos_batch["attention_mask"],
                    "neg_input_ids": neg_batch["input_ids"], "neg_attention_mask": neg_batch["attention_mask"],
                    "pos_confidence": pos_conf,  # [B]
                    "neg_confidence": neg_conf,
                }
            
            # ---- EVAL ----
            flat, group_ids, is_correct, flat_conf = [], [], [], []
            for f in features:
                gid = int(f["group_id"])
                ids_list = f.get("cand_input_ids", [])
                msk_list = f.get("cand_attention_mask", [])
                corr     = f.get("cand_is_correct", [0]*len(ids_list))
                confs    = f.get("cand_confidence", [1.0] * len(ids_list))
                
                if len(confs) < len(ids_list):
                    confs = list(confs) + [1.0] * (len(ids_list) - len(confs))
                if len(corr) < len(ids_list):
                    corr = list(corr) + [0] * (len(ids_list) - len(corr))
                
                for i in range(len(ids_list)):
                    flat.append({"input_ids": ids_list[i], "attention_mask": msk_list[i]})
                    group_ids.append(gid)
                    is_correct.append(int(corr[i]) if i < len(corr) else 0)
                    flat_conf.append(float(confs[i]))
            
            if not flat:
                flat = [{"input_ids": [self.tokenizer.pad_token_id], "attention_mask": [0]}]
                group_ids = [group_ids[0] if group_ids else 0]
                is_correct = [0]
                flat_conf = [0.0]

            batch = self._pad_batch(flat)
            labels = torch.stack([torch.tensor(group_ids, dtype=torch.long), torch.tensor(is_correct, dtype=torch.long)], dim=1)
            N = batch["input_ids"].size(0)
            assert labels.size(0) == N, f"labels({labels.size(0)}) != inputs({N})"

            return {
                "mode": "eval",
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": labels,
                "confidence": torch.tensor(flat_conf, dtype=torch.float),
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
    
    class PairwiseRewardRankerWithConf(nn.Module):
        """
        Pairwise + single scoring model with confidence concatenation.

        Train (pairwise):
        - pos_input_ids, pos_attention_mask, [pos_token_type_ids], pos_confidence: [B], ...
        - neg_input_ids, neg_attention_mask, [neg_token_type_ids], neg_confidence: [B], ...
        Returns: loss (pairwise logistic), logits: [B, 2] (pos, neg)

        Eval / predict (flat):
        - input_ids, attention_mask, [token_type_ids], confidence: [N]
        Returns: logits: [N] (scores)
        """
        def __init__(
            self,
            model_name: str = "answerdotai/ModernBERT-base",
            head_hidden: int = 256,
            dropout: float = 0.1,
            use_mean_pool: bool = True,     # safer default
        ):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            self.use_mean_pool = use_mean_pool
            h = self.encoder.config.hidden_size
            in_features = h + 1  # +1 for confidence

            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, 1),
            )

        @staticmethod
        def _ensure_conf(x, n, device, dtype):
            # x: None or tensor-like (shape [n])
            if x is None:
                return torch.ones(n, device=device, dtype=dtype)
            if not torch.is_tensor(x):
                x = torch.tensor(x, dtype=dtype, device=device)
            return x.to(device=device, dtype=dtype).view(n)

        def _encode(self, input_ids, attention_mask, token_type_ids=None):
            """
            input_ids:       [B, L]
            attention_mask:  [B, L]
            token_type_ids:  [B, L] or None
            returns pooled:  [B, H]
            """
            kwargs = dict(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids
            out = self.encoder(**kwargs)

            if self.use_mean_pool:
                last_hidden = out.last_hidden_state  # [B, L, H]
                mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
                pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            else:
                pooled = getattr(out, "pooler_output", None)
                if pooled is None:
                    pooled = out.last_hidden_state[:, 0, :]  # CLS
            return pooled  # [B, H]

        def _score(self, input_ids, attention_mask, token_type_ids, confidence):
            """
            Batched scoring path. confidence: [B]
            """
            device = input_ids.device
            B = input_ids.size(0)
            pooled = self._encode(input_ids, attention_mask, token_type_ids)  # [B, H]
            conf = self._ensure_conf(confidence, B, device, pooled.dtype).unsqueeze(-1)  # [B, 1]
            x = torch.cat([pooled, conf], dim=-1)  # [B, H+1]
            scores = self.head(x).squeeze(-1)      # [B]
            return scores

        def forward(
            self,
            # eval / single path
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            confidence=None,
            # pairwise path
            pos_input_ids=None,
            pos_attention_mask=None,
            pos_token_type_ids=None,
            pos_confidence=None,
            neg_input_ids=None,
            neg_attention_mask=None,
            neg_token_type_ids=None,
            neg_confidence=None,
            labels=None,   # ignored here (your eval labels carry group/is_correct)
            **kwargs,
        ):
            # ---------- Pairwise training ----------
            if pos_input_ids is not None and neg_input_ids is not None:
                pos_scores = self._score(pos_input_ids, pos_attention_mask, pos_token_type_ids, pos_confidence)  # [B]
                neg_scores = self._score(neg_input_ids, neg_attention_mask, neg_token_type_ids, neg_confidence)  # [B]
                # Pairwise logistic loss: -log σ(s_pos - s_neg)
                diff = pos_scores - neg_scores
                loss = -F.logsigmoid(diff).mean()
                logits = torch.stack([pos_scores, neg_scores], dim=1)  # [B, 2]
                return SequenceClassifierOutput(loss=loss, logits=logits)

            # ---------- Eval / predict ----------
            if input_ids is None:
                raise ValueError("Model.forward needs either pairwise (pos/neg) tensors or flat input_ids for eval.")
            scores = self._score(input_ids, attention_mask, token_type_ids, confidence)  # [N]
            return SequenceClassifierOutput(logits=scores.unsqueeze(-1))

    # === Training ... ==========
    model_ = args.model_name_or_path.split('/')[-1]
    model_output_dir = f'models/rag_selection/pairwise_representation/{model_}'
    model = PairwiseRewardRankerWithConf(args.model_name_or_path)
    tokenized_dataset = dataset.map(preprocess_function, with_indices=True)
    data_collator = PairwiseDataCollator(tokenizer)
    
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.0,
        num_train_epochs=8,
        greater_is_better=True,
        load_best_model_at_end=True,
        lr_scheduler_type="linear",
        warmup_ratio=0.05,
        eval_strategy="epoch",
        save_strategy="epoch",
        eval_steps=2,
        logging_steps=100,
        remove_unused_columns=False,
        save_total_limit=2, 
        metric_for_best_model="acc@1",
        report_to=[],
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
    parser.add_argument('--model_name_or_path', type=str, default='answerdotai/ModernBERT-large', choices=[
        'answerdotai/ModernBERT-large', 'BAAI/bge-large-en-v1.5', 'google/embeddinggemma-300m'
    ])
    parser.add_argument('--semantic_model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--secondary_model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--saved_model_name_or_path', type=str, default='models/rag_selector/checkpoint-800')
    parser.add_argument('--cache_dir', type=str, default='./cache/')
    parser.add_argument("--data_cache_dir", type=str, default="./data/rag_selection")
    parser.add_argument("--max_input_tokens", type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=128)

    # Dataset
    parser.add_argument('--dataset', type=str, default='hotpotqa', choices=[
        'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'
    ])
    parser.add_argument('--subsec', type=str, default='dev', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
    parser.add_argument("--enable_fewshot_examples", action="store_true", help="")
    parser.add_argument('--prompt_format', type=str, default='x_o_g_dc', choices=[
        'x_o', 'x_o_sq', 'x_o_th', 'x_o_dc', 'x_o_g', 'x_o_g_sq', 'x_o_g_dc', 'x_o_sq_dc', 'x_o_sq_th_dc',
        'x_o_c', 'x_o_c_sq', 'x_o_c_th', 'x_o_c_dc', 'x_o_c_g', 'x_o_c_g_sq', 'x_o_c_g_dc', 'x_o_c_sq_dc', 'x_o_c_sq_th_dc',
    ])
    parser.add_argument('--n_docs_prompt', type=int, default=3)
    
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
    training(args)
    
    # python applications/rag_selector/confidence_in_representation/pairwise_run.py
    # accelerate launch --multi_gpu applications/rag_selector/confidence_in_representation/pairwise_run.py

