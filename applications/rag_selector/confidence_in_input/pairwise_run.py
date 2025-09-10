#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import re
import ast
import math
import json, hashlib
import bisect
import torch
from pathlib import Path
import random
import argparse
import numpy as np
import transformers
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import unicodedata as ud
from collections.abc import Sequence
import torch.nn.functional as F
from dataclasses import dataclass
from itertools import combinations
from typing import List, Dict, Any, Optional
from datasets import Dataset, DatasetDict, load_from_disk
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import Trainer, TrainingArguments, AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from safetensors.torch import load_file


# pd.set_option("display.max_rows", None)   # show all rows
# pd.set_option("display.max_columns", None)  # show all columns
# pd.set_option("display.width", None)      # no line wrapping
# pd.set_option("display.max_colwidth", None)  # full cell content

from utils.general_utils import set_seed
from run_mcts_two_actions.src.models.semantic_equivalence import SemanticEquivalenceGenerator

# -- --------------------------------------
def get_prompt_template(prompt_format):
    if prompt_format == 'x_o':
        prompt_template = '{query}{sep_token}{answer}'
    elif prompt_format == 'x_o_sq':
        prompt_template = '{query}{sep_token}{answer}{sep_token}{search_queries}'
    elif prompt_format == 'x_o_th':
        prompt_template = '{query}{sep_token}{answer}{sep_token}{thinks}'
    elif prompt_format == 'x_o_dc':
        prompt_template = '{query}{sep_token}{answer}{sep_token}{docs}'
    elif prompt_format == 'x_o_g':
        prompt_template = '{query}{sep_token}{answer}{sep_token}{generations}'
    elif prompt_format == 'x_o_g_sq':
        prompt_template = '{query}{sep_token}{answer}{sep_token}{generations}{sep_token}{search_queries}'
    elif prompt_format == 'x_o_g_dc':
        prompt_template = '{query}{sep_token}{answer}{sep_token}{generations}{sep_token}{docs}'
    elif prompt_format == 'x_o_sq_dc':
        prompt_template = '{query}{sep_token}{answer}{sep_token}{search_queries}{sep_token}{docs}'
    elif prompt_format == 'x_o_sq_th_dc':
        prompt_template = '{query}{sep_token}{answer}{sep_token}{search_queries}{sep_token}{thinks}{sep_token}{docs}'
    
    elif prompt_format == 'x_o_c':
        prompt_template = '{query}{sep_token}{answer}{sep_token}{conf_score}'
    elif prompt_format == 'x_o_c_sq':
        prompt_template = '{query}{sep_token}{answer}{sep_token}{conf_score}{sep_token}{search_queries}'
    elif prompt_format == 'x_o_c_th':
        prompt_template = '{query}{sep_token}{answer}{sep_token}{conf_score}{sep_token}{thinks}'
    elif prompt_format == 'x_o_c_dc':
        prompt_template = '{query}{sep_token}{answer}{sep_token}{conf_score}{sep_token}{docs}'
    elif prompt_format == 'x_o_c_g':
        prompt_template = '{query}{sep_token}{answer}{sep_token}{conf_score}{sep_token}{generations}'
    elif prompt_format == 'x_o_c_g_sq':
        prompt_template = '{query}{sep_token}{answer}{sep_token}{conf_score}{sep_token}{generations}{sep_token}{search_queries}'
    elif prompt_format == 'x_o_c_g_dc':
        prompt_template = '{query}{sep_token}{answer}{sep_token}{conf_score}{sep_token}{generations}{sep_token}{docs}'
    elif prompt_format == 'x_o_c_sq_dc':
        prompt_template = '{query}{sep_token}{answer}{sep_token}{conf_score}{sep_token}{search_queries}{sep_token}{docs}'
    elif prompt_format == 'x_o_c_sq_th_dc':
        prompt_template = '{query}{sep_token}{answer}{sep_token}{conf_score}{sep_token}{search_queries}{sep_token}{thinks}{sep_token}{docs}'

    return prompt_template

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

def pack_cluster(s):
    return (s["query"], s["prediction"], s["sum_confidence_score"], s["correctness"], s["all_search_queries"], s["all_thinks"], s["all_docs"], s["all_generations"])

def merge_rag_systems_data(args, subsec='train'):
    if subsec == 'train':
        run = 'run_1 (rag_methods_2k)'
        dataset_subsec = 'train'
        correctness_m = 'llm_as_judge'
    else:
        run = 'run_3 (rag_methods_500)'
        dataset_subsec = subsec
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
        search_queries_map, thinks_map, docs_map = {}, {}, {}
        path_file = f"run_output/{run}/{rag_method[0]}/{args.dataset}_{dataset_subsec}/{rag_method[1]}_{args.retriever_name}/inference_results.jsonl"
        if os.path.exists(path_file):
            with open(path_file, "r") as f:
                corr_data = [json.loads(line) for line in f]
            for item in corr_data:
                if "path" in item and isinstance(item["path"], list):
                    sqs = [d.get("search_query") for d in item["path"] if isinstance(d, dict) and "search_query" in d]
                    thinks = [d.get("think") for d in item["path"] if isinstance(d, dict) and "think" in d]
                    search_queries_map[item["qid"]] = sqs
                    thinks_map[item["qid"]] = thinks
                    
                    docs = []
                    for d in item["path"]:
                        if isinstance(d, dict) and "docs" in d:
                            for doc in d["docs"]:
                                docs.append(doc)
                    unique_docs = {doc["id"]: doc['contents'] for doc in docs if "id" in doc}.values()
                    docs_map[item["qid"]] = list(unique_docs)

        # --- Add tuple including search queries
        df_temp[rag_method[1]] = [
            (
                pred, conf, correctness,
                search_queries_map.get(qid, []), thinks_map.get(qid, []), docs_map.get(qid, []),
                gens,
            )
            for qid, pred, conf, correctness, gens in zip(
                df_temp["qid"], df_temp["pred_answer"],
                confidences, df_temp[correctness_m],
                df_temp["final_answer_list"]
            )
        ]
        df = df_temp[["qid", "query", rag_method[1]]]
        dfs.append(df)
    
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(df, on=["qid", "query"], how="outer")
    merged_df = merged_df.sort_values(by="qid", key=lambda x: x.str.extract(r"(\d+)").squeeze().astype(int)).reset_index(drop=True)
    
    return merged_df

def create_perefrence_pairs_v1(df, rag_methods):
    W_EM, W_CONF, MIN_GAP = 0.5, 0.5, 0.4
    records = []
    for _, row in df.iterrows():
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
    
    preference_df = pd.DataFrame.from_records(records, columns=["qid", "pid", "query", "positive_sample", "negative_sample"])
    preference_df_str = preference_df.astype(str)
    preference_ds = Dataset.from_pandas(preference_df_str)
    
    return preference_ds
 
def clustered_samples(args, df, rag_methods):
    
    class ClusteringModel:
        def __init__(self, model, tokenizer, device, args):
            self.se_model = SemanticEquivalenceGenerator(args, device, model, tokenizer)
            self.device = device
            self.args = args
            self.rng = random.Random(args.seed)
            
        def get_clusters(self, question, candidates):
            clusters = []
            for cand in candidates:
                pred = cand[0]
                placed = False

                for cluster in clusters:
                    rep_pred = cluster[0][0]
                    if self.se_model.check_answers_equiv(question, pred, rep_pred):
                        cluster.append(cand)
                        placed = True
                        break

                if not placed:
                    clusters.append([cand])

            return clusters
        
        def summarize_clusters(self, clusters) -> List[Dict[str, Any]]:
            summaries, all_sqs, all_thinks, all_docs, all_generations = [], [], [], [], []
            for cluster in clusters:
                if not cluster:
                    continue
                
                confidence_sum = sum(c[1] for c in cluster)
                
                positives = [c[0] for c in cluster if int(c[2]) == 1]
                if positives:
                    correctness = 1
                    pred_representative = self.rng.choice(positives)
                else:
                    correctness = 0
                    pred_representative = self.rng.choice([c[0] for c in cluster])

                # 
                for c in cluster:
                    all_sqs.extend(c[3])
                    all_thinks.extend(c[4])
                    all_docs.extend(c[5])
                    all_generations.extend(c[6])

                summaries.append((
                    pred_representative, confidence_sum, correctness,
                    all_sqs, all_thinks, all_docs, all_generations
                ))

            return summaries
            
        def select_final_answer(self, clusters_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
            if not clusters_summaries:
                return {"prediction": None, "correctness": None, "chosen_cluster": None}
            
            max_conf = max(c["confidence_sum"] for c in clusters_summaries)
            top_clusters = [c for c in clusters_summaries if c["confidence_sum"] == max_conf]
            chosen_cluster = self.rng.choice(top_clusters)
            prediction, _, _ = chosen_cluster["representative"]
            return prediction, chosen_cluster["correctness"], max_conf
        
        def inference(self, question, candidates):
            clusters = self.get_clusters(question, candidates)
            clusters_summaries = self.summarize_clusters(clusters)
            # prediction, correctness, max_conf = self.select_final_answer(clusters_summaries)
            return clusters_summaries
    
    model = transformers.AutoModelForCausalLM.from_pretrained(args.semantic_model_name_or_path, dtype=torch.bfloat16).to(args.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.semantic_model_name_or_path)
    cluster_based_rag_selector = ClusteringModel(model, tokenizer, args.device, args)
    new_rows = []
    for i, row in tqdm(df.iterrows()):
        # if i == 30:
        #     break
        qid, query = row["qid"], row["query"]
        candidates = []
        for key, val in row.items():
            if key in rag_methods:
                candidates.append((val[0], val[1], val[2], val[3], val[4], val[5], val[6]))  # (pred, conf, correctness, sq, thinks, docs, gens)
        
        clusters_summaries = cluster_based_rag_selector.inference(query, candidates)
        new_rows.append({"qid": qid, "query": query, "clusters": clusters_summaries})
    
    new_df = pd.DataFrame(new_rows)
    return new_df

def create_perefrence_pairs_v2(
    clustered_df: pd.DataFrame,
    add_cross_queries: bool = False,
    cross_samples: int = 3000,
    near_ratio: float = 0.8, # For B
    min_gap: float = 0.5,
    seed: int = 42,
):
    rng = random.Random(seed)
    records = []

    # -------- Pass 1: build intra-query pairs and gather global pools --------
    global_pos_pool, global_neg_pool = [], []
    intra_A = intra_B = intra_C = 0
    
    for _, row in clustered_df.iterrows():
        qid, query, clusters = row["qid"], row["query"], row["clusters"]
        if not isinstance(clusters, (list, tuple)) or len(clusters) < 2:
            continue
        
        # unpack clusters into uniform dicts
        samples = []
        for cluster in clusters:
            prediction, sum_conf, correctness, sq, thinks, docs, gens = cluster
            s = {
                "query": query,
                "prediction": prediction,
                "sum_confidence_score": float(sum_conf),
                "correctness": int(correctness),
                "all_search_queries": sq,
                "all_thinks": thinks,
                "all_docs": docs,
                "all_generations": gens,
            }
            samples.append(s)
            
            # fill global pools
            if s["correctness"] == 1:
                global_pos_pool.append({"qid": qid, "sample": s})
            else:
                global_neg_pool.append({"qid": qid, "sample": s})
                
        pid = 1
        for a, b in combinations(samples, 2):
            pos = neg = None
            if (a['sum_confidence_score'] < b['sum_confidence_score']) and (a['correctness'] > b['correctness']):
                pos, neg = a, b
                intra_A += 1
            elif (a['sum_confidence_score'] > b['sum_confidence_score']) and (a['correctness'] < b['correctness']):
                pos, neg = b, a
                intra_B += 1
            elif (a['correctness'] != b['correctness']) and (abs(a["sum_confidence_score"] - b["sum_confidence_score"]) < min_gap):
                pos, neg = (a, b) if a["correctness"] > b["correctness"] else (b, a)
                intra_C += 1
            
            if pos is not None and neg is not None:
                records.append({
                    "qid": qid, "pid": pid,
                    "positive_sample": pack_cluster(pos),
                    "negative_sample": pack_cluster(neg)
                })
                pid += 1    
    
    
    # Intra-query summary
    intra_total = intra_A + intra_B + intra_C
    print(
        "[create_preference_pairs_v2] Intra-query summary: "
        f"A (pos lower conf & more correct) = {intra_A}, "
        f"B (pos higher conf & more correct) = {intra_B}, "
        f"C (correctness differs & |Δconf| < {min_gap}) = {intra_C}, "
        f"total = {intra_total}"
    )
    
    # -------- Pass 2 (optional): add cross-query pairs --------
    if add_cross_queries and cross_samples > 0 and global_pos_pool and global_neg_pool:
        neg_confs_items = [(neg["sample"]["sum_confidence_score"], idx) for idx, neg in enumerate(global_neg_pool)]
        neg_confs_items.sort(key=lambda x: x[0])
        neg_conf_sorted = [c for c, _ in neg_confs_items]
        neg_idx_sorted = [i for _, i in neg_confs_items]
    
        # Quotas
        if min_gap <= 0:
            target_B = 0
        else:
            target_B = int(round(cross_samples * max(0.0, min(1.0, near_ratio))))
        target_A = max(0, cross_samples - target_B)
    
        # Helper to draw one pair for a specific condition
        def draw_one(mode, seen_pairs, max_global_tries=40, max_local_tries=10):
            """
            mode: 'A' or 'B'
            Returns: (pos_idx, neg_idx) or None
            """
            tries = 0
            while tries < max_global_tries:
                tries += 1
                pos_idx = rng.randrange(len(global_pos_pool))
                pos_item = global_pos_pool[pos_idx]
                pos_qid = pos_item["qid"]
                pos_conf = pos_item["sample"]["sum_confidence_score"]

                if mode == 'A':
                    # Condition A: pos_conf < neg_conf
                    start = bisect.bisect_right(neg_conf_sorted, pos_conf)
                    if start >= len(neg_conf_sorted):
                        continue
                    for _ in range(max_local_tries):
                        k = rng.randrange(start, len(neg_conf_sorted))
                        cand_neg_idx = neg_idx_sorted[k]
                        if (pos_idx, cand_neg_idx) in seen_pairs:
                            continue
                        neg_item = global_neg_pool[cand_neg_idx]
                        if neg_item["qid"] == pos_qid:
                            continue
                        return pos_idx, cand_neg_idx

                else:
                    # Condition B: |pos_conf - neg_conf| < min_gap
                    lo = bisect.bisect_left(neg_conf_sorted, pos_conf - min_gap)
                    hi = bisect.bisect_right(neg_conf_sorted, pos_conf + min_gap)
                    if hi - lo <= 0:
                        continue
                    for _ in range(max_local_tries):
                        k = rng.randrange(lo, hi)
                        cand_neg_idx = neg_idx_sorted[k]
                        if (pos_idx, cand_neg_idx) in seen_pairs:
                            continue
                        neg_item = global_neg_pool[cand_neg_idx]
                        if neg_item["qid"] == pos_qid:
                            continue
                        return pos_idx, cand_neg_idx

            return None
    
        seen_pairs = set()  # (pos_idx, neg_idx)
        next_pid = (max([r["pid"] for r in records if isinstance(r["pid"], int)], default=0) + 1)
        created_A = created_B = 0
        attempts_A = attempts_B = 0
    
        while created_A < target_A and attempts_A < max(50, 20 * target_A):
            attempts_A += 1
            picked = draw_one('A', seen_pairs)
            if picked is None:
                continue
            pos_idx, neg_idx = picked
            seen_pairs.add((pos_idx, neg_idx))

            pos_item = global_pos_pool[pos_idx]
            neg_item = global_neg_pool[neg_idx]
            synthetic_qid = f"{pos_item['qid']}-{neg_item['qid']}"
            # merged_query = f"{pos_item['query']} || {neg_item['query']}"
            records.append({
                "qid": synthetic_qid,
                "pid": next_pid,
                # "query": merged_query,
                "positive_sample": pack_cluster(pos_item["sample"]),
                "negative_sample": pack_cluster(neg_item["sample"]),
            })
            next_pid += 1
            created_A += 1
            
        while created_B < target_B and attempts_B < max(50, 20 * target_B):
            attempts_B += 1
            picked = draw_one('B', seen_pairs)
            if picked is None:
                continue
            pos_idx, neg_idx = picked
            seen_pairs.add((pos_idx, neg_idx))

            pos_item = global_pos_pool[pos_idx]
            neg_item = global_neg_pool[neg_idx]
            synthetic_qid = f"{pos_item['qid']}-{neg_item['qid']}"
            # merged_query = f"{pos_item['query']} || {neg_item['query']}"
            records.append({
                "qid": synthetic_qid,
                "pid": next_pid,
                # "query": merged_query,
                "positive_sample": pack_cluster(pos_item["sample"]),
                "negative_sample": pack_cluster(neg_item["sample"]),
            })
            next_pid += 1
            created_B += 1
            
        # Fallback fill: if either quota underfilled, try the other condition
        total_created = created_A + created_B
        remaining = cross_samples - total_created
        attempts_fill = 0
        while remaining > 0 and attempts_fill < max(100, 30 * remaining):
            attempts_fill += 1
            # Prefer the underfilled bucket
            mode = 'B' if created_B < target_B else 'A'
            picked = draw_one(mode, seen_pairs)
            if picked is None:
                # try the other mode
                picked = draw_one('A' if mode == 'B' else 'B', seen_pairs)
                if picked is None:
                    continue

            pos_idx, neg_idx = picked
            seen_pairs.add((pos_idx, neg_idx))
            pos_item = global_pos_pool[pos_idx]
            neg_item = global_neg_pool[neg_idx]
            synthetic_qid = f"{pos_item['qid']}-{neg_item['qid']}"
            # merged_query = f"{pos_item['query']} || {neg_item['query']}"
            records.append({
                "qid": synthetic_qid,
                "pid": next_pid,
                # "query": merged_query,
                "positive_sample": pack_cluster(pos_item["sample"]),
                "negative_sample": pack_cluster(neg_item["sample"]),
            })
            next_pid += 1
            remaining -= 1
            
        print(
            "[create_preference_pairs_v2] Cross-query summary: "
            f"A (pos_conf < neg_conf) = {created_A}, "
            f"B (|pos_conf - neg_conf| < {min_gap}) = {created_B}, "
            f"total = {created_A + created_B} (requested {cross_samples})"
        )
    else:
        if cross_samples > 0:
            print("[create_preference_pairs_v2] Cross-query pairs not generated.")
        
    # -------- Finalize --------
    preference_df = pd.DataFrame.from_records(records, columns=["qid", "pid", "positive_sample", "negative_sample"])
    preference_df_str = preference_df.astype(str)
    preference_ds = Dataset.from_pandas(preference_df_str)
    return preference_ds


# - Filtering function -----
def extract_correctness(x):
    # robustly get the 3rd element if x is a tuple/list-like
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes)) and len(x) > 2:
        return x[2]
    return np.nan

def drop_all_same_correctness(df: pd.DataFrame, rag_cols):
    # build a correctness-only frame
    corr = df[rag_cols].applymap(extract_correctness)
    # rows where all correctness are 0 or all are 1
    all0 = corr.fillna(-1).eq(0).all(axis=1)
    all1 = corr.fillna(-1).eq(1).all(axis=1)
    # keep everything else
    return df.loc[~(all0 | all1)].reset_index(drop=True)


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
    rag_methods = ['self_ask', 'react', 'search_o1', 'research', 'search_r1']
    
    ### === Train data ======================
    train_df = merge_rag_systems_data(args, subsec='train')
    rag_methods = [c for c in train_df.columns if c not in ("qid", "query")]
    
    ## 1) Filtering: Remove sample with all 0 or all 1
    train_df = drop_all_same_correctness(train_df, rag_methods)
    ## 2) Filtering: Semantically cluster candidates
    clustered_train_df = clustered_samples(args, train_df, rag_methods)
    
    # train_preference_ds = create_perefrence_pairs_v1(train_df, rag_methods)
    train_preference_ds = create_perefrence_pairs_v2(clustered_train_df, add_cross_queries=True)
    
    ### === Test data ======================
    test_df = merge_rag_systems_data(args, subsec=args.subsec)
    clustered_test_df = clustered_samples(args, test_df, rag_methods)
    test_df_str = clustered_test_df.astype(str)
    test_ds = Dataset.from_pandas(test_df_str)
    
    # dataset_dict = DatasetDict({ 'test': test_ds})
    dataset_dict = DatasetDict({"train": train_preference_ds, 'test': test_ds})
    return dataset_dict

def build_or_load_dataset(args, cache_root: str = "./data_cache"):
    cache_root = Path(getattr(args, "data_cache_dir", cache_root))
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_root / f"pw_dataset"

    if cache_dir.exists():
        ds = load_from_disk(str(cache_dir))
        # (Optional) quick sanity check that splits exist
        if isinstance(ds, DatasetDict) and "train" in ds and "test" in ds:
            print(f"[cache] Loaded dataset from {cache_dir}")
            return ds

    # No cache → create and save
    ds = data_creation(args)
    ds.save_to_disk(str(cache_dir))
    print(f"[cache] Saved dataset to {cache_dir}")
    return ds

def training(args):
    # === Load dataset & model ==
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    prompt_template = get_prompt_template(args.prompt_format)
    dataset = build_or_load_dataset(args)
    
    # === Printing Samples ======
    print('------')
    selected_train_sample = dataset['train'][0]
    selected_train_sample_pos_tuple = ast.literal_eval(selected_train_sample['positive_sample'])
    selected_train_sample_pos_str = prompt_template.format(
            sep_token=tokenizer.sep_token,
            query=selected_train_sample_pos_tuple[0],
            answer=selected_train_sample_pos_tuple[1],
            conf_score=selected_train_sample_pos_tuple[2],
            search_queries=' '.join(str(g) for g in selected_train_sample_pos_tuple[4] if g),
            thinks=' '.join(str(g) for g in selected_train_sample_pos_tuple[5] if g),
            docs=' '.join(str(g) for g in selected_train_sample_pos_tuple[6][:args.n_docs_prompt] if g),
            generations=' '.join(str(g) for g in selected_train_sample_pos_tuple[7] if g),
        )
    print(f'Train Prompt:\n{selected_train_sample_pos_str}')
    print('\n---\n')
    selected_test_sample = dataset['test'][0]
    selected_test_sample_tuple = ast.literal_eval(selected_test_sample['clusters'])[0]
    selected_test_sample_tuple_str = prompt_template.format(
            sep_token=tokenizer.sep_token,
            query=selected_test_sample['query'],
            answer=selected_test_sample_tuple[0],
            conf_score=selected_test_sample_tuple[1],
            search_queries=' '.join(str(g) for g in selected_test_sample_tuple[3] if g),
            thinks=' '.join(str(g) for g in selected_test_sample_tuple[4] if g),
            docs=' '.join(str(g) for g in selected_test_sample_tuple[5][:args.n_docs_prompt] if g),
            generations=' '.join(str(g) for g in selected_test_sample_tuple[6] if g),
        )
    print(f'Test Prompt:\n{selected_test_sample_tuple_str}')
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
            }
            
        # -------- EVAL MODE: multi-candidate per query --------
        else:
            cand_ids, cand_masks, cand_is_correct = [], [], []
            clusters_list = ast.literal_eval(example['clusters'])
            for cluster in clusters_list:
                sample = prompt_template.format(
                    sep_token=tokenizer.sep_token,
                    query=example['query'],
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
            
            return {
                "mode": "eval",
                "group_id": int(idx),
                "cand_input_ids": cand_ids,
                "cand_attention_mask": cand_masks,
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

    class PairwiseRewardRanker(nn.Module):
        """
        Pairwise + single scoring model (NO confidence features).

        Train (pairwise):
        - pos_input_ids, pos_attention_mask, [pos_token_type_ids]
        - neg_input_ids, neg_attention_mask, [neg_token_type_ids]
        Returns: loss (pairwise logistic), logits: [B, 2] (pos, neg)

        Eval / predict (flat):
        - input_ids, attention_mask, [token_type_ids]
        Returns: logits: [N] (scores)
        """
        def __init__(
            self,
            model_name: str = "answerdotai/ModernBERT-base",
            head_hidden: int = 256,
            dropout: float = 0.1,
            use_mean_pool: bool = True,  # safer default
        ):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            self.use_mean_pool = use_mean_pool
            h = self.encoder.config.hidden_size

            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(h, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, 1),
            )

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

        def _score(self, input_ids, attention_mask, token_type_ids=None):
            """
            Batched scoring path (no confidence).
            """
            pooled = self._encode(input_ids, attention_mask, token_type_ids)  # [B, H]
            scores = self.head(pooled).squeeze(-1)  # [B]
            return scores

        def forward(
            self,
            # eval / single path
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            # pairwise path
            pos_input_ids=None,
            pos_attention_mask=None,
            pos_token_type_ids=None,
            neg_input_ids=None,
            neg_attention_mask=None,
            neg_token_type_ids=None,
            labels=None,   # ignored (kept for API parity)
            **kwargs,
        ):
            # ---------- Pairwise training ----------
            if pos_input_ids is not None and neg_input_ids is not None:
                pos_scores = self._score(pos_input_ids, pos_attention_mask, pos_token_type_ids)  # [B]
                neg_scores = self._score(neg_input_ids, neg_attention_mask, neg_token_type_ids)  # [B]
                # Pairwise logistic loss: -log σ(s_pos - s_neg)
                diff = pos_scores - neg_scores
                loss = -F.logsigmoid(diff).mean()
                logits = torch.stack([pos_scores, neg_scores], dim=1)  # [B, 2]
                return SequenceClassifierOutput(loss=loss, logits=logits)

            # ---------- Eval / predict ----------
            if input_ids is None:
                raise ValueError("Model.forward needs either pairwise (pos/neg) tensors or flat input_ids for eval.")
            scores = self._score(input_ids, attention_mask, token_type_ids)  # [N]
            return SequenceClassifierOutput(logits=scores.unsqueeze(-1))

    # === Training ... ==========
    model_ = args.model_name_or_path.split('/')[-1]
    model_output_dir = f'models/rag_selection/pairwise_input/{model_}'
    model = PairwiseRewardRanker(args.model_name_or_path)
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
        num_train_epochs=10,
        greater_is_better=True,
        load_best_model_at_end=True,
        lr_scheduler_type="linear",
        warmup_ratio=0.05,
        eval_strategy="epoch",
        save_strategy="epoch",
        eval_steps=2,
        logging_steps=50,
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    prompt_template = get_prompt_template(args.prompt_format)
    dataset = build_or_load_dataset(args)

    # === Printing Samples ======
    print('------')
    selected_test_sample = dataset['test'][0]
    selected_test_sample_tuple = ast.literal_eval(selected_test_sample['clusters'])[0]
    selected_test_sample_tuple_str = prompt_template.format(
            sep_token=tokenizer.sep_token,
            query=selected_test_sample['query'],
            answer=selected_test_sample_tuple[0],
            conf_score=selected_test_sample_tuple[1],
            search_queries=' '.join(str(g) for g in selected_test_sample_tuple[3] if g),
            thinks=' '.join(str(g) for g in selected_test_sample_tuple[4] if g),
            docs=' '.join(str(g) for g in selected_test_sample_tuple[5][:args.n_docs_prompt] if g),
            generations=' '.join(str(g) for g in selected_test_sample_tuple[6] if g),
        )
    print(f'Test Prompt:\n{selected_test_sample_tuple_str}')
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
            }
            
        # -------- EVAL MODE: multi-candidate per query --------
        else:
            cand_ids, cand_masks, cand_is_correct = [], [], []
            clusters_list = ast.literal_eval(example['clusters'])
            for cluster in clusters_list:
                sample = prompt_template.format(
                    sep_token=tokenizer.sep_token,
                    query=example['query'],
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
            
            return {
                "mode": "eval",
                "group_id": int(idx),
                "cand_input_ids": cand_ids,
                "cand_attention_mask": cand_masks,
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

    class PairwiseRewardRanker(nn.Module):
        """
        Pairwise + single scoring model (NO confidence features).

        Train (pairwise):
        - pos_input_ids, pos_attention_mask, [pos_token_type_ids]
        - neg_input_ids, neg_attention_mask, [neg_token_type_ids]
        Returns: loss (pairwise logistic), logits: [B, 2] (pos, neg)

        Eval / predict (flat):
        - input_ids, attention_mask, [token_type_ids]
        Returns: logits: [N] (scores)
        """
        def __init__(
            self,
            model_name: str = "answerdotai/ModernBERT-base",
            head_hidden: int = 256,
            dropout: float = 0.1,
            use_mean_pool: bool = True,  # safer default
        ):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            self.use_mean_pool = use_mean_pool
            h = self.encoder.config.hidden_size

            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(h, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, 1),
            )

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

        def _score(self, input_ids, attention_mask, token_type_ids=None):
            """
            Batched scoring path (no confidence).
            """
            pooled = self._encode(input_ids, attention_mask, token_type_ids)  # [B, H]
            scores = self.head(pooled).squeeze(-1)  # [B]
            return scores

        def forward(
            self,
            # eval / single path
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            # pairwise path
            pos_input_ids=None,
            pos_attention_mask=None,
            pos_token_type_ids=None,
            neg_input_ids=None,
            neg_attention_mask=None,
            neg_token_type_ids=None,
            labels=None,   # ignored (kept for API parity)
            **kwargs,
        ):
            # ---------- Pairwise training ----------
            if pos_input_ids is not None and neg_input_ids is not None:
                pos_scores = self._score(pos_input_ids, pos_attention_mask, pos_token_type_ids)  # [B]
                neg_scores = self._score(neg_input_ids, neg_attention_mask, neg_token_type_ids)  # [B]
                # Pairwise logistic loss: -log σ(s_pos - s_neg)
                diff = pos_scores - neg_scores
                loss = -F.logsigmoid(diff).mean()
                logits = torch.stack([pos_scores, neg_scores], dim=1)  # [B, 2]
                return SequenceClassifierOutput(loss=loss, logits=logits)

            # ---------- Eval / predict ----------
            if input_ids is None:
                raise ValueError("Model.forward needs either pairwise (pos/neg) tensors or flat input_ids for eval.")
            scores = self._score(input_ids, attention_mask, token_type_ids)  # [N]
            return SequenceClassifierOutput(logits=scores.unsqueeze(-1))

    # === Evaluation ... ==========
    # model = PairwiseRewardRanker(args.saved_model_name_or_path)
    model = PairwiseRewardRanker(args.model_name_or_path)
    weights_path = os.path.join(args.saved_model_name_or_path, "model.safetensors")
    state = load_file(weights_path)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:    print("[load] missing keys:", len(missing))
    if unexpected: print("[load] unexpected keys:", len(unexpected))
    model.eval()
    
    tokenized_dataset = dataset.map(preprocess_function, with_indices=True)
    data_collator = PairwiseDataCollator(tokenizer)

    eval_args = TrainingArguments(
        output_dir='./',
        do_train=False,
        do_eval=True,
        per_device_eval_batch_size=4,
        remove_unused_columns=False,      # keep all features used by your custom collator/model
        report_to=[],                     # disable W&B etc.
        seed=args.seed
    )
    
    trainer = RewardTrainer(
        model=model,
        args=eval_args,
        train_dataset=None,
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,  # returns acc@1
    )
    
    metrics = trainer.evaluate()
    print("=== Evaluation Metrics ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model_name_or_path', type=str, default='google/embeddinggemma-300m', choices=[
        'answerdotai/ModernBERT-large', 'BAAI/bge-large-en-v1.5', 'google/embeddinggemma-300m'
    ])
    parser.add_argument('--semantic_model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--saved_model_name_or_path', type=str, default='models/rag_selection/pairwise_input/ModernBERT-large/checkpoint-1768') # 
    parser.add_argument('--secondary_model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--cache_dir', type=str, default='./cache/')
    parser.add_argument("--data_cache_dir", type=str, default="./data/rag_selection")
    parser.add_argument("--max_input_tokens", type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=128)

    # Dataset
    parser.add_argument('--dataset', type=str, default='popqa', choices=[
        'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
    parser.add_argument("--enable_fewshot_examples", action="store_true", help="")
    parser.add_argument('--prompt_format', type=str, default='x_o_c_g_dc', choices=[
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
    # add_correctness(args)
    training(args)
    # inference(args)
    
    # python applications/rag_selector/confidence_in_input/pairwise_run.py
    # accelerate launch --multi_gpu applications/rag_selector/confidence_in_input/pairwise_run.py

