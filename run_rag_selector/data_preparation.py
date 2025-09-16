#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import bisect
import random
import argparse
import numpy as np
import transformers
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any
from itertools import combinations
from collections.abc import Sequence
from datasets import Dataset, DatasetDict, load_from_disk

from utils.general_utils import set_seed
from run_mcts.run_mcts_two_actions.src.models.semantic_equivalence import SemanticEquivalenceGenerator


### === Helper =======================
def pack_cluster(s):
    return (s["query"], s["prediction"], s["sum_confidence_score"], s["correctness"], s["all_search_queries"], s["all_thinks"], s["all_docs"], s["all_generations"])

def stack_rag_methods_to_candidates(df: pd.DataFrame) -> pd.DataFrame:
    # Identify the rag method columns (anything except qid/query)
    method_cols = [c for c in df.columns if c not in ("qid", "query")]
    
    def row_to_candidates(row):
        out = []
        for c in method_cols:
            v = row[c]
            if pd.isna(v):
                continue
            # Make sure each entry is a tuple (merge can turn tuples into lists/objects)
            if isinstance(v, tuple):
                out.append(v)
            elif isinstance(v, list):
                out.append(tuple(v))
            else:
                # Unexpected scalar/object -> skip
                continue
        return out

    new_df = df[["qid", "query"]].copy()
    new_df["candidates"] = df.apply(row_to_candidates, axis=1)
    return new_df

def merge_rag_systems_data(args, subsec='train'):
    if subsec == 'train':
        run = args.run_train
        dataset_subsec = 'train'
        correctness_m = 'llm_as_judge'
    else:
        run = args.run_test
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
    merged_df = stack_rag_methods_to_candidates(merged_df)
    
    return merged_df

def extract_correctness(x):
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes)) and len(x) > 2:
        return x[2]
    return np.nan

def drop_all_same_correctness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where all candidates have correctness=0 or all correctness=1.
    Assumes df has columns: qid, query, candidates
    where candidates is a list of tuples shaped like:
        (pred, conf, correctness, search_queries, thinks, docs, gens)
    """
    def is_all_same_correctness(candidates):
        if not candidates:
            return True  # treat empty as "drop"
        correctness_vals = [c[2] for c in candidates if len(c) > 2]
        if not correctness_vals:
            return True
        return all(v == 0 for v in correctness_vals) or all(v == 1 for v in correctness_vals)

    mask = df["candidates"].apply(lambda c: not is_all_same_correctness(c))
    return df.loc[mask].reset_index(drop=True)

def clustered_samples(args, df):
    
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
    
    model = transformers.AutoModelForCausalLM.from_pretrained(args.secondary_model_name_or_path, dtype=torch.bfloat16).to(args.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.secondary_model_name_or_path)
    cluster_based_rag_selector = ClusteringModel(model, tokenizer, args.device, args)
    new_rows = []
    for i, row in tqdm(df.iterrows()):
        qid, query, org_candidates = row["qid"], row["query"], row["candidates"] 
        candidates = []
        for val in org_candidates:
            candidates.append((val[0], val[1], val[2], val[3], val[4], val[5], val[6]))  # (pred, conf, correctness, sq, thinks, docs, gens)
        
        clusters_summaries = cluster_based_rag_selector.inference(query, candidates)
        new_rows.append({"qid": qid, "query": query, "candidates": clusters_summaries})
    
    new_df = pd.DataFrame(new_rows)
    return new_df

def create_perefrence_pairs(
    clustered_df: pd.DataFrame,
    add_cross_queries: bool = False,
    cross_samples: int = 1000,
    near_ratio: float = 0.8, # For B
    min_gap: float = 0.8,
    seed: int = 42,
):
    rng = random.Random(seed)
    records = []

    # -------- Pass 1: build intra-query pairs and gather global pools --------
    global_pos_pool, global_neg_pool = [], []
    intra_A = intra_B = intra_C = 0
    
    for _, row in clustered_df.iterrows():
        qid, query, clusters = row["qid"], row["query"], row["candidates"]
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
        "[create_preference_pairs] Intra-query summary: "
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
            "[create_preference_pairs] Cross-query summary: "
            f"A (pos_conf < neg_conf) = {created_A}, "
            f"B (|pos_conf - neg_conf| < {min_gap}) = {created_B}, "
            f"total = {created_A + created_B} (requested {cross_samples})"
        )
    else:
        if cross_samples > 0:
            print("[create_preference_pairs] Cross-query pairs not generated.")
        
    # -------- Finalize --------
    preference_df = pd.DataFrame.from_records(records, columns=["qid", "pid", "positive_sample", "negative_sample"])
    preference_df_str = preference_df.astype(str)
    preference_ds = Dataset.from_pandas(preference_df_str)
    return preference_ds

def data_creation(args):
    # == Train setup ---------
    train_df = merge_rag_systems_data(args, subsec='train')
    train_df = drop_all_same_correctness(train_df) # Filtering: Remove sample with all 0 or all 1
    train_df = clustered_samples(args, train_df)
    train_preference_ds = create_perefrence_pairs(train_df, add_cross_queries=True)
    
    # == Test setup ----------
    test_df = merge_rag_systems_data(args, subsec=args.subsec)
    if args.with_clustering:
        test_df = clustered_samples(args, test_df)
    test_df_str = test_df.astype(str)
    test_ds = Dataset.from_pandas(test_df_str)
    
    dataset_dict = DatasetDict({"train": train_preference_ds, 'test': test_ds})
    return dataset_dict


### === Main ========================== 
def data_preparation(args):
    # pairwise-listwise   | clustring 
    
    ### build_or_load_dataset
    os.makedirs(args.data_cache_dir, exist_ok=True)
    clustering_text = 'clustering' if args.with_clustering  else 'wo_clustering'
    cache_dir = f"{args.data_cache_dir}/{args.dataset}/{args.training_method}_{clustering_text}"

    if os.path.exists(cache_dir):
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

def add_new_correctness(args):
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
        file_path = f"run_output/{args.run_train}/{rag_method[0]}/{args.dataset}_{args.subsec}/{rag_method[1]}_{args.retriever_name}/{args.consistency_method}_results.jsonl"
        new_result_file_path = f"run_output/{args.run_train}/{rag_method[0]}/{args.dataset}_{args.subsec}/{rag_method[1]}_{args.retriever_name}/{args.consistency_method}_results_new.jsonl"
        
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--selector_model_name_or_path', type=str, default='answerdotai/ModernBERT-large', choices=[
        'answerdotai/ModernBERT-large', 'BAAI/bge-large-en-v1.5', 'google/embeddinggemma-300m',
        'Alibaba-NLP/gte-Qwen2-7B-instruct', 'Alibaba-NLP/gte-Qwen2-1.5B-instruct' # https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct
    ])
    parser.add_argument('--secondary_model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--cache_dir', type=str, default='./cache/')
    parser.add_argument("--data_cache_dir", type=str, default="./run_rag_selector/datasets")
    parser.add_argument("--max_input_tokens", type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--dataset', type=str, default='musique', choices=[
        'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'
    ])
    parser.add_argument('--subsec', type=str, default='train', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--prompt_format', type=str, default='x_o_c_g_dc', choices=[
        'x_o', 'x_o_sq', 'x_o_th', 'x_o_dc', 'x_o_g', 'x_o_g_sq', 'x_o_g_dc', 'x_o_sq_dc', 'x_o_sq_th_dc',
        'x_o_c', 'x_o_c_sq', 'x_o_c_th', 'x_o_c_dc', 'x_o_c_g', 'x_o_c_g_sq', 'x_o_c_g_dc', 'x_o_c_sq_dc', 'x_o_c_sq_th_dc',
    ])
    parser.add_argument('--n_docs_prompt', type=int, default=3)
    parser.add_argument('--retriever_name', type=str, default='rerank_l6', choices=[
        'bm25', 'contriever', 'rerank_l6', 'rerank_l12', 'e5', 'bge', 'reasonir'
    ])
    parser.add_argument('--consistency_method', type=str, default='rag_consistency', choices=[
        'self_consistency', 'reasoning_consistency', 'rag_consistency'
    ])
    # 
    parser.add_argument('--get_ideal', action='store_false')
    parser.add_argument('--with_training', action='store_true')
    parser.add_argument('--with_clustering', action='store_true')
    parser.add_argument('--confidence_score_injection', type=str, default='in_input', choices=['in_input', 'in_representation'])
    parser.add_argument('--training_method', type=str, default='pairwise', choices=['pairwise', 'listwise'])
    # 
    parser.add_argument('--run_train', type=str, default='run_1 (rag_methods_2k)')
    parser.add_argument('--run_test', type=str, default='run_2 (rag_methods_1k)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--retry", type=int, default=3)
    args = parser.parse_args()
    
    # === Define CUDA device =======
    args.device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. No GPUs detected.")
    
    # === Add variables
    args.rag_methods = ['self_ask', 'react', 'search_o1', 'research', 'search_r1']
    args.semantic_equivalence_prompt_file = "run_mcts/run_mcts_two_actions/prompts/semantic_equivalence_prompt_template.txt"
    
    set_seed(args.seed)
    add_new_correctness(args)
    # main(args)
    
    
    # python run_rag_selector/data_preparation.py
    