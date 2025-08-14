#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import json
import math
import argparse
import pandas as pd
from itertools import combinations
from utils.general_utils import set_seed
import unicodedata as ud
import re


def clean_text(text: str) -> str:
    if not text:
        return text
    
    # Characters to remove
    BIDI = {0x202A,0x202B,0x202C,0x202D,0x202E,0x2066,0x2067,0x2068,0x2069}
    ZERO_WIDTH = {0x200B,0x200C,0x200D,0xFEFF}
    SOFT_HYPHEN = {0x00AD}
    
    PUNCT_MAP = {
        "“":"\"", "”":"\"", "‘":"'", "’":"'", "—":"-", "–":"-", "…":"..."
    }
    
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
    return (s["ans"], s["em"], s["method"], s["conf"], s["gens"], s["sqs"])

# Extract search_query values
def extract_search_queries(lst):
    if not isinstance(lst, list):
        return None
    return [d.get("search_query") for d in lst if isinstance(d, dict) and "search_query" in d]


def create_training_data(args):
    rag_methods = [
        # ('Qwen2.5-7B-Instruct', 'ircot'),
        # ('Qwen2.5-7B-Instruct', 'flare', 0.08),
        # ('Qwen2.5-7B-Instruct', 'dragin', 0.6),
        ('Qwen2.5-7B-Instruct', 'self_ask'),
        ('Qwen2.5-7B-Instruct', 'react'),
        ('Qwen2.5-7B-Instruct', 'search_o1'),
        ('ReSearch-Qwen-7B-Instruct', 'research'),
        ('SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo', 'search_r1')
    ]
    # output file
    output_path = f"run_output/{args.run}/rag_selection_reward_modeling/{args.dataset}_{args.retriever_name}_{args.consistency_method}/train_preference_data.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # --- Load data
    dfs = []
    for rag_method in rag_methods:
        # - Read main file
        results_file = f"run_output/{args.run}/{rag_method[0]}/{args.dataset}_train/{rag_method[1]}_{args.retriever_name}/{args.consistency_method}_results.jsonl"
        with open(results_file, "r") as f:
            data = [json.loads(line) for line in f]
        df_temp = pd.DataFrame(data)[["qid", "query", "pred_answer", "em", "final_answer_list", "ue_scores"]]
        confidences = df_temp["ue_scores"].apply(lambda x: x["majority_voting"]["confidence"])
        
        # - Add path
        search_queries_map = {}
        path_file = f"run_output/{args.run}/{rag_method[0]}/{args.dataset}_train/{rag_method[1]}_{args.retriever_name}/inference_results.jsonl"
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
                df_temp["pred_answer"], df_temp["em"], confidences, df_temp["qid"], df_temp["final_answer_list"]
            )
        ]
        df = df_temp[["qid", "query", rag_method[1]]]
        dfs.append(df)

    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(df, on=["qid", "query"], how="outer")
    merged_df = merged_df.sort_values(by="qid", key=lambda x: x.str.extract(r"(\d+)").squeeze().astype(int)).reset_index(drop=True)
    
    print(merged_df)
    
    # --- Create perefrence pairs
    method_cols = [c for c in merged_df.columns if c not in ["qid", "query"]]
    records = []
    
    # === V1 =========
    # for _, row in merged_df.iterrows():
    #     qid, query = row["qid"], row["query"]
    #     positives, negatives = [], []

    #     for col in method_cols:
    #         val = row.get(col, None)
    #         if val is None or (isinstance(val, float) and math.isnan(val)):
    #             continue
    #         # Expecting val == (pred_answer, em, confidence)
    #         try:
    #             ans, em, conf, sqs = val
    #         except Exception:
    #             # If the cell isn't a tuple, skip it
    #             continue

    #         if em == 1:
    #             positives.append((ans, em, conf, sqs))
    #         elif em == 0:
    #             negatives.append((ans, em, conf, sqs))

    #     # create all positive x negative pairs for this qid
    #     pid = 1
    #     for p in positives:
    #         for n in negatives:
    #             records.append({
    #                 "qid": qid,
    #                 "pid": pid,
    #                 "query": query,
    #                 "positive_sample": p,   # (answer, 1, confidence, sqs)
    #                 "negative_sample": n,   # (answer, 0, confidence, sqs)
    #             })
    #             pid += 1

    # === V2 =========
    W_EM = 0.5
    W_CONF = 0.5
    MIN_GAP = 0.4  # require |score_i - score_j| > 0.2

    for _, row in merged_df.iterrows():
        
        qid, query = row["qid"], row["query"]
        samples = []
        for col in method_cols:
            val = row.get(col, None)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                continue
            
            try:
                if len(val) == 5:
                    ans, em, conf, gens, sqs = val
                elif len(val) == 3:
                    ans, em, conf = val
                    sqs, gens = None, None
                else:
                    continue
            except Exception:
                continue
            
            if conf is None or (isinstance(conf, float) and math.isnan(conf)):
                continue
                        
            score = W_EM * float(em) + W_CONF * float(conf)

            samples.append({
                "method": col,
                "ans": ans,
                "em": int(em),
                "conf": float(conf),
                "sqs": sqs,
                "gens": [clean_text(g) for g in gens],
                "score": score,
            })
        
        pid = 1
        for a, b in combinations(samples, 2):
            gap = abs(a["score"] - b["score"])
            if gap > MIN_GAP:
                pos, neg = (a, b) if a["score"] > b["score"] else (b, a)
                records.append({
                    "qid": qid,
                    "pid": pid,
                    "query": query,
                    "pair_type": "score_gap",
                    "positive_sample": pack(pos),
                    "negative_sample": pack(neg)
                })
                pid += 1

    preference_df = pd.DataFrame.from_records(records, columns=["qid", "pid", "query", "pair_type", "positive_sample", "negative_sample"])
    preference_df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    print(f"Saved preference dataset to {output_path}")

def create_inference_data(args):
    rag_methods = [
        # ('Qwen2.5-7B-Instruct', 'ircot'),
        # ('Qwen2.5-7B-Instruct', 'flare', 0.08),
        # ('Qwen2.5-7B-Instruct', 'dragin', 0.6),
        ('Qwen2.5-7B-Instruct', 'self_ask'),
        ('Qwen2.5-7B-Instruct', 'react'),
        ('Qwen2.5-7B-Instruct', 'search_o1'),
        ('ReSearch-Qwen-7B-Instruct', 'research'),
        ('SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo', 'search_r1')
    ]
    # output file
    output_path = f"run_output/{args.run}/rag_selection_reward_modeling/{args.dataset}_{args.retriever_name}_{args.consistency_method}/{args.subsec}_inference_data.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # --- Load data
    dfs = []
    for rag_method in rag_methods:
        # - Read main file
        file_path = f"run_output/{args.run}/{rag_method[0]}/{args.dataset}_{args.subsec}/{rag_method[1]}_{args.retriever_name}/{args.consistency_method}_results.jsonl"
        with open(file_path, "r") as f:
            data = [json.loads(line) for line in f]
        df_temp = pd.DataFrame(data)[["qid", "query", "pred_answer", "em", "final_answer_list", "ue_scores"]]
        confidences = df_temp["ue_scores"].apply(lambda x: x["majority_voting"]["confidence"])
        
        # - Add path
        search_queries_map = {}
        path_file = f"run_output/{args.run}/{rag_method[0]}/{args.dataset}_{args.subsec}/{rag_method[1]}_{args.retriever_name}/inference_results.jsonl"
        if os.path.exists(path_file):
            with open(path_file, "r") as f:
                corr_data = [json.loads(line) for line in f]
            for item in corr_data:
                if "path" in item and isinstance(item["path"], list):
                    sqs = [d.get("search_query") for d in item["path"] if isinstance(d, dict) and "search_query" in d]
                    search_queries_map[item["qid"]] = sqs
        
        # - Add tuple including search queries
        df_temp[rag_method[1]] = [
            (pred, em, conf, gens, search_queries_map.get(qid, []))
            for pred, em, conf, qid, gens in zip(
                df_temp["pred_answer"], df_temp["em"], confidences, df_temp["qid"], df_temp["final_answer_list"]
            )
        ]
        df = df_temp[["qid", "query", rag_method[1]]]
        dfs.append(df)
    
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(df, on=["qid", "query"], how="outer")
    merged_df = merged_df.sort_values(by="qid", key=lambda x: x.str.extract(r"(\d+)").squeeze().astype(int)).reset_index(drop=True)
    
    # print(merged_df)
    
    method_cols = [c for c in merged_df.columns if c not in ("qid", "query")]
    candidates = []
    for _, row in merged_df.iterrows():
        qid = row["qid"]
        query = row["query"]
        for m in method_cols:
            val = row[m]
            if pd.isna(val):
                continue
            # val is (pred_answer, em, confidence)
            try:
                pred_answer, em, conf, gens, sqs = val
            except Exception:
                # If it isn't a tuple due to some NaN/object oddity, skip
                continue
            candidates.append({
                "qid": qid,
                "query": query,
                "method": m,
                "pred_answer": pred_answer,
                "em": em,
                "confidence": conf,
                'search_queries': sqs,
                "generations": gens,
            })

    candidates_df = pd.DataFrame(candidates)
    if candidates_df.empty:
        raise ValueError("No candidates found. Check that your merged_df has (pred_answer, em, confidence) tuples.")
    # print(candidates_df)
    
    candidates_df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    print(f"Saved preference dataset to {output_path}")
    
    return candidates_df


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
    parser.add_argument('--prompt_format', type=str, default='x_o_c', choices=['o_c', 'x_o_c', 'p_o_c', 'x_p_o_c', 'x_p_o'])
    
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
    
    ### === Run Steps =============
    set_seed(args.seed)
    create_training_data(args) 
    # create_inference_data(args)
    
    # python rag_selection_application/reward_modeling/data_creation.py
    # accelerate launch --multi_gpu rag_selection_application/reward_modeling/data_creation.py














# === V3 ===
# EPS = 0.2  # tune
# for i in range(len(samples)):
#     for j in range(i+1, len(samples)):
#         a, b = samples[i], samples[j]
#         if abs(a["conf"] - b["conf"]) <= EPS and (a["em"] != b["em"]):
#             # positive is the one with higher em/score
#             pos, neg = (a, b) if a["em"] > b["em"] else (b, a)
#             records.append({
#                 "qid": qid,
#                 "pid": pid,
#                 "query": query,
#                 "pair_type": "score_gap",
#                 "positive_sample": pack(pos),
#                 "negative_sample": pack(neg)
#             })
#             pid += 1



# df_temp[rag_method[1]] = list(zip(df_temp["pred_answer"], df_temp["em"], confidences))
# df_main = df_temp[["qid", "query", rag_method[1]]]

# path_file = f"run_output/{args.run}/{rag_method[0]}/{args.dataset}_train/{rag_method[1]}_{args.retriever_name}/inference_results.jsonl"
# if os.path.exists(path_file):
#     with open(path_file, "r") as f:
#         corr_data = [json.loads(line) for line in f]
#     df_corr = pd.DataFrame(corr_data)[["qid", "path"]]  # change 'some_column' to your target column

#     df_corr[f"search_queries_{rag_method[1]}"] = df_corr["path"].apply(extract_search_queries)
#     df_corr = df_corr[["qid", f"search_queries_{rag_method[1]}"]]

#     # Merge with main
#     df_main = df_main.merge(df_corr, on="qid", how="left")

# dfs.append(df_main)