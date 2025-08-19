#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import json
import random
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from upsetplot import UpSet, from_indicators

from utils.general_utils import set_seed


def correctness_distribution_analyze(args):
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
    
    # === Load data
    dfs = []
    for rag_method in rag_methods:
        if rag_method[1] in ['flare', 'dragin']:
            file_path = f"run_output/{args.run}/{rag_method[0]}/{args.dataset}_{args.subsec}/{rag_method[1]}_{args.retriever_name}/inference_results_th{rag_method[2]}.jsonl"
        else:
            file_path = f"run_output/{args.run}/{rag_method[0]}/{args.dataset}_{args.subsec}/{rag_method[1]}_{args.retriever_name}/inference_results.jsonl"
        with open(file_path, "r") as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)[["qid", "em"]].rename(columns={"em": rag_method[1]})
        dfs.append(df)
    
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(df, on="qid", how="outer")
    
    df_filled = merged_df.fillna(0)         # fill missing queries
    df_filled = df_filled.set_index("qid")  # set query_id as index
    df_bool = df_filled.astype(bool).reset_index(drop=True)
    
    print(df_bool)
    print(df_bool.columns)
    
    upset_data = from_indicators(data=df_bool, indicators=df_bool.columns)
    
    #
    df_numeric = merged_df.fillna(0).set_index("qid").astype(int)
    oracle_correct = (df_numeric.max(axis=1) > 0).astype(int)
    oracle_accuracy = oracle_correct.mean()
    print(f"Oracle upper bound accuracy: {oracle_accuracy:.4f}")
    
    # Plot UpSet
    up = UpSet(upset_data, show_counts=True, show_percentages=True)
    up.plot()
    plt.suptitle("Overlap of Correct Answers Across RAG Methods")
    plt.savefig("rag_overlap_upsetplot.png", dpi=300, bbox_inches="tight")

    plt.show()

def rag_selection(args):
    rag_methods = [
        ('Qwen2.5-7B-Instruct', 'self_ask'),
        ('Qwen2.5-7B-Instruct', 'react'),
        ('Qwen2.5-7B-Instruct', 'search_o1'),
        ('ReSearch-Qwen-7B-Instruct', 'research'),
        ('SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo', 'search_r1')
    ]
    
    # === Load data
    dfs = []
    for rag_method in rag_methods:
        file_path = f"run_output/{args.run}/{rag_method[0]}/{args.dataset}_{args.subsec}/{rag_method[1]}_{args.retriever_name}/{args.consistency_method}_results.jsonl"
        with open(file_path, "r") as f:
            data = [json.loads(line) for line in f]
        df_temp = pd.DataFrame(data)[["qid", "pred_answer", "em", "ue_scores"]]
        confidences = df_temp["ue_scores"].apply(lambda x: x["majority_voting"]["confidence"])
        df_temp[rag_method[1]] = list(zip(df_temp["pred_answer"], df_temp["em"], confidences))
        df = df_temp[["qid", rag_method[1]]]
        dfs.append(df)
    
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(df, on="qid", how="outer")
    merged_df = merged_df.sort_values(by="qid", key=lambda x: x.str.extract(r"(\d+)").squeeze().astype(int)).reset_index(drop=True)
    
    
    rag_columns = [rm[1] for rm in rag_methods]
    # weights_list = [25.6, 33.6, 36.0, 39.2, 44.0] # bamboogle
    # weights_list = [35.60, 36.80, 33.20, 38.60, 41.60] # popqa
    # weights_list = [33.00, 27.80, 29.00, 38.80, 41.40] # hotpotqa
    # rag_weights = {col: w for col, w in zip(rag_columns, weights_list)}
    # print(rag_weights)
    
    def select_best_answer(row):
        candidates = [(row[col], col) for col in rag_columns if pd.notnull(row[col])]
        if not candidates:
            return pd.Series([None, None, None, None])
        
        max_conf = max(cand[0][2] for cand in candidates)
        top_candidates = [cand for cand in candidates if cand[0][2] == max_conf]
        
        # Without weight
        selected = random.choice(top_candidates)
        
        # with weight
        # top_weights = [rag_weights[cand[1]] for cand in top_candidates]
        # selected = random.choices(top_candidates, weights=top_weights, k=1)[0]
        
        ans, em, conf = selected[0]  # unpack the tuple (pred_answer, em, confidence)
        method = selected[1]
        return pd.Series([ans, em, method, conf])
        
    merged_df[["best_answer", "best_em", "best_method", "best_confidence"]] = merged_df.apply(select_best_answer, axis=1)

    accuracy = merged_df["best_em"].mean()
    print(f"Dataset Accuracy (based on EM): {accuracy:.4f}")

def rubostness_analysis(args):
    
    def load_scores(path):
        data = {}
        with open(path, "r") as f:
            for line in f:
                obj = json.loads(line)
                qid = obj["qid"]
                em = float(obj["em"])
                data[qid] = em
        return data
    
    # naive_selector_result_file = "run_output/run_3 (rag_methods_500)/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo/hotpotqa_dev/search_r1_rerank_l6/inference_results.jsonl"
    # our_selector_result_file = "run_output/run_3 (rag_methods_500)/rag_selection_reward_modeling/hotpotqa_rerank_l6_rag_consistency/dev_inference_results_x_o_c_best_base.jsonl"  
    
    naive_selector_result_file = "run_output/run_3 (rag_methods_500)/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo/popqa_test/search_r1_rerank_l6/inference_results.jsonl"
    our_selector_result_file = "run_output/run_3 (rag_methods_500)/rag_selection_reward_modeling/popqa_rerank_l6_rag_consistency/test_inference_results_x_o_c.jsonl"
    
    scores1 = load_scores(naive_selector_result_file)
    scores2 = load_scores(our_selector_result_file)
    
    wins = losses = ties = 0
    missing = 0

    for qid, em1 in scores1.items():
        if qid not in scores2:
            missing += 1
            continue

        em2 = scores2[qid]
        if em1 > em2:
            wins += 1
        elif em1 < em2:
            losses += 1
        else:
            ties += 1
            
    total = wins + losses + ties

    print("Results comparing file1 vs file2:")
    print(f"Wins   (file1 > file2): {wins} ({wins/total:.2%})")
    print(f"Losses (file1 < file2): {losses} ({losses/total:.2%})")
    print(f"Ties   (file1 = file2): {ties} ({ties/total:.2%})")
    print(f"Missing qids in file2 : {missing}")
      

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
    parser.add_argument('--consistency_method', type=str, default='self_consistency', choices=[
        'self_consistency', 'reasoning_consistency', 'rag_consistency'
    ])
    parser.add_argument("--n_generations", type=int, default=10)
    parser.add_argument("--mask_left_boundary", type=float, default=0.1)
    parser.add_argument("--mask_right_boundary", type=float, default=0.4)
    parser.add_argument("--consistency_temperature", type=float, default=1.0)
    
    # Others
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--run', type=str, default='run_4 (rag_methods_500)')
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument('--use_counter', action='store_false')
    
    args = parser.parse_args()
    
    
    ### === Run Steps =============
    set_seed(args.seed)
    # correctness_distribution_analyze(args)
    # rag_selection(args)
    rubostness_analysis(args)
    
    # python applications/rag_selector/wo_training.py
    # accelerate launch --multi_gpu applications/rag_selector/wo_training.py