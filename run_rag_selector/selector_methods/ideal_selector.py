#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import json
import pandas as pd


def get_ideal_selector(args):
    rag_methods = [
        ('Qwen2.5-7B-Instruct', 'self_ask'),
        ('Qwen2.5-7B-Instruct', 'react'),
        ('Qwen2.5-7B-Instruct', 'search_o1'),
        ('ReSearch-Qwen-7B-Instruct', 'research'),
        ('SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo', 'search_r1')
    ]
    
    dfs = []
    for rag_method in rag_methods:
        file_path = f"run_output/{args.run_test}/{rag_method[0]}/{args.dataset}_{args.subsec}/{rag_method[1]}_{args.retriever_name}/inference_results.jsonl"
        with open(file_path, "r") as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)[["qid", "em"]].rename(columns={"em": rag_method[1]})
        dfs.append(df)
    
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(df, on="qid", how="outer")
    
    df_numeric = merged_df.fillna(0).set_index("qid").astype(int)
    oracle_correct = (df_numeric.max(axis=1) > 0).astype(int)
    oracle_accuracy = oracle_correct.mean()
    print(f"Oracle upper bound accuracy: {oracle_accuracy:.4f}")
    
    
    # Plot UpSet
    # df_filled = merged_df.fillna(0)         # fill missing queries
    # df_filled = df_filled.set_index("qid")  # set query_id as index
    # df_bool = df_filled.astype(bool).reset_index(drop=True)
    # upset_data = from_indicators(data=df_bool, indicators=df_bool.columns)
    # up = UpSet(upset_data, show_counts=True, show_percentages=True)
    # up.plot()
    # plt.suptitle("Overlap of Correct Answers Across RAG Methods")
    # plt.savefig("rag_overlap_upsetplot.png", dpi=300, bbox_inches="tight")
    # plt.show()
