#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import json
import torch
import random
import argparse
import transformers
import pandas as pd
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
import ast

from utils.general_utils import set_seed
from run_mcts_two_actions.src.models.semantic_equivalence import SemanticEquivalenceGenerator


def ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    # handle stringified lists
    try:
        y = ast.literal_eval(x)
        return y if isinstance(y, list) else [x]
    except Exception:
        return [x]

def extract_generated_text_list(obj):
    """
    Accepts either:
      - list[str]
      - list[dict] with key 'generated_text'
      - mixed / stringified list
    Returns list[str]
    """
    lst = ensure_list(obj)
    out = []
    for item in lst:
        if isinstance(item, dict) and "generated_text" in item:
            out.append(item["generated_text"])
        else:
            out.append(str(item))
    return out

def mv_bm_rag_selection(args):
    accelerator = Accelerator()
    device = accelerator.device
    model = transformers.AutoModelForCausalLM.from_pretrained(args.secondary_model_name_or_path, torch_dtype=torch.bfloat16).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.secondary_model_name_or_path)
    se_model = SemanticEquivalenceGenerator(args, device, model, tokenizer)
    
    output_path = f"run_output/{args.run}/rag_selection_reward_modeling/{args.dataset}_{args.retriever_name}_{args.consistency_method}/conf_with_others.jsonl"
    
    # rag_methods = [
    #     ('Qwen2.5-7B-Instruct', 'self_ask'),
    #     ('Qwen2.5-7B-Instruct', 'react'),
    #     ('Qwen2.5-7B-Instruct', 'search_o1'),
    #     ('ReSearch-Qwen-7B-Instruct', 'research'),
    #     ('SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo', 'search_r1')
    # ]
    
    # # === Load data
    # dfs = []
    # for rag_method in rag_methods:
    #     file_path = f"run_output/{args.run}/{rag_method[0]}/{args.dataset}_{args.subsec}/{rag_method[1]}_{args.retriever_name}/{args.consistency_method}_results.jsonl"
    #     with open(file_path, "r") as f:
    #         data = [json.loads(line) for line in f]
    #     df_temp = pd.DataFrame(data)[["qid", "query", "pred_answer", "em", "final_answer_list", "ue_scores"]]
    #     confidences = df_temp["ue_scores"].apply(lambda x: x["majority_voting"]["confidence"])
        
        
    #     df_temp[rag_method[1]] = list(zip(df_temp["pred_answer"], df_temp["em"], confidences, df_temp["final_answer_list"]))
    #     df = df_temp[["qid", "query", rag_method[1]]]
    #     dfs.append(df)
    
    # merged_df = dfs[0]
    # for df in dfs[1:]:
    #     merged_df = merged_df.merge(df, on=["qid", "query"], how="outer")
    # merged_df = merged_df.sort_values(by="qid", key=lambda x: x.str.extract(r"(\d+)").squeeze().astype(int)).reset_index(drop=True)
    # print(merged_df)
    
    # method_cols = [c for c in merged_df.columns if c not in ["qid", "query"]]
    # for row_idx, row in tqdm(merged_df.iterrows()):
    #     question = row['query']
    #     # if row_idx == 25:
    #     #     break
    #     for col_idx, col in enumerate(method_cols):
    #         # if col_idx == 1:
    #         #     break
    #         others_conf = []
    #         prediction, em, conf, my_final_list = row[col]
    #         for other in method_cols:
    #             if other == col:
    #                 continue
    #             other_generated_texts = extract_generated_text_list(merged_df.at[row_idx, other][3])
    #             len_other_generated_texts = len(other_generated_texts)
    #             num_consistent = sum(se_model.check_answers_equiv(question, prediction, ans) for ans in other_generated_texts)
    #             oth_conf = num_consistent / len_other_generated_texts
    #             others_conf.append(oth_conf)
            
    #         merged_df.at[row_idx, col] = (prediction, em, conf, my_final_list, others_conf)
    
    # merged_df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    # print(f"Saved preference dataset to {output_path}")
    

    merged_df = pd.read_json(output_path, orient="records", lines=True)
    method_cols = [c for c in merged_df.columns if c not in ["qid", "query"]]
    W1, W2 = 0.3, 0.7
    
    def get_final_score(candidate):
        return W1 * candidate[0][2] + W2 * float(np.mean(candidate[0][4]))
    
    def select_best_answer(row):
        candidates = [(row[col], col) for col in method_cols ] # if pd.notnull(row[col])
        if not candidates:
            return pd.Series([None, None, None, None])
        
        max_conf = max(get_final_score(cand) for cand in candidates)
        top_candidates = [cand for cand in candidates if get_final_score(cand) == max_conf]
        selected = random.choice(top_candidates)
        ans, em, conf = selected[0][0], selected[0][1], selected[0][2]
        method = selected[1]
        return pd.Series([ans, em, method, conf])
    
    merged_df[["best_answer", "best_em", "best_method", "best_confidence"]] = merged_df.apply(select_best_answer, axis=1)
    accuracy = merged_df["best_em"].mean()
    print(f"Dataset Accuracy (based on EM): {accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--secondary_model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--max_new_tokens', type=int, default=1024)
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
        'self_consistency', 'reasoning_consistency', 'rag_consistency'
    ])
    parser.add_argument("--n_generations", type=int, default=10)
    parser.add_argument("--mask_left_boundary", type=float, default=0.1)
    parser.add_argument("--mask_right_boundary", type=float, default=0.4)
    parser.add_argument("--consistency_temperature", type=float, default=1.0)
    
    # Others
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--run', type=str, default='run_3 (rag_methods_500)')
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument('--use_counter', action='store_false')
    
    args = parser.parse_args()
    
    args.semantic_equivalence_prompt_file = "run_mcts_two_actions/prompts/semantic_equivalence_prompt_template.txt"
    
    ### === Run Steps =============
    set_seed(args.seed)
    mv_bm_rag_selection(args)
    
    # python rag_selection_application/mv_between_models/mv_bm.py
    # accelerate launch --multi_gpu rag_selection_application/mv_between_models/mv_bm.py