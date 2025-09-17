#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gc
import json
import torch
import argparse

from utils.general_utils import set_seed
from run_rag_selector.training import training
from run_rag_selector.inference import inference
from run_rag_selector.data_preparation import data_preparation
from run_rag_selector.src.ideal_selector import get_ideal_selector
from run_rag_selector.src.wo_training_selector import wo_training_selector
from run_rag_selector.src.significant_tests import t_test_binary, wilcoxon_rank_sum_binary

def main(args):
    print("\n== RAG Selector ...")
    print(f"""
        Selector Model:   {args.selector_model_name_or_path}
        Dataset:          {args.dataset} / {args.subsec} ({args.prompt_format})
        Consistency Meth: {args.consistency_method}
        Get ideal:        {args.get_ideal}
        With training:    {args.with_training}
        With clustering:  {args.with_clustering}
        Conf Score Inj:   {args.confidence_score_injection}
        Training method:  {args.training_method}
        Is Encoder Frozen {args.is_encoder_frozen}
        Seed:             {args.seed}
        Run train:        {args.run_train}
        Run test:         {args.run_test}
    """.replace('        ', ''))
    
    if args.get_ideal:
        get_ideal_selector(args)
    else:
        dataset = data_preparation(args)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
         
        if args.with_training:
            training(args, dataset)
            inference(args, dataset)
        else:
            wo_training_selector(args, dataset)

def stat_testing(args):
    baseline_rag_methods = [
        ('Qwen2.5-7B-Instruct', 'self_ask'),
        ('Qwen2.5-7B-Instruct', 'react'),
        ('Qwen2.5-7B-Instruct', 'search_o1'),
        ('ReSearch-Qwen-7B-Instruct', 'research'),
        ('SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo', 'search_r1')
    ]
    baseline_rag_method = baseline_rag_methods[4]
    baseline_file_path = f"run_output/{args.run_test}/{baseline_rag_method[0]}/{args.dataset}_{args.subsec}/{baseline_rag_method[1]}_{args.retriever_name}/{args.consistency_method}_results.jsonl"

    # Read baseline
    baseline_qids, baseline_correctness = [], []
    with open(baseline_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                baseline_qids.append(data["qid"])
                baseline_correctness.append(data["em"])

    # Read selector into a dictionary for lookup
    selector_dict = {}
    with open(args.save_results_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                selector_dict[data["qid"]] = data["label"]

    # Match selector values to baseline order
    selector_correctness = []
    for qid in baseline_qids:
        if qid not in selector_dict:
            sys.exit(f"Error: qid {qid} from baseline not found in selector results.")
        selector_correctness.append(selector_dict[qid])


    t_test_binary(baseline_correctness, selector_correctness)
    wilcoxon_rank_sum_binary(baseline_correctness, selector_correctness)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--selector_model_name_or_path', type=str, default='Qwen/Qwen3-Embedding-0.6B', choices=[
        'answerdotai/ModernBERT-large', 'BAAI/bge-large-en-v1.5', 'google/embeddinggemma-300m',
        'Qwen/Qwen3-Embedding-0.6B', 'Qwen/Qwen3-Embedding-4B'
    ])
    parser.add_argument('--secondary_model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--cache_dir', type=str, default='./cache/')
    parser.add_argument("--data_cache_dir", type=str, default="./run_rag_selector/datasets")
    parser.add_argument("--max_input_tokens", type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--dataset', type=str, default='hotpotqa', choices=[
        'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'
    ])
    parser.add_argument('--subsec', type=str, default='dev', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--prompt_format', type=str, default='x_o_c_g_dc', choices=[
        'x_o', 'x_o_sq', 'x_o_th', 'x_o_dc', 'x_o_g', 'x_o_g_sq', 'x_o_g_dc', 'x_o_sq_dc', 'x_o_sq_th_dc',
        'x_o_c', 'x_o_c_sq', 'x_o_c_th', 'x_o_c_dc', 'x_o_c_g', 'x_o_c_g_sq', 'x_o_c_g_dc', 'x_o_c_sq_dc', 'x_o_c_sq_th_dc',
    ])
    parser.add_argument('--n_docs_prompt', type=int, default=2)
    parser.add_argument('--retriever_name', type=str, default='rerank_l6', choices=[
        'bm25', 'contriever', 'rerank_l6', 'rerank_l12', 'e5', 'bge', 'reasonir'
    ])
    parser.add_argument('--consistency_method', type=str, default='rag_consistency', choices=[
        'self_consistency', 'reasoning_consistency', 'rag_consistency'
    ])
    # 
    parser.add_argument('--get_ideal', action='store_true')
    parser.add_argument('--with_training', action='store_false')
    parser.add_argument('--with_clustering', action='store_false')
    parser.add_argument('--confidence_score_injection', type=str, default='in_representation', choices=['in_input', 'in_representation'])
    parser.add_argument('--training_method', type=str, default='pairwise', choices=['pairwise', 'listwise'])
    # 
    parser.add_argument('--is_encoder_frozen', action='store_true')
    parser.add_argument('--num_train_epochs', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default="2e-5")
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=4)
    # 
    # Dataset Creation
    parser.add_argument('--add_cross_queries', action='store_false')
    parser.add_argument('--cross_samples', type=int, default=2000)
    parser.add_argument('--near_ratio', type=float, default=0.8, help="For condition B")
    parser.add_argument('--min_gap', type=float, default=0.5, help="")
    # 
    parser.add_argument('--run_train', type=str, default='run_1 (rag_methods_2k)')
    parser.add_argument('--run_test', type=str, default='run_3 (rag_methods_500)')
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
    
    model_ = args.selector_model_name_or_path.split('/')[-1]
    args.saved_model_name_or_path = f"run_rag_selector/models/{args.dataset}/{args.training_method}_confidence_{args.confidence_score_injection}_{args.prompt_format}/{model_}"
    
    clustering_text = 'clustering' if args.with_clustering  else 'wo_clustering'
    results_dir = f"run_output/{args.run_test}/rag_selector/{args.dataset}_{args.subsec}_{args.consistency_method}"
    os.makedirs(results_dir, exist_ok=True)
    args.save_results_path = f"{results_dir}/{args.training_method}_confidence_{args.confidence_score_injection}_{clustering_text}_{args.prompt_format}_results.jsonl"
    
    set_seed(args.seed)
    main(args)
    # stat_testing(args)
    
    
    # python run_rag_selector/run_framework.py
    # accelerate launch --multi_gpu run_rag_selector/run_framework.py

