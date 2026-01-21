#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import argparse

from utils.general_utils import set_seed
from archive.rag_selector_things.training import training
from run_rag_selector.inference import inference
from run_rag_selector.data_preparation import data_preparation
from run_rag_selector.selector_methods.random_selector import get_random_selector
from run_rag_selector.selector_methods.ideal_selector import get_ideal_selector
from run_rag_selector.selector_methods.llm_blender import get_llm_blender
from run_rag_selector.selector_methods.rag_ensemble import get_rag_ensemble
from run_rag_selector.selector_methods.confidence_based_wo_training_selector import wo_training_selector
from run_rag_selector.src.significant_tests import t_test_binary, wilcoxon_rank_sum_binary


def main(args):
    print("\n== RAG Selector ...")
    print(f"""
        Ensemble M.:    {args.ensemble_method}
        W. Clustering:  {args.with_clustering}
        W. Confidence:  {args.with_confidence}
        Dataset & Pro.: {args.dataset} / {args.subsec} ({args.prompt_format})
        Consis. Meth:   {args.consistency_method}
        -----
        Selector M.:    {args.selector_model_name_or_path}
        Confidence Inj: {args.confidence_score_injection}
        Training M.:    {args.training_method}
        Enc. Frozen:    {args.is_encoder_frozen}
        -----
        Random Seed:    {args.seed}
        Run train:      {args.run_train}
        Run test:       {args.run_test}
    """.replace('        ', ''))
    
    dataset = data_preparation(args, only_test=True)
    if args.ensemble_method == 'random':
        get_random_selector(args, dataset)
    elif args.ensemble_method == 'rag_ensemble':
        get_rag_ensemble(args, dataset)
    elif args.ensemble_method == 'ideal':
        get_ideal_selector(args, dataset)
    elif args.ensemble_method == 'llm_blender':
        get_llm_blender(args, dataset)
    elif args.ensemble_method == 'confidence_based_wo_training':
        wo_training_selector(args, dataset)
    
    elif args.ensemble_method == 'confidence_based_w_training':
        dataset = data_preparation(args)
        training(args, dataset)
        inference(args, dataset)
    

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

    file1 = "run_output/run_3 (rag_methods_500)/rag_selector/popqa_test_rag_consistency/confidence_based_wo_training_w_clustering_w_confidence_results.jsonl"
    file2 = "run_output/run_3 (rag_methods_500)/rag_selector/popqa_test_rag_consistency/rag_ensemble_w_clustering_w_confidence_results.jsonl"

    # Read baseline
    baseline_qids, baseline_correctness = [], []
    with open(file1, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                baseline_qids.append(data["qid"])
                baseline_correctness.append(data["em"])

    # Read selector into a dictionary for lookup
    selector_dict = {}
    with open(file2, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                selector_dict[data["qid"]] = data["em"]

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
    parser.add_argument('--selector_model_name_or_path', type=str, default='answerdotai/ModernBERT-large', choices=[
        'answerdotai/ModernBERT-large', 'BAAI/bge-large-en-v1.5', 'google/embeddinggemma-300m',
        'Qwen/Qwen3-Embedding-0.6B', 'Qwen/Qwen3-Embedding-4B'
    ])
    parser.add_argument('--secondary_model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--cache_dir', type=str, default='./cache/')
    parser.add_argument("--data_cache_dir", type=str, default="./run_rag_selector/datasets")
    parser.add_argument("--max_input_tokens", type=int, default=1024)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--dataset', type=str, default='popqa', choices=[
        'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--retriever_name', type=str, default='rerank_l6', choices=[
        'bm25', 'contriever', 'rerank_l6', 'rerank_l12', 'e5', 'bge', 'reasonir'
    ])
    parser.add_argument('--consistency_method', type=str, default='reasoning_consistency', choices=[
        'self_consistency', 'reasoning_consistency', 'rag_consistency'
    ])
    # 
    parser.add_argument('--ensemble_method', default='confidence_based_wo_training', choices=[
        'random', 'llm_blender', 'rag_ensemble', 'ideal', 
        'confidence_based_wo_training', 'confidence_based_w_training'
    ])
    parser.add_argument('--with_clustering', action='store_false')
    parser.add_argument('--with_confidence', action='store_true')
    # 
    # Training Dataset Creation
    parser.add_argument('--is_encoder_frozen', action='store_true')
    parser.add_argument('--num_train_epochs', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default="2e-5")
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=4)
    parser.add_argument('--confidence_score_injection', type=str, default='in_input', choices=['in_input', 'in_representation'])
    parser.add_argument('--training_method', type=str, default='pairwise', choices=['pairwise', 'listwise'])
    parser.add_argument('--add_cross_queries', action='store_false')
    parser.add_argument('--cross_samples', type=int, default=2000)
    parser.add_argument('--near_ratio', type=float, default=0.8, help="For condition B")
    parser.add_argument('--min_gap', type=float, default=0.5, help="")
    parser.add_argument('--prompt_format', type=str, default='x_o_c_g_dc', choices=[
        'x_o', 'x_o_sq', 'x_o_th', 'x_o_dc', 'x_o_g', 'x_o_g_sq', 'x_o_g_dc', 'x_o_sq_dc', 'x_o_sq_th_dc',
        'x_o_c', 'x_o_c_sq', 'x_o_c_th', 'x_o_c_dc', 'x_o_c_g', 'x_o_c_g_sq', 'x_o_c_g_dc', 'x_o_c_sq_dc', 'x_o_c_sq_th_dc',
    ])
    parser.add_argument('--n_docs_prompt', type=int, default=2)
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
    args.semantic_equivalence_prompt_file = "run_rag_methods/prompts/semantic_equivalence_prompt_template.txt"
     
    clustering_text = 'w_clustering' if args.with_clustering  else 'wo_clustering'
    confidence_text = 'w_confidence' if args.with_clustering  else 'wo_confidence'
    results_dir = f"run_output/{args.run_test}/rag_selector/{args.dataset}_{args.subsec}_{args.consistency_method}"
    os.makedirs(results_dir, exist_ok=True)
    
    if args.ensemble_method == 'confidence_based_w_training':
        model_ = args.selector_model_name_or_path.split('/')[-1]
        args.saved_model_name_or_path = f"run_rag_selector/models/{args.dataset}_{args.consistency_method}/{args.training_method}_confidence_{args.confidence_score_injection}_{args.prompt_format}/{model_}"
        args.save_results_path = f"{results_dir}/{args.training_method}_confidence_{args.confidence_score_injection}_{clustering_text}_{confidence_text}_{args.prompt_format}_results.jsonl"
    else:
        args.save_results_path = f"{results_dir}/{args.ensemble_method}_{clustering_text}_{confidence_text}_results.jsonl"
    
    args.data_cache_dir = f"{args.data_cache_dir}/{args.run_test}"
    os.makedirs(args.data_cache_dir, exist_ok=True)
    
    set_seed(args.seed)
    main(args)
    # stat_testing(args)
    
    
    # python run_rag_selector/run_framework.py
    # accelerate launch --multi_gpu run_rag_selector/run_framework.py

