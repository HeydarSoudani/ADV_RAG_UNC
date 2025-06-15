#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import argparse
from tqdm import tqdm
from accelerate import Accelerator

from utils.general_utils import set_seed
from run_rag_methods.src.rag_methods import *
from run_uncertainty_estimation.consistency_methods import *
from run_uncertainty_estimation.uncertainty_estimator import UncertaintyEstimator 

def ue_generation(args):
    # === MultiGPU setup ========================
    accelerator = Accelerator()
    device = accelerator.device
    
    if accelerator.is_main_process:
        print("\n== UE Generation ...")
        print(f"""
            Model name:  {args.generation_model_name_or_path}
            Dataset:     {args.dataset} / {args.subsec} ({args.fraction_of_data_to_use})
            Retriever:   {args.retriever_name} / ({args.retrieval_model_path})
            RAG Method:  {args.rag_method}
            Con. Method: {args.consistency_method}
            UE Method:   {args.ue_method}
            Run:         {args.run}
            Seed:        {args.seed}
        """.replace('        ', ''))
        
        # --- Define CUDA device
        if torch.cuda.is_available():
            print(f"Number of available GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA is not available. No GPUs detected.")
        print('\n')

    # === Read input data =======================
    query_ids, rag_generations = [], {}
    if os.path.exists(args.inference_results_file):
        with open(args.inference_results_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'qid' in data:
                    query_ids.append(data['qid'])
                    rag_generations[data['qid']] = data
    sorted_query_ids = sorted(query_ids, key=lambda x: int(x.split('_')[1]))
    
    # === Read existing (generated) samples ======
    generated_qids = []
    if os.path.exists(args.consistency_results_file):
        with open(args.consistency_results_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'qid' in data:
                    generated_qids.append(data['qid'])
    generated_qids = set(generated_qids)
    filtered_sorted_query_ids = [id_ for id_ in sorted_query_ids if id_ not in generated_qids]
    
    # === Read Models ============================
    if args.rag_method == "fix_sentence_retrieval":
        rag_model = FixSentenceRAG(args, device)
    elif args.rag_method == "fix_length_retrieval":
        rag_model = FixLengthRAG(args, device)
    elif args.rag_method == 'ircot':
        rag_model = IRCOT_RAG(args, device)
    elif args.rag_method == 'flare':
        rag_model = FLARE_RAG_V1(args, device)
    elif args.rag_method == 'dragin':
        rag_model = DRAGIN_RAG(args, device)
    elif args.rag_method == 'self_ask':
        rag_model = SelfAsk_RAG(args, device)
    elif args.rag_method == 'react':
        rag_model = ReAct_RAG(args, device)
    elif args.rag_method == 'search_o1':
        rag_model = SearchO1_RAG(args, device)
    elif args.rag_method == 'search_r1':
        rag_model = SearchR1_RAG(args, device)
    else:
        raise NotImplementedError
    
    if args.consistency_method == 'self_consistency':
        consistency_model = SelfConsistency(device, args, rag_model)
    elif args.consistency_method == 'reasoning_consistency':
        consistency_model = ReasoningConsistency(device, args, rag_model)
    elif args.consistency_method == 'rag_consistency':
        consistency_model = RagConsistency(device, args, rag_model)
    else:
        raise NotImplementedError
    
    uncertainty_estimator_model = UncertaintyEstimator()
    
    # === Main Loop ==============================
    accelerator.wait_for_everyone()
    with accelerator.split_between_processes(filtered_sorted_query_ids) as sorted_query_ids_shard:
        
        consistency_results_file_ranked = (
            f"{args.output_dir}/consistency_results_th{args.hallucination_threshold}_rank{accelerator.process_index}.jsonl"
            if args.rag_method in ['flare', 'dragin'] 
            else f"{args.output_dir}/consistency_results_rank{accelerator.process_index}.jsonl"
        )
        cons_f = open(consistency_results_file_ranked, 'w', encoding='utf-8')

        trace_f = None
        write_traces = args.consistency_method == 'rag_consistency'
        if write_traces:
            masked_traces_results_file_ranked = (
                f"{args.output_dir}/masked_traces_results_th{args.hallucination_threshold}_rank{accelerator.process_index}.jsonl"
                if args.rag_method in ['flare', 'dragin'] 
                else f"{args.output_dir}/masked_traces_results_rank{accelerator.process_index}.jsonl"
            )
            trace_f = open(masked_traces_results_file_ranked, 'w', encoding='utf-8')
    
        try:
            for i, qid in enumerate(tqdm(sorted_query_ids_shard, desc=f"[Rank {accelerator.process_index}]")):
                if i == 1:
                    break
                
                sample = rag_generations[qid]
                user_query, prediction, trace = sample['query'], sample['pred_answer'], sample['path']
                
                # 1) Generate output list
                masked_traces, final_answer_list = consistency_model.get_masked_traces(qid, user_query, trace)
                masked_traces_text = [
                    rag_model.get_input_prompt(masked_trace) for masked_trace in masked_traces
                ]
                
                # 2) Calculate UE scores
                ue_scores = uncertainty_estimator_model.estimate(
                    user_query,
                    input_prompts_text = masked_traces_text,
                    output_texts = final_answer_list
                )
                
                # 3) Print in output files 
                cons_item = {
                    "qid": qid,
                    "query": user_query,
                    "gt_answers": sample['gt_answers'],
                    "pred_answer": prediction,
                    "em": sample['em'],
                    "final_answer_list": final_answer_list,
                    "ue_scores": ue_scores
                }
                cons_f.write(json.dumps(cons_item) + "\n")
                
                if trace_f:
                    new_masked_traces = [
                        [
                            {
                                "think": step["think"],
                                "search_query": step["search_query"],
                                "docs": [{"id": doc["id"]} for doc in step["docs"]]
                            } if "docs" in step else step
                            for step in masked_trace
                        ]
                        for masked_trace in masked_traces
                    ]
                    trace_item = {
                        "qid": qid,
                        "query": user_query,
                        "masked_traces": new_masked_traces
                    }
                    trace_f.write(json.dumps(trace_item) + '\n')

        finally:
            cons_f.close()
            if trace_f:
                trace_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--generation_model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--paraphrase_model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--max_new_token', type=int, default=1024)
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='bamboogle', choices=[
        'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test', 'validation'])
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
    
    # RAG methods (input)
    parser.add_argument('--rag_method', type=str, default='search_r1', choices=[
        'fix_sentence_retrieval', 'fix_length_retrieval', 'ircot', 'flare', 'dragin',
        'react', 'self_ask', 'search_o1', 'search_r1',
        'RASPberry'
    ])
    
    # Consistency Generation Methods (answer list) ---
    parser.add_argument('--consistency_method', type=str, default='rag_consistency', choices=[
        'self_consistency', 'reasoning_consistency', 'rag_consistency'
    ])
    parser.add_argument("--n_generations", type=int, default=10)
    parser.add_argument("--cutoff_rollout", type=int, default=-1)
    parser.add_argument("--start_idx", type=int, default=-1)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--mask_left_boundary", type=float, default=0.2)
    parser.add_argument("--mask_right_boundary", type=float, default=0.5)
    parser.add_argument("--rc_mode", type=str, default="mid", choices=["loose", "mid", "strict", "maj"])
    parser.add_argument("--rc_temperature", type=float, default=1.0)
    parser.add_argument("--rc_n_completions", type=int, default=1)
    parser.add_argument("--rc_criteria", type=str, default="freq", choices=["freq", "reward"])
    parser.add_argument("--threshold", type=float, default=0.999)
    parser.add_argument("--extend_rc_mode", type=str, default="majority_vote", choices=["original", "BoN", "majority_vote"])
    parser.add_argument("--best_of", type=int, default=5)
    
    parser.add_argument('--ue_method', type=str, default='ptrue', choices=[
        'ptrue', 'predictive_entropy', 'semantic_entropy', 
        'num_semantic_set', 'eccentricity', 'matrix_degree', 'sum_eigen'
    ])
    
    # Others
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--run', type=str, default='run_2 (mcts_2k_rollout4)')
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument('--use_counter', action='store_false')
    
    args = parser.parse_args()
    
    # === Files ====================
    model_ = args.model_name_or_path.split('/')[-1]
    args.output_dir = f"run_output/{args.run}/{model_}/{args.dataset}_{args.subsec}/{args.rag_method}_{args.retriever_name}"
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.rag_method in ['flare', 'dragin']:
        args.inference_results_file = f"{args.output_dir}/inference_results_th{args.hallucination_threshold}.jsonl"
        args.consistency_results_file = f"{args.output_dir}/{args.consistency_method}_results_th{args.hallucination_threshold}.jsonl"
        if args.consistency_method == "rag_consistency":
            args.masked_traces_results_file = f"{args.output_dir}/{args.consistency_method}_masked_traces_th{args.hallucination_threshold}.jsonl"
    else:
        args.inference_results_file = f"{args.output_dir}/inference_results.jsonl"
        args.consistency_results_file = f"{args.output_dir}/{args.consistency_method}_results.jsonl"
        if args.consistency_method == "rag_consistency":
            args.masked_traces_results_file = f"{args.output_dir}/{args.consistency_method}_masked_traces.jsonl"
        
    
    ### === Run Steps =============
    set_seed(args.seed)
    ue_generation(args)
    
    
    # python run_uncertainty_estimation/run_framework.py
    # accelerate launch --multi_gpu run_uncertainty_estimation/run_framework.py
