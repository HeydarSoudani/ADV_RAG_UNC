#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import glob
import json
import torch
import random
import argparse
import transformers
from tqdm import tqdm
from accelerate import Accelerator
from typing import Iterable, Dict, Any, Tuple
import numpy as np


from utils.general_utils import set_seed
from applications.selective_generation.selective_eval import *
from applications.selective_generation.plotting_utils import *

# Src -> Controlling Risk of Retrieval-augmented Generation: A Counterfactual Prompting Framework
def compute_rc_rag_metrics(correctness, confidence, threshold):
    """
    correctness: list of bools (True=answerable, False=unanswerable)
    confidence: list of floats
    threshold: float threshold for keeping vs discarding
    """
    assert len(correctness) == len(confidence), "Mismatched input lengths"

    # Counts
    N_AK = N_AD = N_UK = N_UD = 0

    for corr, conf in zip(correctness, confidence):
        keep = conf >= threshold
        if corr:  # Answerable
            if keep:
                N_AK += 1
            else:
                N_AD += 1
        else:  # Unanswerable
            if keep:
                N_UK += 1
            else:
                N_UD += 1

    N = N_AK + N_AD + N_UK + N_UD

    # Metrics
    risk = N_UK / (N_AK + N_UK) if (N_AK + N_UK) > 0 else 0.0
    carefulness = N_UD / (N_UK + N_UD) if (N_UK + N_UD) > 0 else 0.0
    coverage = (N_AK + N_UK) / N if N > 0 else 0.0
    alignment = (N_AK + N_UD) / N if N > 0 else 0.0

    return {
        "Risk": risk,
        "Carefulness": carefulness,
        "Coverage": coverage,
        "Alignment": alignment,
        "Counts": {"AK": N_AK, "AD": N_AD, "UK": N_UK, "UD": N_UD}
    }

# Src: Don’t Hallucinate, Abstain: Identifying LLM Knowledge Gaps via Multi-LLM Collaboration
def abstention_metrics(
    correctness: Iterable[int | bool],
    scores: Iterable[float],
    threshold: float,
    answer_if_above: bool = True,
) -> Dict[str, Any]:
    """
    Compute A, B, C, D and the four metrics:
      - Reliable Accuracy (R-Acc) = A / (A + C)
      - Effective Reliability (ER) = (A - C) / N
      - Abstain Accuracy (A-Acc) = (A + D) / N
      - Abstain F1 (A-F1) with precision = D / (B + D), recall = D / (C + D)

    Args:
        correctness: iterable of 0/1 or False/True for each example's generated answer.
        scores: iterable of consistency/confidence scores aligned with `correctness`.
        threshold: score threshold for answering vs abstaining.
        answer_if_above: if True, answer when score >= threshold; else answer when score <= threshold.

    Returns:
        dict with counts {"A","B","C","D"} and metrics {"R-Acc","ER","A-Acc","A-Precision","A-Recall","A-F1"}.
    """
    y = np.asarray(correctness, dtype=bool)
    s = np.asarray(scores, dtype=float)
    if y.shape != s.shape:
        raise ValueError("correctness and scores must have the same length")

    answered = s >= threshold if answer_if_above else s <= threshold
    abstained = ~answered

    # Confusion-style counts
    A = int(np.sum(answered & y))          # answered & correct
    C = int(np.sum(answered & ~y))         # answered & incorrect
    B = int(np.sum(abstained & y))         # abstained & (would have been) correct
    D = int(np.sum(abstained & ~y))        # abstained & (would have been) incorrect
    N = A + B + C + D

    # Safe divisions
    def safe_div(num: float, den: float) -> float | float:
        return float(num / den) if den != 0 else float("nan")

    r_acc = safe_div(A, A + C)
    er = safe_div(A - C, N)
    a_acc = safe_div(A + D, N)

    a_precision = safe_div(D, B + D)   # precision of "abstain" as a classifier for incorrectness
    a_recall = safe_div(D, C + D)      # recall of "abstain" on incorrect cases
    a_f1 = safe_div(2 * a_precision * a_recall, a_precision + a_recall) if np.isfinite(a_precision + a_recall) else float("nan")

    return {
        "metrics": {
            "R-Acc": r_acc,
            "ER": er,
            "A-Acc": a_acc,
            "A-Precision": a_precision,
            "A-Recall": a_recall,
            "A-F1": a_f1,
        },
        "counts": {"A": A, "B": B, "C": C, "D": D, "N": N},
        "threshold": threshold,
        "answer_if_above": answer_if_above,
    }

def accuracy_rejection_curve(
    correctness: Iterable[int],
    certainty: Iterable[float],
    include_terminal: bool = True,
    terminal_accuracy: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the Accuracy–Rejection curve.

    Args:
        correctness: Iterable of 0/1 values (1 = correct prediction, 0 = incorrect).
        certainty:   Iterable of certainty/confidence scores (higher = more certain).
        include_terminal: If True, append the terminal point at rejection=1.0.
        terminal_accuracy: Accuracy value to use at rejection=1.0 (no kept samples).
                           A common convention is 1.0.

    Returns:
        rejection: np.ndarray of rejection rates in [0, 1].
        accuracy:  np.ndarray of accuracy measured on the kept set (highest certainty)
                   after rejecting a given fraction (rejection rate).
    """
    c = np.asarray(correctness, dtype=float)
    s = np.asarray(certainty, dtype=float)
    assert c.shape == s.shape, "correctness and certainty must have the same length"
    n = c.size
    if n == 0:
        return np.array([0.0]), np.array([np.nan])

    # Sort examples by ascending certainty (least certain first to be rejected first)
    order = np.argsort(s, kind="stable")
    c_sorted = c[order]

    # For each possible rejection k (0..n), keep the n-k most certain examples:
    # accuracy on kept = sum(correct[k:]) / (n - k). We get these efficiently with suffix sums.
    suffix_correct = np.cumsum(c_sorted[::-1])[::-1]  # suffix sums, length n
    kept_counts = np.arange(n, 0, -1)                 # n, n-1, ..., 1

    # Points for k = 0..n-1 (at least one kept)
    rejections = np.arange(0, n, dtype=float) / n
    accuracies = suffix_correct / kept_counts

    # Optionally add terminal point at rejection = 1.0 with a chosen accuracy convention
    if include_terminal:
        rejections = np.append(rejections, 1.0)
        accuracies = np.append(accuracies, terminal_accuracy)

    return rejections, accuracies

def auarc(
    correctness: Iterable[int],
    certainty: Iterable[float],
    include_terminal: bool = True,
    terminal_accuracy: float = 1.0,
) -> float:
    """
    Compute AUARC (Area Under the Accuracy–Rejection Curve) via trapezoidal rule.

    The curve is constructed by rejecting increasingly uncertain examples and
    computing accuracy on the remaining (kept) set.

    Args:
        correctness: Iterable of 0/1 values (1 = correct, 0 = incorrect).
        certainty:   Iterable of certainty/confidence scores (higher = more certain).
        include_terminal: Whether to include the terminal point at rejection=1.0.
        terminal_accuracy: Accuracy assigned at rejection=1.0 (when nothing is kept).
                           Common convention is 1.0.

    Returns:
        AUARC as a float in [0, 1].
    """
    rej, acc = accuracy_rejection_curve(
        correctness, certainty, include_terminal=include_terminal, terminal_accuracy=terminal_accuracy
    )
    # Numerical integration with trapezoidal rule over rejection ∈ [0, 1]
    return float(np.trapz(acc, rej))


def selective_generation(args):
    
    print("\n== Selective Generation ...")
    print(f"""
        Model name:    {args.model_name_or_path}
        Secondary M.:  {args.secondary_model_name_or_path}
        Dataset:       {args.dataset} / {args.subsec} ({args.fraction_of_data_to_use})
        Retriever:     {args.retriever_name} / ({args.retrieval_model_path})
        RAG Method:    {args.rag_method}
        Con. Method:   {args.consistency_method}
        Run:           {args.run}
        Seed:          {args.seed}
    """.replace('            ', ''))

    ### === Load samples ===============
    if args.rag_method in ['flare', 'dragin']:
        consistency_results_file = f"{args.output_dir}/{args.consistency_method}_results_th{args.hallucination_threshold}.jsonl"
        risk_coverage_file = f"{args.output_dir}/selective_generation_plots/{args.consistency_method}_results_th{args.hallucination_threshold}_risk_coverage.png"
        accuracy_coverage_file = f"{args.output_dir}/selective_generation_plots/{args.consistency_method}_results_th{args.hallucination_threshold}_accuracy_coverage.png"
        reliability_file = f"{args.output_dir}/selective_generation_plots/{args.consistency_method}_results_th{args.hallucination_threshold}_reliability.png"
    else:
        consistency_results_file = f"{args.output_dir}/{args.consistency_method}_results.jsonl"
        risk_coverage_file = f"{args.output_dir}/selective_generation_plots/{args.consistency_method}_results_risk_coverage.png"
        accuracy_coverage_file = f"{args.output_dir}/selective_generation_plots/{args.consistency_method}_results_accuracy_coverage.png"
        reliability_file = f"{args.output_dir}/selective_generation_plots/{args.consistency_method}_results_reliability.png"
    os.makedirs(f"{args.output_dir}/selective_generation_plots", exist_ok=True)
          
    scores, labels = [], []
    with open(consistency_results_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if args.consistency_method == 'rag_consistency':
                scores.append(float(obj["ue_scores"]["majority_voting"]["confidence"]))
            else:
                scores.append(float(obj["ue_scores"]["p_true"]["confidence"]))
                # scores.append(float(obj["ue_scores"]["majority_voting"]["most_confident_answer"][1]))
            
            labels.append(int(obj["em"]))
    scores, labels = np.array(scores), np.array(labels) 
    # print(scores)
    # print(labels)
    
    
    threshold = 0.8
    # metrics = compute_rc_rag_metrics(labels, scores, threshold)
    # print(metrics)
    # metrics = abstention_metrics(labels, scores, threshold)
    # print(metrics['counts'])
    # print(metrics['metrics'])
    
    # # scores: np.ndarray in [0,1], labels: np.ndarray in {0,1}
    # rc = compute_risk_coverage(scores, labels)
    # rel = reliability_diagram(scores, labels, n_bins=10)
    # print("AURC:", rc.aurc)      # lower is better
    # print("ECE:", rel.ece)       # lower is better
    
    # # Plot
    # plot_risk_coverage(rc, risk_coverage_file)
    # plot_accuracy_coverage(rc, accuracy_coverage_file)
    # plot_reliability(rel, reliability_file)


    auarc_score = auarc(labels, scores)
    print("AUARC:", auarc_score)
    
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model_name_or_path', type=str, default='PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo')
    # parser.add_argument('--model_name_or_path', type=str, default="agentrl/ReSearch-Qwen-7B-Instruct")
    # parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--secondary_model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='popqa', choices=[
        'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=2000.0)
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
        'direct_inference', 'cot_inference', 'cot_single_retrieval',
        'fix_length_retrieval', 'fix_sentence_retrieval',
        'ircot', 'flare', 'dragin',
        'self_ask', 'react', 'search_o1',
        'research', 'search_r1'
    ])
    parser.add_argument('--generate_fix_length', type=int, default=25)
    parser.add_argument('--modifier_method', type=str, default='token', choices=['token', 'entity'])          # for FLARE
    parser.add_argument('--query_formulation', type=str, default='real_words', choices=[                          # for FLARE & DRAGIN
        'direct', 'forward_all',
        'real_words', 'current', 'current_wo_wrong', 'last_sentence', 'last_n_tokens',
    ])
    parser.add_argument('--sentence_solver', type=str, default='avg', choices=['avg', 'max', 'min'])          # for FLARE
    parser.add_argument('--hallucination_threshold', type=float, default=0.6)                                # for FLARE & DRAGIN
    parser.add_argument('--retrieve_keep_top_k', type=int, default=25)                                        # for DRAGIN
    parser.add_argument('--check_real_words', action='store_false')                                           # for DRAGIN
    parser.add_argument('--max_iter', type=int, default=5)
    
    # Consistency Generation Methods (answer list)
    parser.add_argument('--consistency_method', type=str, default='self_consistency', choices=[
        'fa_consistency', 'rrr_consistency', 'reasoning_consistency', 'self_consistency', 'rag_consistency'
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
    
    # === Files ====================
    model_ = args.model_name_or_path.split('/')[-1]
    if args.rag_method in ['direct_inference', 'cot_inference']:
        args.output_dir = f"run_output/{args.run}/{model_}/{args.dataset}_{args.subsec}/{args.rag_method}"
    else:
        args.output_dir = f"run_output/{args.run}/{model_}/{args.dataset}_{args.subsec}/{args.rag_method}_{args.retriever_name}"
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.rag_method in ['flare', 'dragin']:
        args.inference_results_file = f"{args.output_dir}/inference_results_th{args.hallucination_threshold}.jsonl"
        args.consistency_results_file = f"{args.output_dir}/{args.consistency_method}_results_th{args.hallucination_threshold}.jsonl"
        if args.consistency_method != "self_consistency":
            args.masked_traces_results_file = f"{args.output_dir}/{args.consistency_method}_masked_traces_th{args.hallucination_threshold}.jsonl"
    else:
        args.inference_results_file = f"{args.output_dir}/inference_results.jsonl"
        args.consistency_results_file = f"{args.output_dir}/{args.consistency_method}_results.jsonl"
        if args.consistency_method != "self_consistency":
            args.masked_traces_results_file = f"{args.output_dir}/{args.consistency_method}_masked_traces.jsonl"
        
    # === Prompt files =============
    args.query_decomposition_prompt_file = "run_mcts_two_actions/prompts/query_decomposition_prompt_template.txt"
    args.semantic_equivalence_prompt_file = "run_mcts_two_actions/prompts/semantic_equivalence_prompt_template.txt"
    
    ### === Run Steps =============
    set_seed(args.seed)
    selective_generation(args)
    
    # python applications/selective_generation/run_module.py
    # accelerate launch --multi_gpu applications/selective_generation/run_module.py

