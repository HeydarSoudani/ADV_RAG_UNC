#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import ast
import json
import torch
import random
import argparse
import numpy as np
import transformers
from tqdm import tqdm
from accelerate import Accelerator
from typing import List, Tuple, Dict, Any, Optional

from utils.general_utils import set_seed
from applications.rag_selector.confidence_in_input.listwise_run import data_creation
from run_mcts_two_actions.src.models.semantic_equivalence import SemanticEquivalenceGenerator

Candidate = Tuple[str, float, int]

def rag_selector(args):
    # === MultiGPU setup ==========
    accelerator = Accelerator()
    device = accelerator.device
    
    
    # === RAGSelector =============
    class ClusterBasedRAGSelector:
        def __init__(self, model, tokenizer, device, args):
            self.se_model = SemanticEquivalenceGenerator(args, device, model, tokenizer)
            self.device = device
            self.args = args
            self.rng = random.Random(args.seed)
            
        def get_clusters(self, question, candidates):
            clusters = []
            for cand in candidates:
                pred, conf, corr = cand
                placed = False

                for cluster in clusters:
                    rep_pred, _, _ = cluster[0]
                    if self.se_model.check_answers_equiv(question, pred, rep_pred):
                        cluster.append(cand)
                        placed = True
                        break

                if not placed:
                    clusters.append([cand])

            return clusters
        
        def summarize_clusters(self, clusters: List[List[Candidate]]) -> List[Dict[str, Any]]:
            summaries = []
            for cluster in clusters:
                if not cluster:
                    continue

                confidence_sum = sum(conf for _, conf, _ in cluster)
                positives = [c for c in cluster if int(c[2]) == 1]
                if positives:
                    correctness = 1
                    representative = self.rng.choice(positives)
                else:
                    correctness = 0
                    representative = self.rng.choice(cluster)

                summaries.append({
                    "members": cluster,
                    "confidence_sum": confidence_sum,
                    "correctness": correctness,
                    "representative": representative,
                })

            return summaries
            
        def select_final_answer(self, clusters_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
            if not clusters_summaries:
                return {"prediction": None, "correctness": None, "chosen_cluster": None}

            max_conf = max(c["confidence_sum"] for c in clusters_summaries)
            top_clusters = [c for c in clusters_summaries if c["confidence_sum"] == max_conf]
            chosen_cluster = self.rng.choice(top_clusters)
            prediction, _, _ = chosen_cluster["representative"]
            return prediction, chosen_cluster["correctness"]
        
        def inference(self, question, candidates):
            clusters = self.get_clusters(question, candidates)
            clusters_summaries = self.summarize_clusters(clusters)
            prediction, correctness = self.select_final_answer(clusters_summaries)
            return prediction, correctness, clusters
    
    
    # === Load dataset ============
    RAG_METHODS = ['self_ask', 'react', 'search_o1', 'research', 'search_r1']
    dataset = data_creation(args, train=False, test=True)
    print('---')
    print(f"Test sample:  {dataset['test'][0]}")
    print('---')
    
    
    # === Inference ... ===========
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path, dtype=torch.bfloat16).to(args.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    cluster_based_rag_selector = ClusterBasedRAGSelector(model, tokenizer, device, args)
    
    em_evaluation = []
    with open(args.result_file, "w", encoding="utf-8") as fout:
        for idx, sample in enumerate(tqdm(dataset["test"])):
            # if idx == 5:
            #     break
            
            qid, query = sample['qid'], sample['query']
            candidates = []
            for key, val in sample.items():
                if key in RAG_METHODS:
                    parsed = ast.literal_eval(val)
                    candidates.append((parsed[0], parsed[2], parsed[1]))  # (pred, conf, em)
                        
            prediction, correctness_em, clusters = cluster_based_rag_selector.inference(query, candidates)

            item = {
                'qid': qid, 'query': query,
                'prediction': prediction, 'em': correctness_em,
                'clusters': clusters,
                'candidates': candidates
            }
            fout.write(json.dumps(item) + "\n")
            em_evaluation.append(correctness_em)
    
    print("\nEvaluation Result:")
    print(f"EM: {np.mean(em_evaluation)*100}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    # Dataset
    parser.add_argument('--dataset', type=str, default='popqa', choices=['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test', 'validation'])
    # Retriever
    parser.add_argument('--retriever_name', type=str, default='rerank_l6', choices=[
        'bm25', 'contriever', 'rerank_l6', 'rerank_l12', 'e5', 'bge', 'reasonir'
    ])    
    # Consistency Generation Methods (answer list)
    parser.add_argument('--consistency_method', type=str, default='rag_consistency', choices=[
        'fa_consistency', 'rrr_consistency', 'reasoning_consistency', 'self_consistency', 'rag_consistency'
    ])
    # Others
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--run', type=str, default='run_3 (rag_methods_500)')
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument('--use_counter', action='store_false')
    args = parser.parse_args()
    
    args.result_file = f"run_output/{args.run}/rag_selection_reward_modeling/{args.dataset}_{args.retriever_name}_{args.consistency_method}/{args.subsec}_inference_results_cluster_based.jsonl"
    args.semantic_equivalence_prompt_file = "run_mcts_two_actions/prompts/semantic_equivalence_prompt_template.txt"
    
    # === Run Steps ================
    set_seed(args.seed)
    rag_selector(args)
    
    # python applications/rag_selector/cluster_based.py
    