#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from utils.general_utils import set_seed


def prepare_data(args, subsec='train'):
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
        file_path = f"run_output/{args.run}/{rag_method[0]}/{args.dataset}_{subsec}/{rag_method[1]}_{args.retriever_name}/{args.consistency_method}_results.jsonl"
        with open(file_path, "r") as f:
            data = [json.loads(line) for line in f]
        df_temp = pd.DataFrame(data)[["qid", "query", "pred_answer", "em", "ue_scores"]]
        confidences = df_temp["ue_scores"].apply(lambda x: x["majority_voting"]["confidence"])
        
        df_temp[rag_method[1]] = list(zip(df_temp["pred_answer"], df_temp["em"], confidences))
        df = df_temp[["qid", "query", rag_method[1]]]
        dfs.append(df)
    
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(df, on=["qid", "query"], how="outer")
    merged_df = merged_df.sort_values(by="qid", key=lambda x: x.str.extract(r"(\d+)").squeeze().astype(int)).reset_index(drop=True)
    
    # ===
    method_cols = [col for col in merged_df.columns if col not in ["qid", "query"]]
    merged_df["pred_answers"] = merged_df[method_cols].apply(lambda row: [x[0] for x in row], axis=1)
    merged_df["labels"] = merged_df[method_cols].apply(lambda row: [x[1] for x in row], axis=1)
    merged_df["context"] = merged_df[method_cols].apply(lambda row: [x[2] for x in row], axis=1)
    new_df = merged_df[["qid", "query", "pred_answers", "context", "labels"]]
    
    return new_df
    

class LinUCB:
    def __init__(self, n_arms, n_features, alpha, train_df, test_df):
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        self.correctness_weight = 1.0
        self.training_data = None
        self.train_df = train_df
        self.test_df = test_df
        
        # Initialization
        self.A = [np.identity(n_features) for _ in range(n_arms)] 
        self.b = [np.zeros((n_features, 1)) for _ in range(n_arms)]
        
        # Other variables
        self.expected_reward_matrix_history = [] 
        self.ucb_matrix_history = []
        self.context_history = []
        self.chosen_arm_history = []
        self.real_reward_history = []
        self.action_choice_counts = np.zeros((n_features, n_arms, 0))

    def select_arm(self, context):
        UCBs = []
        context = context.reshape(-1, 1)
        expected_rewards = []
        
        for arm in range(self.n_arms):
            theta_arm = np.linalg.inv(self.A[arm]) @ self.b[arm]
            expected_reward_for_arm = theta_arm.T @ context
            expected_rewards.append(expected_reward_for_arm)
            exploration_term = self.alpha * np.sqrt(context.T @ np.linalg.inv(self.A[arm]) @ context)
            UCB = expected_reward_for_arm + exploration_term
            UCB = UCB[0][0]
            UCBs.append(UCB)
            
        self.expected_reward_matrix_history.append((context.flatten(), expected_rewards))
        self.ucb_matrix_history.append((context.flatten(), UCBs))
        self.context_history.append(context.flatten())      
        
        max_UCB = max(UCBs)
        indices_with_max_UCB = [i for i, value in enumerate(UCBs) if value == max_UCB] 
        
        if len(indices_with_max_UCB) > 1:
            selected = np.random.choice(indices_with_max_UCB)
        else:
            selected = indices_with_max_UCB[0]
        
        self.chosen_arm_history.append(selected)
        context_index = np.argmax(context)
        current_counts = self.action_choice_counts[:, :, -1] if self.action_choice_counts.shape[2] > 0 else np.zeros((self.n_features, self.n_arms))
        new_counts = current_counts.copy()
        new_counts[context_index, selected] += 1
        self.action_choice_counts = np.concatenate((self.action_choice_counts, new_counts[:, :, np.newaxis]), axis=2)

        return selected
    
    def update(self, chosen_arm, context, reward):
        context = context.reshape(-1, 1)
        self.A[chosen_arm] += context @ context.T
        self.b[chosen_arm] += reward * context
        self.real_reward_history.append((chosen_arm, reward, np.argmax(context)))  # Record arm, reward, and context

    def get_rewards(self, correctness):
        return self.correctness_weight * correctness

    def run_simulation(self):
        
        # train
        for i, sample in enumerate(self.train_df.to_dict(orient="records")):
            qid, query, context, labels = sample['qid'], sample['query'], sample['context'], sample['labels']
            context = np.array(context)
            chosen_arm = self.select_arm(context)
            correct = labels[chosen_arm]
            actual_reward = self.get_rewards(correct)
            self.update(chosen_arm, context, actual_reward)
            
        # test
        accuracy, total_reward = 0, 0
        cumulative_rewards, expected_rewards = [], []
        for i, sample in enumerate(self.test_df.to_dict(orient="records")):
            qid, query, context, labels = sample['qid'], sample['query'], sample['context'], sample['labels']
            context = np.array(context)
            chosen_arm = self.select_arm(context)
            correct = labels[chosen_arm]
            actual_reward = self.get_rewards(correct)
            self.update(chosen_arm, context, actual_reward)
        
            total_reward += actual_reward
            accuracy += correct
            cumulative_rewards.append(total_reward)
            expected_rewards.append(actual_reward)
    
        accuracy_f = accuracy / len(self.test_df)
        print(f"Acc: {accuracy_f}")
        
        # self.plot_cumulative_rewards(range(len(self.data_df)), cumulative_rewards)
        self.plot_expected_rewards(range(len(self.test_df)), expected_rewards)

    
    # --------- Plots -------------------
    # def plot_cumulative_rewards(self, x_axis, cumulative_rewards):
    #     plt.plot(x_axis, cumulative_rewards)
    #     plt.xlabel("Rounds")
    #     plt.ylabel("Cumulative Reward")
    #     plt.title("LinUCB: Cumulative Reward over Time")
    #     plt.savefig("aaa.png", dpi=300, bbox_inches="tight")
    
    def plot_expected_rewards(self, x_axis, expected_rewards):
        plt.plot(x_axis, expected_rewards)
        plt.xlabel("Rounds")
        plt.ylabel("Cumulative Reward")
        plt.title("LinUCB: Expected Reward over Time")
        plt.savefig("bbb.png", dpi=300, bbox_inches="tight")



    # def simulation(self, test_contexts, test_labels, update=True):
    #     print("Perfoming simulation...")
    #     policy = []
    #     policy_index = []
    #     accuracy = 0
    #     total_price = 0
    #     for i, context in enumerate(pbar := tqdm(test_contexts, disable=False)):
    #         # Select arm and update policy
    #         selected_arm = self.select_arm(context)
    #         selected_arm_index = self.arms.index(selected_arm)
    #         policy.append(selected_arm)
    #         policy_index.append(selected_arm_index)

    #         # Get reward and update arm's model
    #         correct = test_labels[i, selected_arm_index]
    #         reward = self.get_rewards(arm=selected_arm, corrects=correct)
    #         accuracy += correct.item()
    #         total_price += self.prices[selected_arm]
    #         if update:
    #             self.update(selected_arm, context, reward=reward)
    #         pbar.set_description(f'Test Accuracy: {accuracy / (i + 1):.4f}, Test Price: {total_price:.2f}')
        
    #     # Calculate accuracy and price
    #     chosen_arm_idx = torch.tensor(policy_index).cpu()
    #     correct = test_labels[range(len(test_labels)), chosen_arm_idx]
    #     accuracy = correct.sum().item() / len(test_labels)
    
    #     # price
    #     counter = Counter(policy)
    #     total_price = 0.0
    #     for arm, count in counter.items():
    #         total_price += self.prices[arm] * count

    #     # Calculate price per 10000 samples
    #     total_price = total_price / len(test_contexts) * 10000
        
    #     print(f'Test Accuracy: {accuracy / len(test_contexts):.4f}, Test Price: {total_price:.2f}')
    #     return accuracy / len(test_contexts), total_price


def train_rag_selector(args):
    alpha = 2.5 # Exploration parameter
    n_arms = 5  
    n_features = 5 # Context dim
    arms = ['self_ask', 'react', 'search_o1', 'research', 'search_r1']
    train_df = prepare_data(args, 'train')
    test_df = prepare_data(args, 'dev')
    lin_ucb = LinUCB(n_arms, n_features, alpha, train_df, test_df)
    lin_ucb.run_simulation()
    

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
    parser.add_argument('--consistency_method', type=str, default='rag_consistency', choices=[
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
    train_rag_selector(args)
    
    # python rag_selection_application/rag_selection_trained.py
    # accelerate launch --multi_gpu rag_selection_application/rag_selection_trained.py
