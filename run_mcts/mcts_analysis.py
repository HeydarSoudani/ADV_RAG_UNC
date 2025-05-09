import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import argparse
import json
import matplotlib.pyplot as plt
from collections import Counter

def load_jsonl(file_path):
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            qid = item['qid']
            
            if 'em' in list(item.keys()):
                em = item['em']
            else:
                em = item['EM']
            data[qid] = em
    return data

def categorize_qids(file_a_path, file_b_path):
    a_data = load_jsonl(file_a_path)
    b_data = load_jsonl(file_b_path)

    a_qids = set(a_data.keys())
    b_qids = set(b_data.keys())
    common_qids = a_qids & b_qids

    both_correct = []
    both_incorrect = []
    a_correct_b_incorrect = []
    a_incorrect_b_correct = []
    not_common = list((a_qids ^ b_qids))  # symmetric difference

    for qid in common_qids:
        a_em = a_data[qid]
        b_em = b_data[qid]
        if a_em == 1 and b_em == 1:
            both_correct.append(qid)
        elif a_em == 0 and b_em == 0:
            both_incorrect.append(qid)
        elif a_em == 1 and b_em == 0:
            a_correct_b_incorrect.append(qid)
        elif a_em == 0 and b_em == 1:
            a_incorrect_b_correct.append(qid)

    return {
        'both_correct': both_correct,
        'both_incorrect': both_incorrect,
        'a_correct_b_incorrect': a_correct_b_incorrect,
        'a_incorrect_b_correct': a_incorrect_b_correct,
        'not_common': not_common
    }


def plot_num_retrieval(file_path, output_png):
    lengths = []

    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            path = data['path']
            
            if isinstance(path, list):
                path_length = len(path) - 1
            elif isinstance(path, str):
                path_length = len(path) - 1
            else:
                continue  # Skip if path is not string or list
            lengths.append(path_length)

    print(f"Avg: {sum(lengths) / len(lengths)}")

    length_counter = Counter(lengths)
    sorted_lengths = sorted(length_counter.items())
    x, y = zip(*sorted_lengths)
    
    plt.figure(figsize=(8, 5))
    plt.bar(x, y)
    plt.xlabel('Path Length')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of Path Lengths')
    # plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_png)
    print(f"Plot saved to {output_png}")



def mcts_analysis():
    # === Read files ============
    file_a_path = 'run_output/run_4 (search_r1)/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo/hotpotqa_dev/rerank_l6/inference_results.jsonl' # File one -> Search-R1 results
    file_b_path = 'run_output/run_5 (edited_prompt_roll4)/Qwen2.5-7B-Instruct/hotpotqa_dev/rerank_l6/rc_discriminate_results_v2.jsonl'   # File two -> mine results
    
    output = categorize_qids(file_a_path, file_b_path)
    for key, value in output.items():
        print(f"{key}: {len(value)} | {value}")
        print('\n')
        
        # print(f"{key}: {len(value)}")
    
    # print(output)
    # file_path = "run_output/run_4 (search_r1)/Qwen2.5-7B-Instruct/2wikimultihopqa_dev/rerank_l6/path_results.jsonl"
    # output_png = "ret_num_dist.png"
    # plot_num_retrieval(file_path, output_png)

# Example usage:
# result = categorize_qids('file_a.jsonl', 'file_b.jsonl')
# for group, qids in result.items():
#     print(f"{group}: {len(qids)} qids")

    
    

if __name__ == "__main__":    
    mcts_analysis()
    
    
    # python run_mcts/mcts_analysis.py
