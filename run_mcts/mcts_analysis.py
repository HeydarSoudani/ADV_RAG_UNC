import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import argparse

def load_jsonl(file_path):
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            qid = item['qid']
            em = item['em']
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


def mcts_analysis():
    # === Read files ============
    file_a_path = 'run_output/run_4 (search_r1)/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo/bamboogle_test/rerank_l6/inference_results.jsonl' # File one -> Search-R1 results
    file_b_path = 'run_output/run_5 (edited_prompt_roll4)/Qwen2.5-7B-Instruct/bamboogle_test/rerank_l6/rc_discriminate_results_v2.jsonl'   # File two -> mine results
    
    output = categorize_qids(file_a_path, file_b_path)
    for key, value in output.items():
        print(f"{key}: {len(value)} | {value}")
    # print(output)
    

# Example usage:
# result = categorize_qids('file_a.jsonl', 'file_b.jsonl')
# for group, qids in result.items():
#     print(f"{group}: {len(qids)} qids")

    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--max_new_token', type=int, default=1024)
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='bamboogle', choices=[
        'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
    
    # Retriever
    parser.add_argument('--retriever_name', type=str, default='rerank_l6', choices=[
        'bm25', 'contriever', 'rerank_l6', 'rerank_l12', 'e5', 'bge'
    ])
    args = parser.parse_args()


    
    mcts_analysis()
    
    
    # python run_mcts/mcts_analysis.py
