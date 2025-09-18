#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import json
import ast
import json
import numpy as np
from tqdm import tqdm


def get_ideal_selector(args, dataset):
    rng = np.random.default_rng(getattr(args, "seed", 42))
    ds = dataset['test']
    
    em_evaluation = []
    with open(args.save_results_path, "w") as fout:
        for idx, sample in enumerate(tqdm(ds)):
            # if idx == 10:
            #     break
            qid, query, gt_answers = sample['qid'], sample['query'], ast.literal_eval(sample['gt_answers'])
            candidates_str = sample.get("candidates", None)
            candidates = ast.literal_eval(candidates_str)
    
            correct_idxs = [i for i, c in enumerate(candidates) if c[2] == 1]
            if correct_idxs:
                best_idx = rng.choice(correct_idxs)
            else:
                best_idx = rng.integers(len(candidates))
                
            prediction = candidates[best_idx][0]
            correctness = candidates[best_idx][2]
                
            item = {
                'qid': qid, 'query': query, 'gt_answers': gt_answers,
                'prediction': prediction, 'em': correctness
            }
            fout.write(json.dumps(item) + "\n")
            em_evaluation.append(correctness)
    
    print(f"Ideal Selector accuracy: {np.mean(em_evaluation)*100}")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # rag_methods = [
    #     ('Qwen2.5-7B-Instruct', 'self_ask'),
    #     ('Qwen2.5-7B-Instruct', 'react'),
    #     ('Qwen2.5-7B-Instruct', 'search_o1'),
    #     ('ReSearch-Qwen-7B-Instruct', 'research'),
    #     ('SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo', 'search_r1')
    # ]
    
    # dfs = []
    # for rag_method in rag_methods:
    #     file_path = f"run_output/{args.run_test}/{rag_method[0]}/{args.dataset}_{args.subsec}/{rag_method[1]}_{args.retriever_name}/inference_results.jsonl"
    #     with open(file_path, "r") as f:
    #         data = [json.loads(line) for line in f]
    #     df = pd.DataFrame(data)[["qid", "em"]].rename(columns={"em": rag_method[1]})
    #     dfs.append(df)
    
    # merged_df = dfs[0]
    # for df in dfs[1:]:
    #     merged_df = merged_df.merge(df, on="qid", how="outer")
    
    # df_numeric = merged_df.fillna(0).set_index("qid").astype(int)
    # oracle_correct = (df_numeric.max(axis=1) > 0).astype(int)
    # oracle_accuracy = oracle_correct.mean()
    # print(f"Oracle upper bound accuracy: {oracle_accuracy:.4f}")
    
    
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
