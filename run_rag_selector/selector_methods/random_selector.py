import json
import pandas as pd
import random

def get_random_selector(args):
    rag_methods = [
        ('Qwen2.5-7B-Instruct', 'self_ask'),
        ('Qwen2.5-7B-Instruct', 'react'),
        ('Qwen2.5-7B-Instruct', 'search_o1'),
        ('ReSearch-Qwen-7B-Instruct', 'research'),
        ('SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo', 'search_r1')
    ]
    
    dfs = []
    for rag_method in rag_methods:
        file_path = f"run_output/{args.run_test}/{rag_method[0]}/{args.dataset}_{args.subsec}/{rag_method[1]}_{args.retriever_name}/inference_results.jsonl"
        with open(file_path, "r") as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)[["qid", "em"]].rename(columns={"em": rag_method[1]})
        dfs.append(df)
    
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(df, on="qid", how="outer")
    
    df_numeric = merged_df.fillna(0).set_index("qid").astype(int)
    # Randomly select one method per question
    random_correct = []
    for qid, row in df_numeric.iterrows():
        chosen_method = random.choice(row.index.tolist())
        random_correct.append(row[chosen_method])
    
    random_accuracy = sum(random_correct) / len(random_correct)
    print(f"Random selector accuracy: {random_accuracy:.4f}")
