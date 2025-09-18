import ast
import json
import numpy as np
from tqdm import tqdm


def get_random_selector(args, dataset):
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
    
            selected_idx = rng.integers(len(candidates))
            prediction = candidates[selected_idx][0]
            correctness = candidates[selected_idx][2]
                
            item = {
                'qid': qid, 'query': query, 'gt_answers': gt_answers,
                'prediction': prediction, 'em': correctness
            }
            fout.write(json.dumps(item) + "\n")
            em_evaluation.append(correctness)
    
    print(f"Random Selector accuracy: {np.mean(em_evaluation)*100}")

    












    
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
# # Randomly select one method per question
# random_correct = []
# for qid, row in df_numeric.iterrows():
#     chosen_method = random.choice(row.index.tolist())
#     random_correct.append(row[chosen_method])

# random_accuracy = sum(random_correct) / len(random_correct)
# print(f"Random selector accuracy: {random_accuracy:.4f}")
