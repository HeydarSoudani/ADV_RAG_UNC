import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import json
import ast
import json
import numpy as np
from tqdm import tqdm

def wo_training_selector(args, dataset):
    ds = dataset['test']
    em_evaluation = []
    with open(args.save_results_path, "w") as fout:
        for idx, sample in enumerate(tqdm(ds)):
            qid, query, gt_answers = sample['qid'], sample['query'], ast.literal_eval(sample['gt_answers'])
            candidates_str = sample.get("candidates", None)
            candidates = ast.literal_eval(candidates_str)
            best_cand = max(candidates, key=lambda c: (c[1] if c is not None and len(c) > 1 and c[1] is not None else float("-inf")))
            prediction = best_cand[0]
            correctness = best_cand[2]
        
            item = {
                'qid': qid, 'query': query, 'gt_answers': gt_answers,
                'prediction': prediction,
                'em': correctness
            }
            fout.write(json.dumps(item) + "\n")
            em_evaluation.append(correctness)
        
    print(f"Ideal Selector accuracy: {np.mean(em_evaluation)*100}")