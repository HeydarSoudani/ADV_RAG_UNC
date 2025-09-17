import ast
import torch
import numpy as np
import llm_blender
import transformers
from tqdm import tqdm


def get_llm_blender(args, dataset):
    blender = llm_blender.Blender()
    blender.loadranker("llm-blender/PairRM") # load PairRM
    
    ds = dataset['test']
    total_correctness = count = 0
    for idx, sample in enumerate(tqdm(ds)):
        # if idx == 10:
        #     break
        
        query = sample.get("query", None)
        candidates_str = sample.get("candidates", None)
        candidates = ast.literal_eval(candidates_str)
        predictions = [f"{c[0]} with confidence {str(c[1])}" for c in candidates]
        # predictions = [f"{c[0]}" for c in candidates]
        ranks = blender.rank([query], [predictions], return_scores=False, batch_size=1)[0]
        best_idx = np.where(ranks == 1)[0][0]
        correctness = candidates[best_idx][2]
        
        total_correctness += correctness
        count += 1
        
    print(f"LLM Blender accuracy: {(total_correctness/count):.4f}")

 