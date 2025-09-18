import ast
import json
import numpy as np
import llm_blender
from tqdm import tqdm


def get_llm_blender(args, dataset):
    blender = llm_blender.Blender()
    blender.loadranker("llm-blender/PairRM") # load PairRM
    ds = dataset['test']
    
    em_evaluation = []
    with open(args.save_results_path, "w") as fout:
        for idx, sample in enumerate(tqdm(ds)):
            # if idx == 10:
            #     break
            qid, query, gt_answers = sample['qid'], sample['query'], ast.literal_eval(sample['gt_answers'])
            candidates_str = sample.get("candidates", None)
            candidates = ast.literal_eval(candidates_str)
            # predictions = [f"{c[0]}" for c in candidates]
            predictions = [f"{c[0]} with confidence {str(c[1])}" for c in candidates]
            ranks = blender.rank([query], [predictions], return_scores=False, batch_size=1)[0]
            best_idx = np.where(ranks == 1)[0][0]
            
            correctness = candidates[best_idx][2]
            prediction = candidates[best_idx][0]
        
            item = {
                'qid': qid, 'query': query, 'gt_answers': gt_answers,
                'prediction': prediction, 'em': correctness
            }
            fout.write(json.dumps(item) + "\n")
            em_evaluation.append(correctness)
        
    print(f"Prompt accuracy: {np.mean(em_evaluation)*100}")

 