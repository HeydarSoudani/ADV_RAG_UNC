import ast

def wo_training_selector(args, dataset):
    total = correct = 0
    ds = dataset['test']
    
    for idx, ex in enumerate(ds):
        
        # if idx == 1:
        #     break
        
        candidates_str = ex.get("candidates", None)
        candidates = ast.literal_eval(candidates_str)
        if not candidates:
            total += 1
            continue
        
        # print(candidates[0][0])
        
        try:
            best = max(candidates, key=lambda c: (c[1] if c is not None and len(c) > 1 and c[1] is not None else float("-inf")))
            best_correctness = best[2] if len(best) > 2 else 0
        except Exception:
            best_correctness = 0
            
        total += 1
        if best_correctness == 1:
            correct += 1
    
    acc = (correct / total) if total else 0.0
    print(f"Accuracy: {acc:.4f}  ({correct}/{total})")