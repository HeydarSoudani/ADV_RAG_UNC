
import json

file_a_path = "run_output/run_4 (search_r1)/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo/bamboogle_test/rerank_l6/rag_consistency_results.jsonl"
file_b_path = "run_output/run_4 (search_r1)/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo/bamboogle_test/rerank_l6/rag_consistency_paths.jsonl"
output_path = "merged_output.jsonl"

id_to_score = {}
with open(file_b_path, "r") as f:
    for line in f:
        entry = json.loads(line)
        id_to_score[entry["qid"]] = entry["consistency_answers"]

with open(file_a_path, "r") as fa, open(output_path, "w") as out:
    for line in fa:
        entry = json.loads(line)
        entry_id = entry["qid"]
        if entry_id in id_to_score:
            entry["score"] = id_to_score[entry_id]
        
        item = {
            "qid": entry["qid"],
            "query": entry["query"],
            "gt_answers": entry["gt_answers"],
            "pred_answer": entry["pred_answer"],
            "correctness_em": entry["correctness_em"],
            "consistency_answers": id_to_score[entry["qid"]]
        }
        out.write(json.dumps(item) + "\n")
