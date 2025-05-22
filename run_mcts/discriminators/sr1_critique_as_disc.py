#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import json
import torch
import random
import argparse
import jsonlines
import numpy as np
import transformers
from tqdm import tqdm, trange

from run_rag_methods.src.correctness import em_score, normalize_answer
from utils.general_utils import set_seed, read_jsonl
from run_searchr1.inference import get_think, get_query, get_answer, _passages2string, StopOnSequence
from run_searchr1.retrieval_local import BM25Retriever, ContrieverRetriever, RerankRetriever, DenseRetriever
from run_mcts.searchr1_discrimination import SemanticEquivalenceGenerator

def get_decision(text):
    pattern = re.compile(r"<decision>(.*?)</decision>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None

def select_item(options):
    # Strip whitespace and filter correct answers
    correct_items = [country.strip() for country, status in options if status.strip() == 'Correct']
    
    if len(correct_items) == 1:
        return correct_items[0]
    elif len(correct_items) > 1:
        return random.choice(correct_items)
    else:
        # No correct answers, pick from all options
        all_items = [country.strip() for country, _ in options]
        return random.choice(all_items)

def _filter_none(candidates: list[str]) -> list[str]:
        candidates = [c for c in candidates if c is not None]
        return candidates

def _filter_long(candidates: list[str]) -> list[str]:
    candidates = [c for c in candidates if len(c) <= 80]
    return candidates

def _filter_white_space(candidates: list[str]) -> list[str]:
    candidates = [c for c in candidates if c.strip()]
    return candidates

def _filter_specific_words(candidates: list[str]) -> list[str]:
    words = ['can not answer', 'CAN NOT ANSWER', 'not enough information provided', 'unknown', 'more information needed', 'none', 'not specified in the given information', 'information not specified', 'no direct information available in current context', 'no direct information available in the knowledge base.']
    filtered_candidates = []
    for c in candidates:
        normalized_c = normalize_answer(c)
        if not any(w in normalized_c for w in words):
            filtered_candidates.append(c)
    return filtered_candidates


def em_score(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score

    


def sr1_critique_discrimination(args):
    print("\n== Search-R1 Critique Discrimination ...")
    print(f"""
        Model name:  {args.model_name_or_path}
        Dataset:     {args.dataset} / {args.subsec} ({args.fraction_of_data_to_use})
        Retriever:   {args.retriever_name} / ({args.retrieval_model_path})
        Seed:        {args.seed}
        Run:         {args.run}
    """.replace('        ', ''))
    
    # === Output files =========================
    entries = os.listdir(args.generation_trees_results_dir)
    query_ids = [entry for entry in entries if os.path.isdir(os.path.join(args.generation_trees_results_dir, entry))]
    sorted_query_ids = sorted(query_ids, key=lambda x: int(x.split('_')[1]))
    
    # === generator Model ======================
    
    
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path_sr, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path_sr)
    target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
    stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])
    curr_eos = [151645, 151643] # for Qwen2.5 series models
    curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

    challenging_samples = ['test_24', 'test_27', 'test_47', 'test_52', 'test_64']

    # === Static Retriever ===================== 
    if args.retriever_name == 'bm25':
        retriever = BM25Retriever(args)  
    elif args.retriever_name == 'contriever':
        retriever = ContrieverRetriever(args)
    elif args.retriever_name in ['rerank_l6', 'rerank_l12']:
        retriever = RerankRetriever(args)
    elif args.retriever_name in ['e5', 'bge']:
        retriever = DenseRetriever(args)
    else:
        raise NotImplementedError("...")
    
    # === Semantic clustering ================== 
    se_model = SemanticEquivalenceGenerator(args, model=None, tokenizer=None)
    
    # === Prompt ===============================
    sys_prompt = """"""
    user_prompt = """Critique the given answer option for the question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
Your reasoning must begin by analyzing the provided option in the context of the question, not by independently answering the question. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the decision inside <decision> and </decision>, without detailed illustrations. \
The final decision must be either <decision> Correct </decision> or <decision> Incorrect </decision> to reflect the correctness of the given answer. \
\nQuestion: {question}\nOption: {option}\n"""
    
    # === Read existing data ===================
    generated_qids = []
    generated_em = []
    if os.path.exists(args.discriminate_results_file):
        with open(args.discriminate_results_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'qid' in data:
                    generated_qids.append(data['qid'])
                    generated_em.append(data['em'])
    
    # === Inference ============================
    em_evaluation = generated_em
    with jsonlines.open(args.discriminate_results_file, mode='a') as inf_file:
        for idx, qid in enumerate(tqdm(sorted_query_ids)):
            # if idx == 40:
            #     break
            
            final_solutions_file = f"{args.generation_trees_results_dir}/{qid}/final_solutions.jsonl"
            trace_js = read_jsonl(final_solutions_file)  
            question = trace_js[0]["trace"]["0"]["user_question"]
            question = question.strip()
            if question[-1] != '?':
                question += '?'
            
            if qid in generated_qids:
                print(f"The answer for query {qid} has been already generated")
            else:
                per_query_decision = []
                gt_answers = trace_js[0]["trace"]["0"]["ground_truth"]
                options = [s["trace"][list(s["trace"].keys())[-1]]['think_answer']["answer"] for s in trace_js]
                options = _filter_none(options)
                options = _filter_long(options)
                options = _filter_white_space(options)
                options = _filter_specific_words(options)
                cls_options = se_model.cluster_by_meaning(question, options)
                
                if len(cls_options) == 0:
                    pred_answer = ''
                    per_query_decision.append((pred_answer, 'Incorrect'))
                    cnt = 0
                elif len(cls_options) == 1:
                    pred_answer = random.choice(cls_options[0])
                    per_query_decision.append((pred_answer, 'Correct'))
                    cnt = 0
                else:
                    # print(cls_options)
                    print(f"Generating search-R1 for query {qid} ...")
                    unq_options = [random.choice(cls_) for cls_ in cls_options]
                    for answer_option in unq_options:
                        
                        input_prompt = user_prompt.format(question=question, option=answer_option)
                        if tokenizer.chat_template:
                            input_prompt = tokenizer.apply_chat_template(
                                [{"role": "user", "content": input_prompt}],
                                add_generation_prompt=True,
                                tokenize=False
                            )
                        # print(input_prompt)
                            
                        cnt = 0
                        while True:
                            input_ids = tokenizer.encode(input_prompt, return_tensors='pt').to(args.device)
                            attention_mask = torch.ones_like(input_ids)
                            
                            # Generate text with the stopping criteria
                            outputs = model.generate(
                                input_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=args.max_new_token,
                                stopping_criteria=stopping_criteria,
                                pad_token_id=tokenizer.eos_token_id,
                                do_sample=True,
                                temperature=0.7
                            )

                            if outputs[0][-1].item() in curr_eos:
                                generated_tokens = outputs[0][input_ids.shape[1]:]
                                output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                                break

                            generated_tokens = outputs[0][input_ids.shape[1]:]
                            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                                    
                            # print(output_text)
                            # print('---')
                                                
                            tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
                            if tmp_query:
                                search_docs = retriever.search(tmp_query)
                                search_results = _passages2string(search_docs)
                            else:
                                search_docs = []
                                search_results = ''

                            # path.append({
                            #     'think': get_think(output_text),
                            #     'search_query': tmp_query,
                            #     'docs': search_docs
                            # })

                            search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
                            input_prompt += search_text
                            cnt += 1
                        
                        # print(output_text)
                        pred_answer = get_decision(output_text)
                        per_query_decision.append((answer_option, pred_answer))
                
                print(per_query_decision)
                print('\n')
                
                pred_answer = select_item(per_query_decision)
                correctness_em = em_score(pred_answer, gt_answers)
                em_evaluation.append(correctness_em)
                
                item = {
                    "qid": qid,
                    "query": question,
                    "gt_answers": gt_answers,
                    "pred_answer": pred_answer,
                    "em": correctness_em,
                    "Decisions": per_query_decision,
                    "cluster_options": cls_options,
                    "num_ret": cnt
                }
                inf_file.write(item)
                

    # === Print results ========================
    print("\nEvaluation Result:")
    print(f"EM: {np.mean(em_evaluation)*100}")
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--model_name_or_path_sr', type=str, default='PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo')
    parser.add_argument('--max_new_token', type=int, default=1024)
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='bamboogle', choices=[
        'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
    parser.add_argument("--enable_fewshot_examples", action="store_true", help="")
    
    # Retriever
    parser.add_argument('--retriever_name', type=str, default='rerank_l6', choices=[
        'bm25', 'contriever', 'rerank_l6', 'rerank_l12', 'e5', 'bge'
    ])
    parser.add_argument('--corpus_path', type=str, default='data/search_r1_files/wiki-18.jsonl')
    parser.add_argument('--index_path', type=str, default='data/search_r1_files/bm25', choices=[
        'data/search_r1_files/bm25',          # For BM25 & Rerank
        'data/search_r1_files/e5_Flat.index', # For E5
    ])
    parser.add_argument("--retrieval_model_path", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2", choices=[
        "cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-MiniLM-L12-v2", # For Rerank
        "intfloat/e5-base-v2" # For E5
    ])
    parser.add_argument('--retrieval_topk', type=int, default=3)
    parser.add_argument('--faiss_gpu', action='store_false', help='Use GPU for computation')
    parser.add_argument('--retrieval_pooling_method', type=str, default="mean")
    parser.add_argument('--retrieval_query_max_length', type=int, default=256)
    parser.add_argument('--retrieval_use_fp16', action='store_false', help='')
    parser.add_argument('--retrieval_batch_size', type=int, default=512)
    parser.add_argument("--bm25_k1", type=float, default=0.9)
    parser.add_argument("--bm25_b", type=float, default=0.4)
    
    # Others
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--run', type=str, default='run_5 (edited_prompt_roll4)')
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument('--use_counter', action='store_false')
    
    # MCTS ---
    parser.add_argument("--enable_critique", action="store_true", help="")
    parser.add_argument("--verbose", action="store_true", help="extra login")
    parser.add_argument("--mcts_discount_factor", type=float, default=1.0)
    parser.add_argument("--mcts_exploration_weight", type=float, default=2.0)
    parser.add_argument("--mcts_weight_scheduler", choices=["exp", "lin", "const"], default="const")
    parser.add_argument("--save_tree", action="store_true")
    parser.add_argument("--num_rollouts", type=int, default=4)
    parser.add_argument("--max_depth_allowed", type=int, default=10)
    parser.add_argument("--num_votes", type=int, default=1)
    parser.add_argument("--mcts_num_last_votes", type=int, default=1)
    parser.add_argument("--enable_potential_score", action="store_true")
    
    # Discrimination ---
    parser.add_argument("--cutoff_rollout", type=int, default=-1)
    parser.add_argument("--start_idx", type=int, default=-1)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--mask_left_boundary", type=float, default=0.2)
    parser.add_argument("--mask_right_boundary", type=float, default=0.5)
    parser.add_argument("--num_masked_solution_traces", type=int, default=4)
    parser.add_argument("--rc_mode", type=str, default="mid", choices=["loose", "mid", "strict", "maj"])
    parser.add_argument("--rc_temperature", type=float, default=1.0)
    parser.add_argument("--rc_n_completions", type=int, default=1)
    parser.add_argument("--rc_criteria", type=str, default="freq", choices=["freq", "reward"])
    parser.add_argument("--threshold", type=float, default=0.999)
    parser.add_argument("--extend_rc_mode", type=str, default="majority_vote", choices=["original", "BoN", "majority_vote"])
    parser.add_argument("--best_of", type=int, default=5)
    
    args = parser.parse_args()
    
    # === Files ====================
    model_ = args.model_name_or_path.split('/')[-1]
    output_dir = f"run_output/{args.run}/{model_}/{args.dataset}_{args.subsec}/{args.retriever_name}"
    args.generation_trees_results_dir = f'{output_dir}/generation_trees'
    args.discriminate_results_file = f"{output_dir}/discriminate_results_sr1c.jsonl"
    os.makedirs(args.generation_trees_results_dir, exist_ok=True)
    
    # === Prompt files =============
    args.query_decomposition_prompt_file = "prompts_mcts/query_decomposition_prompt_template.txt"
    args.semantic_equivalence_prompt_file = "prompts_mcts/semantic_equivalence_prompt_template.txt"
    
    # === Define CUDA device =======
    args.device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. No GPUs detected.")
        
    ### === Run Steps =============
    set_seed(args.seed)
    sr1_critique_discrimination(args)
    
    
    # python run_mcts/sr1_critique_discrimination.py
    
