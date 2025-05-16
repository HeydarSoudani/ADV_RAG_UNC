#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import glob
import torch
import random
import argparse
import transformers
from tqdm import tqdm
from accelerate import Accelerator
from sklearn.metrics import roc_auc_score

from utils.general_utils import set_seed
from run_searchr1.correctness import em_score
from run_searchr1.inference import StopOnSequence, get_think, get_query, get_answer, _passages2string
from run_searchr1.retrieval_local import BM25Retriever, ContrieverRetriever, RerankRetriever, DenseRetriever
from run_mcts.src.generate_paraphrase import SearchQueryGenerator, ThinkGenerator, get_paraphrased_query, get_paraphrased_think
from run_searchr1.src.ue_methods import BlackBoxUncertainty, PTrue
from run_mcts.searchr1_discrimination import SemanticEquivalenceGenerator

def searchr1_rag_consistency(args):
    # === MultiGPU setup =======================
    accelerator = Accelerator()
    device = accelerator.device
    
    if accelerator.is_main_process:
        print("\n== Search R1 RAG Consistency ...")
        print(f"""
            Model name:  {args.model_name_or_path}
            Dataset:     {args.dataset} / {args.subsec} ({args.fraction_of_data_to_use})
            Retriever:   {args.retriever_name} / ({args.retrieval_model_path})
            Seed:        {args.seed}
            Run:         {args.run}
        """.replace('        ', ''))
    
        # Define CUDA device 
        if torch.cuda.is_available():
            print(f"Number of available GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA is not available. No GPUs detected.")
        print('\n')
        
    # === Generation Model ======================
    generator = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
    stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])
    curr_eos = [151645, 151643] # for Qwen2.5 series models
    curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'
    
    # === Other models =========================
    sq_model = transformers.AutoModelForCausalLM.from_pretrained(args.paraphrase_model_name_or_path, torch_dtype=torch.bfloat16).to(device)
    sq_tokenizer = transformers.AutoTokenizer.from_pretrained(args.paraphrase_model_name_or_path)
    sq_generator = SearchQueryGenerator(args, sq_model, sq_tokenizer)
    think_generator = ThinkGenerator(args, sq_model, sq_tokenizer)

    # === Static Retriever ===================== 
    # if accelerator.is_main_process:
    if args.retriever_name == 'bm25':
        retriever = BM25Retriever(args)  
    elif args.retriever_name == 'contriever':
        retriever = ContrieverRetriever(args)
    elif args.retriever_name in ['rerank_l6', 'rerank_l12']:
        retriever = RerankRetriever(args)
    elif args.retriever_name in ['e5', 'bge']:
        retriever = DenseRetriever(args)

    # === Prompt ===============================
    prompt = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""


    # === Read path ============================
    path_qids = {}
    if os.path.exists(args.path_results_file):
        with open(args.path_results_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'qid' in data:
                    path_qids[data['qid']] = data
    
    generated_qids = []
    if os.path.exists(args.rag_consistency_results_file):
        with open(args.rag_consistency_results_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'qid' in data:
                    generated_qids.append(data['qid'])
    filtered_qids = [x for x in list(path_qids.keys()) if x not in generated_qids]


    # === Inference ============================
    accelerator.wait_for_everyone() # sync GPUs
    with accelerator.split_between_processes(filtered_qids) as filtered_qids_shard:
        rag_consistency_results_file_ranked = f"{args.output_dir}/rag_consistency_results_rank{accelerator.process_index}.jsonl"
        # rag_consistency_paths_file_ranked = f"{args.output_dir}/rag_consistency_paths_rank{accelerator.process_index}.jsonl" , open(rag_consistency_paths_file_ranked, "w") as path_f
        with open(rag_consistency_results_file_ranked, "w") as res_f:
            for i, qid in enumerate(tqdm(filtered_qids_shard, desc=f"[Rank {accelerator.process_index}]")):
                # if i == 10:
                #     break

                sample = path_qids[qid]
                query, gt_answers = sample['query'], sample['gt_answers']
                pred_answer, path = sample['pred_answer'], sample['path']

                ### === Get RAG-consistency
                consistency_paths = []
                if len(path) == 1: # process without search
                    original_think = path[0].get('think', '')
                    think_prompt = think_generator.get_instruction(original_think, n=args.num_rag_generations)
                    think_output = think_generator.generate(think_prompt)[0]
                    paraphrased_thinks = get_paraphrased_think(think_output)
                    for paraphrased_think in paraphrased_thinks:
                        
                        # After break point
                        input_prompt = prompt.format(question=query)
                        input_prompt += f"\n\n<think>{paraphrased_think}</think>\n"
                        
                        if tokenizer.chat_template:
                            input_prompt = tokenizer.apply_chat_template(
                                [{"role": "user", "content": input_prompt}],
                                add_generation_prompt=True, tokenize=False
                            )
                        
                        # ---- Generation ----------------
                        while True:
                            input_ids = tokenizer.encode(input_prompt, return_tensors='pt').to(device)
                            attention_mask = torch.ones_like(input_ids)
                            outputs = generator.generate(
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
                            tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
                            if tmp_query:
                                search_docs = retriever.search(tmp_query)
                                search_results = _passages2string(search_docs)
                            else:
                                search_docs, search_results = [], ''

                            new_path.append({
                                'think': get_think(output_text),
                                'search_query': tmp_query,
                                'docs': search_docs
                            })
                            search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
                            input_prompt += search_text
                        
                        one_step_think = get_think(output_text)
                        pred_answer = get_answer(output_text)
                        new_path.append({'think': one_step_think, 'answer': pred_answer})
                        
                        # Add new complete path
                        consistency_paths.append(new_path)
                        # ---------------------------------

                elif len(path) > 1: # process with search
                    search_indices = range(0, len(path)-1)
                    selected_indices = random.choices(search_indices, k=args.num_rag_generations)
                    selected_indices_group = [(x, selected_indices.count(x)) for x in sorted(set(selected_indices))]
                    
                    for (selected_index, repeat) in selected_indices_group:
                        original_sq = path[selected_index].get('search_query', '')
                        sq_prompt = sq_generator.get_instruction(original_sq, n=repeat)
                        sq_output = sq_generator.generate(sq_prompt, temperature=1.0)[0]
                        paraphrased_queries = get_paraphrased_query(sq_output)
                        
                        # check if paraphrased_queries are None
                        if paraphrased_queries == None:
                            print(f"Paraphrased queries are not provided for query {qid} ...")
                            for i in range(args.retry):
                                print(f"Think, try {i+1} ...")
                                sq_output = sq_generator.generate(sq_prompt, temperature=1.2)[0]
                                paraphrased_queries = get_paraphrased_query(sq_output)
                                if paraphrased_queries != None:
                                    break
                            else:
                                print(f"Failed to generate 'paraphrased queries' after all retries for query {qid}!!!")
                                paraphrased_queries = []
                        
                        for paraphrased_query in paraphrased_queries:
                            # Before break point
                            new_path = path[:selected_index]
                            
                            # On break point
                            retrieved_docs = retriever.search(paraphrased_query) if paraphrased_query else []
                            new_path.append({
                                "think": path[selected_index].get('think', ''),
                                "search_query": paraphrased_query,
                                "docs": retrieved_docs
                            })
                            
                            # After break point
                            input_prompt = prompt.format(question=query)
                            if tokenizer.chat_template:
                                input_prompt = tokenizer.apply_chat_template(
                                    [{"role": "user", "content": input_prompt}],
                                    add_generation_prompt=True, tokenize=False
                                )
                            for new_step in new_path:
                                input_prompt += f"\n\n<think>{new_step['think']}</think>\n"
                                input_prompt += f"<search>{new_step['search_query']}</search>\n"
                                input_prompt += f"<information>{_passages2string(retrieved_docs)}</information>\n\n"
                            
                            # ---- Generation ----------------
                            while True:
                                # print(input_prompt)
                                # print('------')
                                input_ids = tokenizer.encode(input_prompt, return_tensors='pt').to(device)
                                attention_mask = torch.ones_like(input_ids)
                                outputs = generator.generate(
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
                                tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
                                if tmp_query:
                                    search_docs = retriever.search(tmp_query)
                                    search_results = _passages2string(search_docs)
                                else:
                                    search_docs, search_results = [], ''

                                new_path.append({
                                    'think': get_think(output_text),
                                    'search_query': tmp_query,
                                    'docs': search_docs
                                })
                                search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
                                input_prompt += search_text
                            
                            one_step_think = get_think(output_text)
                            pred_answer = get_answer(output_text)
                            new_path.append({'think': one_step_think, 'answer': pred_answer})
                            
                            # Add new complete path
                            consistency_paths.append(new_path)
                            # ---------------------------------
                
                # --- Compute paths to confidence 
                path_answers = [path[-1].get('answer', '') for path in consistency_paths]
                
                # --- Save in file
                correctness_em = em_score(pred_answer, gt_answers)
                item = {
                    "qid": qid,
                    "query": query,
                    "gt_answers": gt_answers,
                    "pred_answer": pred_answer,
                    "correctness_em": correctness_em,
                    "consistency_answers": path_answers,
                }
                res_f.write(json.dumps(item) + "\n")
                # item2 = {
                #     "qid": qid,
                #     "query": query,
                #     "gt_answers": gt_answers,
                #     "pred_answer": pred_answer,
                #     "pred_answer": pred_answer,
                #     # "consistency_paths": consistency_paths
                # }
                # path_f.write(json.dumps(item2) + "\n")

def merge_result_files(args):
    results_shard_files = f"{args.output_dir}/rag_consistency_results_rank*.jsonl"
    paths_shard_files = f"{args.output_dir}/rag_consistency_paths_rank*.jsonl"

    results_shard_files = sorted(glob.glob(results_shard_files))
    with open(args.rag_consistency_results_file, "a") as fout:
        for shard_file in results_shard_files:
            if shard_file == args.rag_consistency_results_file:
                continue  # Don't append the output file to itself
            with open(shard_file, "r") as fin:
                for line in fin:
                    fout.write(line)
            os.remove(shard_file)
            print(f"Deleted shard file: {shard_file}")

    paths_shard_files = sorted(glob.glob(paths_shard_files))
    with open(args.rag_consistency_paths_file, "a") as fout:
        for shard_file in paths_shard_files:
            if shard_file == args.rag_consistency_paths_file:
                continue  # Don't append the output file to itself
            with open(shard_file, "r") as fin:
                for line in fin:
                    fout.write(line)
            os.remove(shard_file)
            print(f"Deleted shard file: {shard_file}")

def compute_uncertainty(args):
    
    def em_consistency(prediction, generated_texts):
        return sum(1 for item in generated_texts if item is not None and em_score(item, prediction)) / len(generated_texts)
    
    def group_candidates_by_answer(se_model, question:str, candidates: list[str]):
        """Return answer2candidates"""
        answer2candidates = {}
        for c in candidates:
            has_existed = False
            for existing_answer in answer2candidates.keys():
                if se_model.check_answers_equiv(question, c, existing_answer):
                    has_existed = True
                    answer2candidates[str(existing_answer)].append(c)
                    break

            if not has_existed:
                if str(c) in answer2candidates:
                    answer2candidates[str(c)].append(c)
                else:
                    answer2candidates[str(c)] = [c]
        return answer2candidates
    
    ptrue_model = transformers.AutoModelForCausalLM.from_pretrained(args.paraphrase_model_name_or_path, torch_dtype=torch.bfloat16).to(args.device)
    ptrue_tokenizer = transformers.AutoTokenizer.from_pretrained(args.paraphrase_model_name_or_path)
    # se_model = SemanticEquivalenceGenerator(args, ptrue_model, ptrue_tokenizer)
    
    bb_unc = BlackBoxUncertainty()
    ptrue_unc = PTrue(ptrue_model, ptrue_tokenizer)
    
    correlations = {}
    correctness = []
    uncertainties = {
        # "p_true": [],
        # "num_semantic_set": [],
        # "sum_eigen": [],
        # "eccentricity": [],
        "matrix_degree": [],
        # "em_conf": []
    }
    
    with open(args.rag_consistency_results_file, "r") as fin:
        for line in tqdm(fin):
            data = json.loads(line)
            question = data['query']
            gt_answers = data['gt_answers']
            generated_texts = data['consistency_answers']
            prediction = data['pred_answer']
            
            ### Correctness: most-likely
            correctness.append(data["correctness_em"])
            ### Correctness: MV
            # answer2candidates = group_candidates_by_answer(se_model, question, generated_texts)  
            # majority_answer = max(answer2candidates, key=lambda k: len(answer2candidates[k]))
            # correctness_em = em_score(majority_answer, gt_answers)
            # correctness.append(correctness_em)
            
            # print(question)
            # print(generated_texts)
            # print(sum_eigen_unc.calculate_uncertainty(question, generated_texts))
            # uncertainties['p_true'].append(ptrue_unc.ptrue_uncertainty(question, generated_texts, prediction))
            # uncertainties['num_semantic_set'].append(bb_unc.num_semantic_set_uncertainty(question, generated_texts))
            # uncertainties['sum_eigen'].append(bb_unc.sum_eigen_uncertainty(question, random.sample(generated_texts, 5)))
            # uncertainties['eccentricity'].append(bb_unc.eccentricity_uncertainty(question, random.sample(generated_texts, 5)))
            uncertainties['matrix_degree'].append(bb_unc.matrix_degree_uncertainty(question, generated_texts))
            # uncertainties['em_conf'].append(em_consistency(prediction, generated_texts))
            
    for ue_title, ue_value in uncertainties.items():
        correlations[ue_title] = roc_auc_score([1 - c for c in correctness], ue_value) #1-correctness
        # correlations[ue_title] = roc_auc_score(correctness, ue_value)
        
    print(correlations)
        
            
# def get_correlation(args):
#     correctness, confidence = [], []
#     with open(args.rag_consistency_results_file, "r") as fin:
#         for line in fin:
#             data = json.loads(line)
#             correctness.append(data["correctness_em"])
#             confidence.append(data["rag_conf"])
#     auroc = roc_auc_score(correctness, confidence)
#     print("AUROC:", auroc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model_name_or_path', type=str, default='PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo')
    parser.add_argument('--paraphrase_model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--max_new_token', type=int, default=1024)
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='popqa', choices=[
        'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=500.0)
    
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
    
    # RAG-consistency
    parser.add_argument("--num_rag_generations", type=int, default=10)
        
    # Others
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--run', type=str, default='run_4 (search_r1)')
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--retry", type=int, default=3)
    args = parser.parse_args()
    
    # === Files ====================
    args.output_dir = f"run_output/{args.run}" 
    model_ = args.model_name_or_path.split('/')[-1]
    args.output_dir = f"{args.output_dir}/{model_}/{args.dataset}_{args.subsec}/{args.retriever_name}"
    args.inference_results_file = f"{args.output_dir}/inference_results.jsonl"
    args.rag_consistency_results_file = f"{args.output_dir}/rag_consistency_results.jsonl"
    args.rag_consistency_paths_file = f"{args.output_dir}/rag_consistency_paths.jsonl"
    os.makedirs(args.output_dir, exist_ok=True)
    
    # === Prompt files ===========================
    args.semantic_equivalence_prompt_file = "prompts_mcts/semantic_equivalence_prompt_template.txt"    
    
    ### === Run Steps =============
    set_seed(args.seed)
    # searchr1_rag_consistency(args)
    # merge_result_files(args)
    compute_uncertainty(args)
    
    
    # python run_searchr1/rag_consistency.py
    # accelerate launch --multi_gpu run_searchr1/rag_consistency.py
    # accelerate launch --num_processes 1 run_searchr1/rag_consistency.py
    # accelerate launch run_searchr1/rag_consistency.py