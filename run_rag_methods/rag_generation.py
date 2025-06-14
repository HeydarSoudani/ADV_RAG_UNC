#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import json
import glob
import torch
import datasets
import argparse
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import gather_object

from run_rag_methods.src.rag_methods import *
from run_rag_methods.src.correctness import em_score, subem_score, f1_score
from utils.general_utils import set_seed, sample_sorted_qids, extract_qid_number


def get_answer(text):
    parts = text.split("the answer is", 1)  # Split at the first occurrence
    pred = parts[1].strip() if len(parts) > 1 else ""
    pattern = r"\.?</s>"
    pred = re.sub(pattern, "", pred)
    return pred

def rag_generation(args):
    # === MultiGPU setup ========================
    accelerator = Accelerator()
    device = accelerator.device
    if accelerator.is_main_process:
        print("\n== RAG Generation ...")
        print(f"""
            Model name:  {args.model_name_or_path}
            Dataset:     {args.dataset} / {args.subsec} ({args.fraction_of_data_to_use})
            RAG Method:  {args.rag_method}
            Retriever:   {args.retriever_name} / ({args.retrieval_model_path})
            Hallu_thre:  {args.hallucination_threshold}
            Seed:        {args.seed}
            Run:         {args.run}
        """.replace('        ', ''))
        # --- Define CUDA device
        # args.device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"Number of available GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA is not available. No GPUs detected.")
        print('\n')
    
    
    # === Dataset ===============================
    dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', args.dataset)
    if args.subsec == 'train' and 'train' in dataset:
        print(f'Using the {args.dataset} train dataset...')
        test_dataset_ = dataset['train']
    elif 'test' in dataset:
        print(f'Using the {args.dataset} test dataset...')
        test_dataset_ = dataset['test']
    elif 'dev' in dataset:
        print(f'Using the {args.dataset} dev dataset...')
        test_dataset_ = dataset['dev']
    
    if args.fraction_of_data_to_use < 1.0:
        shuffled_dataset = test_dataset_.shuffle(seed=args.seed)
        num_samples = int(args.fraction_of_data_to_use * len(shuffled_dataset))
        test_dataset = shuffled_dataset.select(range(num_samples))
    elif args.fraction_of_data_to_use > 1.0:
        shuffled_dataset = test_dataset_.shuffle(seed=args.seed)
        test_dataset = shuffled_dataset.select(range(int(args.fraction_of_data_to_use)))
    else:
        test_dataset = test_dataset_
    
    if accelerator.is_main_process:
        sample_index = 0
        print(f"Length of Dataset: {len(test_dataset)}")
        print(f"Dataset example {sample_index}:")
        print(f"Id:             {test_dataset[sample_index]['id']}")
        print(f"Question:       {test_dataset[sample_index]['question']}")
        print(f"Answers:        {test_dataset[sample_index]['golden_answers']}")


    # === Read existing data ===================
    generated_qids = []
    generated_em = []
    if os.path.exists(args.inference_results_file):
        with open(args.inference_results_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'qid' in data:
                    generated_qids.append(data['qid'])
                    generated_em.append(data['em'])
    generated_qids = set(generated_qids)
    filtered_dataset = test_dataset.filter(lambda example: example['id'] not in generated_qids)


    # === Select RAG Method =====================
    if args.rag_method == 'direct_inference':
        rag_model = DirectInference(args, device)
    elif args.rag_method == 'cot_inference':
        rag_model = CoTInference(args, device)
    elif args.rag_method == "cot_single_retrieval":
        rag_model = CoTSingleRAG(args, device)
    elif args.rag_method == "fix_sentence_retrieval":
        rag_model = FixSentenceRAG(args, device)
    elif args.rag_method == "fix_length_retrieval":
        rag_model = FixLengthRAG(args, device)
    elif args.rag_method == 'ircot':
        rag_model = IRCOT_RAG(args, device)
    elif args.rag_method == 'flare':
        rag_model = FLARE_RAG_V1(args, device)
    elif args.rag_method == 'dragin':
        rag_model = DRAGIN_RAG(args, device)
    elif args.rag_method == 'self_ask':
        rag_model = SelfAsk_RAG(args, device)
    elif args.rag_method == 'react':
        rag_model = ReAct_RAG(args, device)
    elif args.rag_method == 'search_o1':
        rag_model = SearchO1_RAG(args, device)
    elif args.rag_method == 'search_r1':
        rag_model = SearchR1_RAG(args, device)
    else:
        raise NotImplementedError


    # === Inference ============================
    em_evaluation = generated_em
    accelerator.wait_for_everyone()
    with accelerator.split_between_processes(filtered_dataset) as test_dataset_shard:
        if args.rag_method in ['flare', 'dragin']:
            inference_results_file_ranked = f"{args.output_dir}/inference_results_th{args.hallucination_threshold}_rank{accelerator.process_index}.jsonl"
        else:
            inference_results_file_ranked = f"{args.output_dir}/inference_results_rank{accelerator.process_index}.jsonl"
        with open(inference_results_file_ranked, 'w') as res_f:
            for i, sample in enumerate(tqdm(test_dataset_shard, desc=f"[Rank {accelerator.process_index}]")):
                if i == 1:
                    break
                qid, question, gt_answers = sample['id'], sample['question'], sample['golden_answers']
                question = question.strip()
                if question[-1] != '?':
                    question += '?'
                
                pred_answer, path = rag_model.inference(question)
                if pred_answer:
                    correctness_em = em_score(pred_answer, gt_answers)
                    correctness_f1 = f1_score(pred_answer, gt_answers)
                else:
                    correctness_em = 0
                    correctness_f1 = {'f1': 0, 'precision': 0, 'recall': 0}
                
                item = {
                    "qid": qid,
                    "query": question,
                    "gt_answers": gt_answers,
                    "pred_answer": pred_answer,
                    "em": correctness_em,
                    "f1": correctness_f1,
                    "path": path
                }
                res_f.write(json.dumps(item) + '\n')
                em_evaluation.append(correctness_em)

    em_evaluation_gathered = gather_object(em_evaluation)
    if accelerator.is_main_process:
        print("\nEvaluation Result:")
        print(f"EM: {np.mean(em_evaluation_gathered)*100}")

def merge_result_files(args):
    if args.rag_method in ['flare', 'dragin']:
        results_shard_files = f"{args.output_dir}/inference_results_th{args.hallucination_threshold}_rank*.jsonl"
    else:
        results_shard_files = f"{args.output_dir}/inference_results_rank*.jsonl"
    
    # results_shard_files = f"{args.output_dir}/inference_results_rank*.jsonl"
    results_shard_files = sorted(glob.glob(results_shard_files))
    with open(args.inference_results_file, "a") as fout:
        for shard_file in results_shard_files:
            if shard_file == args.inference_results_file:
                continue
            with open(shard_file, "r") as fin:
                for line in fin:
                    fout.write(line)
            os.remove(shard_file)
            print(f"Deleted shard file: {shard_file}")

def get_num_retrieval(args):
    all_ret = []
    with open(args.inference_results_file, 'r') as infile:
        for line in infile:
            path = json.loads(line)['path']
            all_ret.append(len(path)-1)
    print(f"\n# of retrieval: {np.mean(all_ret)}")
    
    # TODO: different for FLARE and DRAGIN, check whether the search_query empty

def evaluate(args):
    em_full_evaluation, em_sub_evaluation = [], []
    with open(args.inference_results_file, 'r') as infile:
        for line in infile:
            data = json.loads(line)
            gt_answers = data['gt_answers']
            pred_answer = data['pred_answer']
            if pred_answer:
                em_socre_full = em_score(pred_answer, gt_answers)
                em_socre_sum = subem_score(pred_answer, gt_answers)
            else:
                em_socre_full = 0
                em_socre_sum = 0
            em_full_evaluation.append(em_socre_full)
            em_sub_evaluation.append(em_socre_sum)
    
    # === Print results ========================
    print("\nEvaluation Result:")
    print(f"EM (full): {np.mean(em_full_evaluation)*100}")
    print(f"EM (sub): {np.mean(em_sub_evaluation)*100}")

def subsample_generation(args):
    
    def get_all_qids_from_jsonl(jsonl_file):
        qids = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.search(r'"qid"\s*:\s*"([^"]+)"', line)
                if match:
                    qids.append(match.group(1))
        return qids

    sample_size = 500
    src_file = args.inference_results_file
    
    # Subsampling qids
    all_qids = get_all_qids_from_jsonl(src_file)
    sampled_qids = sample_sorted_qids(all_qids, sample_size=sample_size)
    qid_set = set(sampled_qids)
    
    # dst file
    model_ = args.model_name_or_path.split('/')[-1]
    run_ = f"run_4 (rag_methods_{sample_size})"
    dst_output_dir = f"run_output/{run_}/{model_}/{args.dataset}_{args.subsec}/{args.rag_method}" \
        if args.rag_method in ['direct_inference', 'cot_inference'] \
        else f"run_output/{run_}/{model_}/{args.dataset}_{args.subsec}/{args.rag_method}_{args.retriever_name}"
    os.makedirs(dst_output_dir, exist_ok=True)
    dst_inference_results_file = f"{dst_output_dir}/inference_results_th{args.hallucination_threshold}.jsonl" \
        if args.rag_method in ['flare', 'dragin'] \
        else f"{dst_output_dir}/inference_results.jsonl"
    
    matched_data = []
    with open(src_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            data = json.loads(line)
            qid = data.get('qid', '')
            if qid in qid_set:
                matched_data.append(data)
    matched_data.sort(key=lambda item: extract_qid_number(item.get('qid', '')))
    
    with open(dst_inference_results_file, 'w', encoding='utf-8') as fout:
        for item in matched_data:
            fout.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model
    # parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--model_name_or_path', type=str, default="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo")
    parser.add_argument('--max_new_tokens', type=int, default=128)
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='hotpotqa', choices=[
        'nq', 'triviaqa', 'popqa', '2wikimultihopqa', 'hotpotqa', 'musique', 'bamboogle'
    ])
    parser.add_argument('--subsec', type=str, default='dev', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=2000.0)
    parser.add_argument("--enable_fewshot_examples", action="store_true", help="")
    parser.add_argument('--fewshot', type=int, default=6)
    
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
    parser.add_argument('--retrieval_query_max_length', type=int, default=64)
    parser.add_argument('--retrieval_use_fp16', action='store_false', help='')
    parser.add_argument('--retrieval_batch_size', type=int, default=512)
    parser.add_argument("--bm25_k1", type=float, default=0.9)
    parser.add_argument("--bm25_b", type=float, default=0.4)
    
    # RAG setup
    parser.add_argument('--rag_method', type=str, default='search_r1', choices=[
        'direct_inference', 'cot_inference', 'cot_single_retrieval',
        'fix_length_retrieval', 'fix_sentence_retrieval', 'ircot',
        'flare', 'dragin',
        'self_ask', 'react', 'search_o1', 'search_r1'
    ])
    parser.add_argument('--generate_fix_length', type=int, default=25)
    parser.add_argument('--modifier_method', type=str, default='token', choices=['token', 'entity'])          # for FLARE
    parser.add_argument('--query_formulation', type=str, default='real_words', choices=[                      # for FLARE & DRAGIN
        'direct', 'forward_all',
        'real_words', 'current', 'current_wo_wrong', 'last_sentence', 'last_n_tokens',
    ])
    parser.add_argument('--sentence_solver', type=str, default='avg', choices=['avg', 'max', 'min'])          # for FLARE
    parser.add_argument('--hallucination_threshold', type=float, default=0.08)                                # for FLARE & DRAGIN
    parser.add_argument('--retrieve_keep_top_k', type=int, default=25)                                        # for DRAGIN
    parser.add_argument('--check_real_words', action='store_false')                                           # for DRAGIN
    parser.add_argument('--max_iter', type=int, default=5)
    
    # Others
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--run', type=str, default='run_1 (rag_methods_2k)')
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument('--use_counter', action='store_false')
    
    args = parser.parse_args()
    
    # === Files ====================
    model_ = args.model_name_or_path.split('/')[-1]
    if args.rag_method in ['direct_inference', 'cot_inference']:
        args.output_dir = f"run_output/{args.run}/{model_}/{args.dataset}_{args.subsec}/{args.rag_method}"
    else:
        args.output_dir = f"run_output/{args.run}/{model_}/{args.dataset}_{args.subsec}/{args.rag_method}_{args.retriever_name}"
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.rag_method in ['flare', 'dragin']:
        args.inference_results_file = f"{args.output_dir}/inference_results_th{args.hallucination_threshold}.jsonl"
    else:
        args.inference_results_file = f"{args.output_dir}/inference_results.jsonl"
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    

    ### === Run Steps ============================
    set_seed(args.seed)
    # rag_generation(args)
    # merge_result_files(args)
    # get_num_retrieval(args)
    # evaluate(args)
    subsample_generation(args)
        
    # python run_rag_methods/rag_generation.py
    # accelerate launch --multi_gpu run_rag_methods/rag_generation.py
    # accelerate launch --multi_gpu --num_processes 2 run_rag_methods/rag_generation.py
    # accelerate launch --num_processes 1 run_rag_methods/rag_generation.py


















### === Define CUDA device =================== 
# args.device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
# if torch.cuda.is_available():
#     print(f"Number of available GPUs: {torch.cuda.device_count()}")
#     for i in range(torch.cuda.device_count()):
#         print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
# else:
#     print("CUDA is not available. No GPUs detected.")
        
# print(pred_answer)
# print('----')
# print(path)
# print('----')

# cot, n_halluc, gen_path, stat = model.inference(question)
# cot = cot.strip()
# pred_answer = ""
# if "the answer is" not in cot:
#     tmp = model.reinference(question, cot).strip()  
#     pred_answer = get_answer(tmp) if "the answer is" in tmp else tmp
# else:
#     pred_answer = get_answer(cot)
# "path": cot,
# "stat": stat,
# "n_hallucination": n_halluc,
# "generation_path": gen_path,



# def subsample_generation(args):
    
#     def extract_qid_number(line):
#         match = re.search(r'"qid"\s*:\s*"[^_]*_(\d+)"', line)
#         return int(match.group(1)) if match else float('inf')
    
#     sample_size = 500
#     src_file = args.inference_results_file
    
#     model_ = args.model_name_or_path.split('/')[-1]
#     run_ = "run_4 (rag_methods_500)"
#     dst_output_dir = f"run_output/{run_}/{model_}/{args.dataset}_{args.subsec}/{args.rag_method}" \
#         if args.rag_method in ['direct_inference', 'cot_inference'] \
#         else f"run_output/{run_}/{model_}/{args.dataset}_{args.subsec}/{args.rag_method}_{args.retriever_name}"
#     os.makedirs(dst_output_dir, exist_ok=True)
    
#     dst_inference_results_file = f"{dst_output_dir}/inference_results_th{args.hallucination_threshold}.jsonl" \
#         if args.rag_method in ['flare', 'dragin'] \
#         else f"{dst_output_dir}/inference_results.jsonl"
    

#     with open(src_file, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#     lines.sort(key=extract_qid_number)

#     if len(lines) < sample_size:
#         raise ValueError(f"Input file only has {len(lines)} lines, but {sample_size} are required.")
    
#     sampled_lines = random.sample(lines, sample_size)
#     sampled_lines.sort(key=extract_qid_number)
#     with open(dst_inference_results_file, 'w', encoding='utf-8') as f:
#         for line in sampled_lines:
#             f.write(line)

