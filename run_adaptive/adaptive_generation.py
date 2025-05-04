#!/usr/bin/env python3

import re
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import argparse
import datasets
from tqdm import tqdm


from src_adaptive.dataset import BaseDataset
from src_adaptive.rag import *
from src_adaptive.evaluate import CorrectnessEval
from utils.general_utils import set_seed


def adaptive_generation(args):
    print("\n== Adaptive Generation ...")
    print(f"""
        Model name:  {args.model_name_or_path}
        Dataset:     {args.dataset}/{args.subsec} ({args.fraction_of_data_to_use})
        RAG Method:  {args.rag_method} 
        Retriever:   {args.retriever_name}
        Hallu_thre:  {args.hallucination_threshold}
        Seed:        {args.seed}
    """.replace('        ', ''))
    
    
    # === Dataset ===============================
    dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', args.dataset)
    if 'test' in dataset:
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
    
    sample_index = 0
    print(f"Length of Dataset: {len(test_dataset)}")
    print(f"Dataset example {sample_index}:")
    print(f"Id:             {test_dataset[sample_index]['id']}")
    print(f"Question:       {test_dataset[sample_index]['question']}")
    print(f"Answers:        {test_dataset[sample_index]['golden_answers']}")


    # === Dataset & Metric Setup ================
    correctness = CorrectnessEval()

    # === Select RAG Method =====================
    if args.rag_method == "no_retrieval":
        model = NoRAG(args)
    elif args.rag_method == "single_retrieval":
        model = SingleRAG(args)
    elif args.rag_method in ["fix_length_retrieval", "fix_sentence_retrieval"]:
        model = FixLengthRAG(args)
    elif args.rag_method == 'flare':
        model = FLARE_RAG(args)
    elif args.rag_method == 'dragin':
        model = DRAGIN_RAG(args)
    else:
        raise NotImplementedError
    
    
    # === Generation ============================
    def get_answer(text):
        parts = text.split("the answer is", 1)  # Split at the first occurrence
        pred = parts[1].strip() if len(parts) > 1 else ""
        pattern = r"\.?</s>"
        pred = re.sub(pattern, "", pred)
        return pred
    
    correctness_res = {
        'EM': [],
        'F1': [],
        'Recall': [],
        'Precision': [],
    }
    os.makedirs(os.path.dirname(args.generations_output_file), exist_ok=True)
    
    with open(args.generations_output_file, 'w', encoding='utf-8') as g_file, open(args.generation_path_output_file, 'w', encoding='utf-8') as gp_file:
        for i, sample in enumerate(tqdm(test_dataset)):
            
            # if i == 5:
            #     break
            
            qid, question, gt_answers = sample['id'], sample['question'], sample['golden_answers']
            question = question.strip()
            if question[-1] != '?':
                question += '?'
            
            cot, n_halluc, gen_path = model.inference(question)
            cot = cot.strip()
            
            final_ans = ""
            if "the answer is" not in cot:
                tmp = model.reinference(question, cot).strip()  
                final_ans = get_answer(tmp) if "the answer is" in tmp else tmp
            else:
                final_ans = get_answer(cot)
            em_socre = correctness.exact_match_score(final_ans, gt_answers)
            f1_score = correctness.f1_score(final_ans, gt_answers)
                        
            g_item = {
                "qid": qid,
                "question": question,
                "em_score": em_socre['correct'],
                "f1_score": f1_score,
                "gold_answer": gt_answers,
                "pred": final_ans,
                "cot": cot,
                
            }
            g_file.write(json.dumps(g_item, ensure_ascii=False) + '\n')
            
            if args.rag_method in ['flare', 'dragin']:
                gp_item = {
                    "qid": qid,
                    "question": question,
                    "gold_answer": gt_answers,
                    "pred": final_ans,
                    "n_hallucination": n_halluc,
                    "generation_path": gen_path
                    
                }
                gp_file.write(json.dumps(gp_item, ensure_ascii=False) + '\n')
            
            correctness_res['EM'].append(em_socre['correct'])
            correctness_res['F1'].append(f1_score['f1'])
            correctness_res['Recall'].append(f1_score['recall'])
            correctness_res['Precision'].append(f1_score['precision'])


    # === Save results ==========================
    reuslts_dict = {
        'EM': np.mean(correctness_res['EM'])*100,
        'F1': np.mean(correctness_res['F1'])*100,
        'Recall': np.mean(correctness_res['Recall'])*100,
        'Precision': np.mean(correctness_res['Precision'])*100,
        'retrieve_count': model.counter.retrieve / len(dataset),
        'generate_count': model.counter.generate / len(dataset),
        'hallucinated_count': model.counter.hallucinated / len(dataset),
        'token_count': model.counter.token / len(dataset),
        'sentence_count': model.counter.sentence / len(dataset),
    }
    with open(args.results_output_file, 'w') as file:
        json.dump(reuslts_dict, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--max_new_token', type=int, default=128) # generate_max_length
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='bamboogle', choices=[
        'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
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
    
    # Others
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--run', type=str, default='run_11 (test_adaptive_rag)')
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument('--use_counter', action='store_false')
    
    # RAG setup
    parser.add_argument('--rag_method', type=str, default='dragin', choices=[
        'no_retrieval', 'single_retrieval',
        'fix_length_retrieval', 'fix_sentence_retrieval',
        'flare', 'dragin'
    ])
    parser.add_argument('--generate_fix_length', type=int, default=25)
    parser.add_argument('--modifier_method', type=str, default='token', choices=['token', 'entity'])          # for FLARE
    parser.add_argument('--query_formulation', type=str, default='real_words', choices=[                      # for FLARE & DRAGIN
        'direct', 'forward_all',
        'real_words', 'current', 'current_wo_wrong', 'last_sentence', 'last_n_tokens',
    ])
    parser.add_argument('--sentence_solver', type=str, default='avg', choices=['avg', 'max', 'min'])          # for FLARE
    parser.add_argument('--hallucination_threshold', type=float, default=2.0)                                # for FLARE & DRAGIN
    parser.add_argument('--retrieve_keep_top_k', type=int, default=25)                                        # for DRAGIN
    parser.add_argument('--check_real_words', action='store_false')                                           # for DRAGIN
    
    args = parser.parse_args()
    
    # === Files ====================
    model_ = args.model_name_or_path.split('/')[-1]
    if args.rag_method == 'no_retrieval':
        output_dir = f"run_output/{args.run}/{model_}/{args.dataset}_{args.subsec}/{args.rag_method}"
    else:
        output_dir = f"run_output/{args.run}/{model_}/{args.dataset}_{args.subsec}/{args.rag_method}_{args.retriever_name}"
    os.makedirs(output_dir, exist_ok=True)
    args.generations_output_file = f'{output_dir}/generations.jsonl'
    args.results_output_file = f'{output_dir}/results.json'
    args.generation_path_output_file = f'{output_dir}/generation_path.jsonl'
    

    ### === Define CUDA device =================== 
    args.device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. No GPUs detected.")
        
    
    ### === Run Steps ============================
    set_seed(args.seed)
    adaptive_generation(args)
    
    
    # python run_adaptive/adaptive_generation.py

