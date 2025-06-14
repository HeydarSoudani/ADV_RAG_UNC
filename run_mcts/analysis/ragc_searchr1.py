#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
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
from run_rag_methods.src.rag_methods import *
from run_rag_methods.src.correctness import em_score
from run_mcts.src.models.semantic_equivalence import SemanticEquivalenceGenerator
from run_rag_methods.src.retrievers_local import BM25Retriever, ContrieverRetriever, RerankRetriever, DenseRetriever
from run_mcts.src.models.generate_paraphrase import (
    SearchQueryGenerator, ThinkGenerator,
    get_paraphrased_query, get_paraphrased_think
)

def get_auroc(correctness, confidence):
    try:
        auroc = roc_auc_score(correctness, confidence)
    except:
        print("Auroc couldn't be calculated because there is only one class. Returning 0.5 as auroc.")
        auroc = 0.5
    return auroc

class RAGConsistency:
    def __init__(self, device, args):
        self.args = args
        # === Static Retriever =====================
        if args.retriever_name == 'bm25':
            self.retriever = BM25Retriever(args)  
        elif args.retriever_name == 'contriever':
            self.retriever = ContrieverRetriever(args)
        elif args.retriever_name in ['rerank_l6', 'rerank_l12']:
            self.retriever = RerankRetriever(args)
        elif args.retriever_name in ['e5', 'bge']:
            self.retriever = DenseRetriever(args)
    
        # === RAG Generator: SearchR1 ===============
        self.rag_model = SearchR1_RAG(args, device)
        self.se_model = SemanticEquivalenceGenerator(args, device, self.rag_model.generator.generator, self.rag_model.generator.tokenizer)
        self.paraphrase_model = transformers.AutoModelForCausalLM.from_pretrained(args.paraphrase_model_name_or_path, torch_dtype=torch.bfloat16).to(device)
        self.paraphrase_tokenizer = transformers.AutoTokenizer.from_pretrained(args.paraphrase_model_name_or_path)
        self.search_query_generator = SearchQueryGenerator(args, self.paraphrase_model, self.paraphrase_tokenizer)
        self.think_generator = ThinkGenerator(args, self.paraphrase_model, self.paraphrase_tokenizer)
    
    def get_partial_answer(self, text):
        pattern = re.compile(r"(.*?)</answer>", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            return matches[-1]
        else:
            return None
    
    def get_masked_traces(self, qid, question, trace, repeats=5):
        masked_traces = []
        has_search = len(trace) > 1
        
        # print(trace)
        # print('------')
        
        if has_search:
            think_search_indices = range(0, len(trace)-1)
            selected_indices = random.choices(think_search_indices, k=repeats)
            selected_indices_group = [(x, selected_indices.count(x)) for x in sorted(set(selected_indices))]
            
            # print(selected_indices_group)
            for (selected_index, repeat) in selected_indices_group:
                
                #! Step 1: Generating paraphrased search queries 
                original_sq = trace[selected_index].get('search_query', '')
                
                sq_prompt = self.search_query_generator.get_instruction(original_sq, n=repeat)
                sq_output = self.search_query_generator.generate(sq_prompt, temperature=0.7)[0]
                paraphrased_queries = get_paraphrased_query(sq_output)
            
                # check if paraphrased_queries are None
                if paraphrased_queries == None:
                    print(f"Paraphrased queries are not provided for query {qid} ...")
                    for i in range(args.retry):
                        print(f"Paraphrased queries, try {i+1} ...")
                        sq_output = self.search_query_generator.generate(sq_prompt, temperature=1.0)[0]
                        paraphrased_queries = get_paraphrased_query(sq_output)
                        if paraphrased_queries != None:
                            break
                    else:
                        print(f"Failed to generate 'paraphrased queries' after all retries for query {qid}!!!")
                        paraphrased_queries = []
                
                # Check if the number of paraphrased_queries is equal to "repeat"
                if paraphrased_queries is None:
                    paraphrased_queries = []
                
                max_iterations = 10
                iteration = 0
                while len(paraphrased_queries) != repeat and iteration < max_iterations:
                    remaining = repeat - len(paraphrased_queries)        
                    extra_prompt = self.search_query_generator.get_instruction(original_sq, n=remaining)
                    extra_output = self.search_query_generator.generate(extra_prompt, temperature=1.0)[0]
                    extra_queries = get_paraphrased_query(extra_output)

                    if extra_queries:
                        paraphrased_queries.extend(extra_queries)
                        paraphrased_queries = paraphrased_queries[:repeat]  # trim if over
                    else:
                        print(f"Failed to generate extra queries on iteration {iteration + 1}")
                    iteration += 1
                if len(paraphrased_queries) != repeat:
                    print(f"Warning: Only generated {len(paraphrased_queries)} queries out of {repeat} after {iteration} iterations.")
                
                # print(original_sq)
                # print('--')
                # print(paraphrased_queries)
                # print('------')
                
                #! Step 2: Generating new masked traces
                for paraphrased_query in paraphrased_queries:
                    paraphrased_query = paraphrased_query.strip()
                    new_trace = []
                    
                    # Before break point: Keep steps excluding the selected one
                    new_trace = trace[:selected_index]
                    
                    # On break point
                    retrieved_docs = self.retriever.search(paraphrased_query) if paraphrased_query else []
                    new_trace.append({
                        "think": trace[selected_index].get('think', ''),
                        "search_query": paraphrased_query,
                        "docs": retrieved_docs,
                    })
                    
                    # After break point: ask searchR1 to generate
                    _, rest_of_trace = self.rag_model.inference_with_partial_trace(question, new_trace)
                    new_trace.extend(rest_of_trace)
                    
                    # print(new_trace)
                    # print("============")
                    # print("============")
                    
                    masked_traces.append(new_trace)
        
        else:
            ## Step 1: Generating paraphrased thinks
            original_think = trace[0].get('think', '')
            think_prompt = self.think_generator.get_instruction(original_think, n=repeats)
            think_output = self.think_generator.generate(think_prompt, temperature=0.7)[0]
            paraphrased_thinks = get_paraphrased_think(think_output)
            
            # check if paraphrased_thinks are None
            if paraphrased_thinks == None:
                print(f"Paraphrased thinks are not provided for query {qid} ...")
                for i in range(args.retry):
                    print(f"Paraphrased thinks, try {i+1} ...")
                    think_output = self.think_generator.generate(think_prompt, temperature=1.0)[0]
                    paraphrased_thinks = get_paraphrased_query(think_output)
                    if paraphrased_thinks != None:
                        break
                else:
                    print(f"Failed to generate 'paraphrased thinks' after all retries for query {qid}!!!")
                    paraphrased_thinks = []
        
            ## Step 2: Generating new masked traces
            input_prompt_text = self.rag_model.prompt.format(question=question)
            for pt in paraphrased_thinks:
                input_text_pt = input_prompt_text + f"<think> {pt} </think>\n<answer>"
                messages = [{"role": "user", "content": input_text_pt}]
                _, output_text = self.rag_model.generator.generate(messages, self.rag_model.generator.searchr1_stopping_criteria)
                pred_answer = self.get_partial_answer(output_text)
                new_trace = [{'think': pt, 'answer': pred_answer}]
                masked_traces.append(new_trace)
            
        return masked_traces
    
    def get_score(self, qid, question, prediction, trace, repeats=5):
        masked_traces = self.get_masked_traces(qid, question, trace, repeats)
        answer_list = [masked_trace[-1].get("answer", '') for masked_trace in masked_traces]
        num_consistent = sum(self.se_model.check_answers_equiv(question, prediction, ans) for ans in answer_list)
        return num_consistent / repeats, answer_list, masked_traces

def rag_consistency_score(args):
    # === MultiGPU setup =======================
    accelerator = Accelerator()
    device = accelerator.device
    if accelerator.is_main_process:
        if torch.cuda.is_available():
            print(f"Number of available GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA is not available. No GPUs detected.")
    
    # === Read generated samples ================
    query_ids = []
    rag_generations = {}
    if os.path.exists(args.inference_results_file):
        with open(args.inference_results_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'qid' in data:
                    query_ids.append(data['qid'])
                    rag_generations[data['qid']] = data
    sorted_query_ids = sorted(query_ids, key=lambda x: int(x.split('_')[1]))
    
    # === Read existing data ===================
    generated_qids = []
    if os.path.exists(args.consistency_results_file):
        with open(args.consistency_results_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'qid' in data:
                    generated_qids.append(data['qid'])
    generated_qids = set(generated_qids)
    filtered_sorted_query_ids = [id_ for id_ in sorted_query_ids if id_ not in generated_qids]
    
    # === Main Loop =============================
    ragc = RAGConsistency(device, args)
    accelerator.wait_for_everyone()
    with accelerator.split_between_processes(filtered_sorted_query_ids) as sorted_query_ids_shard:
        consistency_results_file_ranked = f"{args.output_dir}/consistency_results_th{args.hallucination_threshold}_rank{accelerator.process_index}.jsonl" \
            if args.rag_method in ['flare', 'dragin'] else f"{args.output_dir}/consistency_results_rank{accelerator.process_index}.jsonl"
        masked_traces_results_file_ranked = f"{args.output_dir}/masked_traces_results_th{args.hallucination_threshold}_rank{accelerator.process_index}.jsonl" \
            if args.rag_method in ['flare', 'dragin'] else f"{args.output_dir}/masked_traces_results_rank{accelerator.process_index}.jsonl"
        
        with open(consistency_results_file_ranked, 'w', encoding='utf-8') as cons_f, open(masked_traces_results_file_ranked, 'w', encoding='utf-8') as trace_f:
            for i, qid in enumerate(tqdm(sorted_query_ids_shard, desc=f"[Rank {accelerator.process_index}]")):
                # if i == 10:
                #     break
                sample = rag_generations[qid]
                user_query, prediction, trace = sample['query'], sample['pred_answer'], sample['path']
                consistency_score, answer_list, masked_traces = ragc.get_score(qid, user_query, prediction, trace, repeats=10)
                
                #! == Print 
                new_masked_traces = [
                    [
                        {
                            "think": step["think"],
                            "search_query": step["search_query"],
                            "docs": [{"id": doc["id"]} for doc in step["docs"]]
                        } if "docs" in step else step
                        for step in masked_trace
                    ]
                    for masked_trace in masked_traces
                ]
                
                cons_item = {
                    "qid": qid,
                    "query": user_query,
                    "gt_answers": sample['gt_answers'],
                    "pred_answer": prediction,
                    "em": sample['em'],
                    "consistency": consistency_score,
                    "answer_list": answer_list
                }
                trace_item = {
                    "qid": qid,
                    "query": user_query,
                    "masked_traces": new_masked_traces
                }
                cons_f.write(json.dumps(cons_item) + '\n')
                trace_f.write(json.dumps(trace_item) + '\n')

def merge_result_files(args):
    if args.rag_method in ['flare', 'dragin']:
        consistency_results_shard_file = f"{args.output_dir}/consistency_results_th{args.hallucination_threshold}_rank*.jsonl"
        masked_traces_results_shard_file = f"{args.output_dir}/masked_traces_results_th{args.hallucination_threshold}_rank*.jsonl"
    else:
        consistency_results_shard_file = f"{args.output_dir}/consistency_results_rank*.jsonl"
        masked_traces_results_shard_file = f"{args.output_dir}/masked_traces_results_rank*.jsonl"

    consistency_results_shard_file = sorted(glob.glob(consistency_results_shard_file))
    masked_traces_results_shard_file = sorted(glob.glob(masked_traces_results_shard_file))
    
    with open(args.consistency_results_file, "a") as fout:
        for shard_file in consistency_results_shard_file:
            if shard_file == args.consistency_results_file:
                continue
            with open(shard_file, "r") as fin:
                for line in fin:
                    fout.write(line)
            os.remove(shard_file)
            print(f"Deleted shard file: {shard_file}")
    
    with open(args.masked_traces_results_file, "a") as fout:
        for shard_file in masked_traces_results_shard_file:
            if shard_file == args.masked_traces_results_file:
                continue
            with open(shard_file, "r") as fin:
                for line in fin:
                    fout.write(line)
            os.remove(shard_file)
            print(f"Deleted shard file: {shard_file}")

def evaluation(args):
    correctness_list, consistency_list = [], []
    with open(args.consistency_results_file, 'r') as infile:
        for line in infile:
            data = json.loads(line)
            correctness_list.append(data['em'])
            consistency_list.append(data['consistency'])
            
            # Consistency with EM
            # answer_list, prediction = data['answer_list'], data['pred_answer']
            # num_answers = len(answer_list)
            # num_consistent = sum(em_score(ans, prediction) for ans in answer_list if ans != None)
            # consistency_list.append(num_consistent / num_answers)
            
    
    print("\nEvaluation Result:")    
    print(f"AUROC: {get_auroc(correctness_list, consistency_list)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model_name_or_path', type=str, default="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo")
    parser.add_argument('--paraphrase_model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
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
    parser.add_argument('--hallucination_threshold', type=float, default=0.08)                                 # for FLARE & DRAGIN
    parser.add_argument('--retrieve_keep_top_k', type=int, default=25)                                        # for DRAGIN
    parser.add_argument('--check_real_words', action='store_false')                                           # for DRAGIN
    parser.add_argument('--max_iter', type=int, default=5)
    
    # Others
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--run', type=str, default='run_4 (rag_methods_500)')
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
        args.consistency_results_file = f"{args.output_dir}/consistency_results_th{args.hallucination_threshold}.jsonl"
        args.masked_traces_results_file = f"{args.output_dir}/masked_traces_results_th{args.hallucination_threshold}.jsonl"
    else:
        args.inference_results_file = f"{args.output_dir}/inference_results.jsonl"
        args.consistency_results_file = f"{args.output_dir}/consistency_results.jsonl"
        args.masked_traces_results_file = f"{args.output_dir}/masked_traces_results.jsonl"
        
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    
    # === Prompt files =============
    args.query_decomposition_prompt_file = "run_mcts/prompts/query_decomposition_prompt_template.txt"
    args.semantic_equivalence_prompt_file = "run_mcts/prompts/semantic_equivalence_prompt_template.txt"
    
    ### === Run Steps ==============
    set_seed(args.seed)
    # rag_consistency_score(args)
    # merge_result_files(args)
    evaluation(args)
    
    # python run_mcts/analysis/ragc_searchr1.py
    # accelerate launch --multi_gpu run_mcts/analysis/ragc_searchr1.py
