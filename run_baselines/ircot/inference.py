#!/usr/bin/env python3

import re
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import json
import glob
import torch
import requests
import datasets
import argparse
import numpy as np
import transformers
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import gather_object

from utils.general_utils import set_seed
from run_searchr1.correctness import em_score, f1_score
from run_searchr1.retrieval_local import BM25Retriever, ContrieverRetriever, RerankRetriever, DenseRetriever
from run_searchr1.inference import StopOnSequence


IRCOT_INSTRUCTION = 'You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. This task is illustrated through demonstrations, each consisting of a document set paired with a relevant question and its multi-hop reasoning thoughts. Your task is to generate one thought for current step, DON\'T generate the whole thoughts at once! If you reach what you believe to be the final step, start with "So the answer is:".'
IRCOT_EXAMPLE = "Wikipedia Title: Kurram Garhi\nKurram Garhi is a small village located near the city of Bannu, which is the part of Khyber Pakhtunkhwa province of Pakistan. Its population is approximately 35000. Barren hills are near this village. This village is on the border of Kurram Agency. Other nearby villages are Peppal, Surwangi and Amandi Kala.\n\nWikipedia Title: 2001–02 UEFA Champions League second group stage\nEight winners and eight runners- up from the first group stage were drawn into four groups of four teams, each containing two group winners and two runners- up. Teams from the same country or from the same first round group could not be drawn together. The top two teams in each group advanced to the quarter- finals.\n\nWikipedia Title: Satellite tournament\nA satellite tournament is either a minor tournament or event on a competitive sporting tour or one of a group of such tournaments that form a series played in the same country or region.\n\nWikipedia Title: Trojkrsti\nTrojkrsti is a village in Municipality of Prilep, Republic of Macedonia.\n\nWikipedia Title: Telephone numbers in Ascension Island\nCountry Code:+ 247< br> International Call Prefix: 00 Ascension Island does not share the same country code( +290) with the rest of St Helena.\n\nQuestion: Are both Kurram Garhi and Trojkrsti located in the same country?\nThought: Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country. So the answer is: no.\n\n"

def passages2string(retrieval_result):
    format_reference = ''
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item['contents']
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Wikipedia Title: {title}\n{text}\n\n"
    return format_reference

def get_answer(text):
    parts = text.split("the answer is: ", 1)  # Split at the first occurrence
    pred = parts[1].strip() if len(parts) > 1 else ""
    pattern = r"\.?</s>"
    pred = re.sub(pattern, "", pred)
    return pred
    

def ircot_inference(args):
    # === MultiGPU setup =======================
    accelerator = Accelerator()
    device = accelerator.device
    
    if accelerator.is_main_process:
        print("\n== Search R1 Inference ...")
        print(f"""
            Model name:  {args.model_name_or_path}
            Dataset:     {args.dataset} / {args.subsec} ({args.fraction_of_data_to_use})
            Retriever:   {args.retriever_name} / ({args.retrieval_model_path})
            Seed:        {args.seed}
            Run:         {args.run}
        """.replace('        ', ''))
    
        # === Define CUDA device =======
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
    
    # === generator Model ======================
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16).to(device)
    target_sequences = [".", " .", ".\n", " .\n", ".\n\n", " .\n\n", "\n", " \n", "\n\n", " \n\n"]
    stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])
    curr_eos = [151645, 151643] # for Qwen2.5 series models

    # === Static Retriever =====================
    if args.retriever_name == 'bm25':
        retriever = BM25Retriever(args)  
    elif args.retriever_name == 'contriever':
        retriever = ContrieverRetriever(args)
    elif args.retriever_name in ['rerank_l6', 'rerank_l12']:
        retriever = RerankRetriever(args)
    elif args.retriever_name in ['e5', 'bge']:
        retriever = DenseRetriever(args)    

    # === Prompt ===============================
    # TODO: prompt
    system_prompt = f"{IRCOT_INSTRUCTION}\n\n{IRCOT_EXAMPLE}"
    user_prompt = "{documents}Question: {question}\nThought:"

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

    # === Functions ============================
    def generate(chat):
        if tokenizer.chat_template:
            input_prompt = tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False
            )
        
        input_ids = tokenizer.encode(input_prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_token,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )
        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return outputs, output_text

    # === Inference ============================
    em_evaluation = generated_em
    accelerator.wait_for_everyone()
    with accelerator.split_between_processes(filtered_dataset) as test_dataset_shard:
        inference_results_file_ranked = f"{args.output_dir}/inference_results_rank{accelerator.process_index}.jsonl"
        with open(inference_results_file_ranked, "w") as res_f:
            for i, sample in enumerate(tqdm(test_dataset_shard, desc=f"[Rank {accelerator.process_index}]")):
                if i == 10:
                    break
                qid, question, gt_answers = sample['id'], sample['question'], sample['golden_answers']
                question = question.strip()
                if question[-1] != '?':
                    question += '?'
                
                # initial retrieval
                cur_search_docs = retriever.search(question)
                input_chat = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt.format(documents=passages2string(cur_search_docs), question=question)}
                ]

                path, iter_num = [], 0
                while iter_num < args.max_iter:
                    outputs, output_text = generate(input_chat)
                    path.append({'thought': output_text, 'docs': cur_search_docs})
                    
                    if outputs[0][-1].item() in curr_eos or ("So the answer is:" in output_text):
                        break
                    iter_num += 1
                    if iter_num == args.max_iter:
                        break  # don't perform another retrieval or prompt construction
                    
                    cur_search_docs = retriever.search(output_text) if output_text else []
                    tmp = [doc for step in path for doc in step['docs']] + cur_search_docs
                    input_chat = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt.format(documents=passages2string(tmp), question=question)},
                        {"role": "assistant", "content": ' '.join([step['thought'] for step in path])}
                    ]
            
                # Regenerate the last sentence if it is needed
                if "So the answer is:" not in output_text:
                    input_chat = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt.format(documents=passages2string([doc for step in path for doc in step['docs']]), question=question)},
                        {"role": "assistant", "content": ' '.join([step['thought'] for step in path]) + " So the answer is:"}
                    ]                    
                    outputs, output_text = generate(input_chat)
                    path.append({'think': output_text})
                
                pred_answer = get_answer(output_text)
                print(gt_answers)
                print(pred_answer)
                print('-----')
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
                res_f.write(json.dumps(item) + "\n")
                em_evaluation.append(correctness_em)
                
    em_evaluation_gathered = gather_object(em_evaluation)
    if accelerator.is_main_process:
        print("\nEvaluation Result:")
        print(f"EM: {np.mean(em_evaluation_gathered)*100}")


def merge_result_files(args):
    results_shard_files = f"{args.output_dir}/inference_results_rank*.jsonl"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--max_new_token', type=int, default=1024)
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='bamboogle', choices=[
        'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
    
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
    
    # IRCoT setup
    parser.add_argument('--max_iter', type=int, default=5)
    
    # Others
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--run', type=str, default='run_5 (ircot_2k)')
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--retry", type=int, default=3)
    
    args = parser.parse_args()
    
    # === Files ====================
    args.output_dir = f"run_output/{args.run}" 
    model_ = args.model_name_or_path.split('/')[-1]
    args.output_dir = f"{args.output_dir}/{model_}/{args.dataset}_{args.subsec}/{args.retriever_name}"
    args.inference_results_file = f"{args.output_dir}/inference_results.jsonl"
    os.makedirs(args.output_dir, exist_ok=True)
        
    ### === Run Steps =============
    set_seed(args.seed)
    ircot_inference(args)
    # merge_result_files(args)
    
    # python run_baselines/ircot/inference.py
    # accelerate launch --multi_gpu run_baselines/ircot/inference.py

