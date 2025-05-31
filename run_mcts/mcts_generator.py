#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import shutil
import random
import argparse
import datasets
import transformers
from tqdm import tqdm, trange
from accelerate import Accelerator

from utils.general_utils import set_seed
from run_rag_methods.src.retrievers_local import BM25Retriever, ContrieverRetriever, RerankRetriever, DenseRetriever
from run_mcts.src.generate_node import Generator
from run_mcts.src.MCTS_backbone import MCTS_Searcher
from run_mcts.src.MCTS_reasoning import Reasoning_MCTS_Node
from utils.mcts_utils import Node_Type, stochastic_find_best_solution, print_tree_from_root


def mcts_generation(args):
    # === MultiGPU setup =======================
    accelerator = Accelerator()
    device = accelerator.device
    
    if accelerator.is_main_process:
        print("\n== MCTS Generation ...")
        print(f"""
            Model name:  {args.model_name_or_path}
            Dataset:     {args.dataset} / {args.subsec} ({args.fraction_of_data_to_use})
            Retriever:   {args.retriever_name} / ({args.retrieval_model_path})
            Rollouts:    {args.num_rollouts} ({args.max_depth_allowed})
            Seed:        {args.seed}
            Run:         {args.run}
        """.replace('        ', ''))
        
        # --- Define CUDA device
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
    else:
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
    
    if accelerator.is_main_process:
        sample_index = 0
        print(f"Length of Dataset: {len(test_dataset)}")
        print(f"Dataset example {sample_index}:")
        print(f"Id:             {test_dataset[sample_index]['id']}")
        print(f"Question:       {test_dataset[sample_index]['question']}")
        print(f"Answers:        {test_dataset[sample_index]['golden_answers']}")

    
    # === Static Retriever ===================== 
    if args.retriever_name == 'bm25':
        retriever = BM25Retriever(args)  
    elif args.retriever_name == 'contriever':
        retriever = ContrieverRetriever(args)
    elif args.retriever_name in ['rerank_l6', 'rerank_l12']:
        retriever = RerankRetriever(args)
    elif args.retriever_name in ['e5', 'bge', 'reasonir']:
        retriever = DenseRetriever(args)
    
    
    # === LLM Generator ==========================
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    generator = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16).to(device)
    node_generator = Generator(args, retriever, generator, tokenizer)
    
    
    # === Read existing data =====================
    challenging_samples = ['test_24', 'test_27', 'test_47', 'test_52', 'test_64', 'test_69', 'test_73', 'test_74', 'test_83']
    generated_qids = [name for name in os.listdir(args.generation_trees_results_dir) if os.path.isdir(os.path.join(args.generation_trees_results_dir, name))]
    filtered_dataset = test_dataset.filter(lambda example: example['id'] not in generated_qids)
    
    
    # === Generation =============================
    accelerator.wait_for_everyone()
    with accelerator.split_between_processes(filtered_dataset) as test_dataset_shard:
        for i, sample in enumerate(tqdm(test_dataset_shard, desc=f"[Rank {accelerator.process_index}]")):
            # if i == 3:
            #     break
            qid, question, gt_answers = sample['id'], sample['question'], sample['golden_answers']
            print(f"[Rank {accelerator.process_index}] Generating MCTS for query {qid} ...")
            question = question.strip()
            if question[-1] != '?':
                question += '?'
            
            #! build an MCTS searcher
            mcts_searcher = MCTS_Searcher(
                exploration_weight=args.mcts_exploration_weight,
                weight_scheduler=args.mcts_weight_scheduler,
                num_rollouts=args.num_rollouts,
                discount=args.mcts_discount_factor,
                verbose=args.verbose,
            )
            #! build the MCTS tree
            root_node = Reasoning_MCTS_Node(
                parent=None,
                depth=0,
                node_type=Node_Type.USER_QUESTION,
                verbose=args.verbose,
                generator=node_generator,
                question_id=qid,
                user_question=question,
                gt_answer=gt_answers,
                gt_reasoning_steps=[],
                answer_candidates=[],
                max_depth_allowed=args.max_depth_allowed,
                enable_potential_score=args.enable_potential_score,
                enable_critique=args.enable_critique,
                enable_doc_generation=args.enable_doc_generation,
            )
            
            model_solutions = []
            model_all_solutions = []
            model_rollout_nodes = []
            for i in (pbar := trange(args.num_rollouts, disable=True, position=0)):
                rollout_node = mcts_searcher.do_rollout(root_node, i)
                model_rollout_nodes.append(rollout_node)

                all_solution_nodes, all_solutions = stochastic_find_best_solution(
                    root_node, enable_potential_score=args.enable_potential_score
                )
                model_all_solutions.append(all_solutions)

                os.makedirs(f"{args.generation_trees_results_dir}/{qid}", exist_ok=True)
                with open(f"{args.generation_trees_results_dir}/{qid}/rollout_{i}.tree", "w", encoding='utf-8') as f:
                    print_tree_from_root(
                        mcts_searcher=mcts_searcher,
                        rollout_id=i,
                        root_node=root_node,
                        chosen_node=None,
                        file=f,
                    )

            #! record final traces
            js = [{"trace": node.solution_trace, "rollout_id": node.rollout_id} for node in all_solution_nodes]
            with open(f"{args.generation_trees_results_dir}/{qid}/final_solutions.jsonl", "w") as f:
                for item in js:
                    f.write(json.dumps(item) + "\n")
            js2 = [{"trace": node.solution_trace, "rollout_id": i} for i, node in enumerate(model_rollout_nodes)]
            with open(f"{args.generation_trees_results_dir}/{qid}/rollout_solutions.jsonl", "w") as f:
                for item in js2:
                    f.write(json.dumps(item) + "\n")

            #! record conf analysis 
            # un_object = {}
            # for node in all_solution_nodes:
            #     depth = len(node.solution_trace) -1
            #     # all_keys = list(node.solution_trace.keys())
            #     # print(all_keys)
            #     last_key = max(list(node.solution_trace.keys()))
            #     # print(node.solution_trace)
            #     un_object[depth] = {
                    
            #         "ue_param": node.solution_trace[last_key]["think_answer"]["scores"][1]["param"]["PE"]["uncertainty"],
            #         "ue_cont": node.solution_trace[last_key]["think_answer"]["scores"][1]["cont"]["PE"]["uncertainty"],
            #         "prediction": node.solution_trace[last_key]["think_answer"]["answer"],
            #         "reward": node.solution_trace[last_key]["think_answer"]["node_reward"]
            #     }
            # # print(un_object)
            # sorted_data = dict(sorted(un_object.items(), key=lambda item: int(item[0])))
            # print('-' * 40)
            # print(f'Gt: {all_solution_nodes[0].solution_trace[0]['ground_truth']}')
            # for key, value in sorted_data.items():
            #     print(f"D {key} | {value['ue_param']:.3f} | {value['ue_cont']:.3f} | {value['reward']:.3f} | {value['prediction']}")
            # print('-' * 40)


    # # === Save results ===========================
    # reuslts_dict = {
    #     'Tokens': node_generator.counter.token / len(dataset),
    #     'Sentences': node_generator.counter.sentence / len(dataset),
    #     'Retrieves': node_generator.counter.retrieve / len(dataset)
    # }
    # with open(args.statistics_results_file, 'w') as file:
    #     json.dump(reuslts_dict, file, indent=4)

def subsample_generation(args):
    sample_size = 500
    src_dir = args.generation_trees_results_dir
    
    # === Output file 
    run_ = "run_4 (mcts_500_rollout4)"
    # run_ = "run_5 (mcts_500_rollout8)"
    model_ = args.model_name_or_path.split('/')[-1]
    output_dir = f"run_output/{run_}/{model_}/{args.dataset}_{args.subsec}/{args.retriever_name}"
    dst_dir = f'{output_dir}/generation_trees'
    os.makedirs(dst_dir, exist_ok=True)

    # 
    folders = [f for f in os.listdir(src_dir)]
    sampled_folders = random.sample(folders, sample_size)
    
    for folder in tqdm(sampled_folders):
        src_path = os.path.join(src_dir, folder)
        dst_path = os.path.join(dst_dir, folder)
        shutil.copytree(src_path, dst_path)
        # print(f"Copied {folder} to {dst_dir}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--max_new_token', type=int, default=1024)
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='nq', choices=[
        'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'
    ])
    parser.add_argument('--subsec', type=str, default='test', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=2000.0)
    parser.add_argument("--enable_fewshot_examples", action="store_true", help="")
    
    # Retriever
    parser.add_argument('--retriever_name', type=str, default='rerank_l6', choices=[
        'bm25', 'contriever', 'rerank_l6', 'rerank_l12', 'e5', 'bge', 'reasonir'
    ])
    parser.add_argument('--corpus_path', type=str, default='data/search_r1_files/wiki-18.jsonl')
    parser.add_argument('--index_path', type=str, default='data/search_r1_files/bm25', choices=[
        'data/search_r1_files/bm25',          # For BM25 & Rerank
        'data/search_r1_files/e5_Flat.index', # For E5
        'data/search_r1_files/reasonir_Flat.index', # For ReasonIR
    ])
    parser.add_argument("--retrieval_model_path", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2", choices=[
        "cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-MiniLM-L12-v2", # For Rerank
        "intfloat/e5-base-v2",  # For E5
        "reasonir/ReasonIR-8B", # For ReasonIR
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
    parser.add_argument('--run', type=str, default='run_2 (mcts_2k_rollout4)')
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument('--use_counter', action='store_false')
    
    # MCTS ---
    parser.add_argument('--discriminator_method', type=str, default='reasoning_consistency', choices=[
        'majority_voting', 'best_of_n', 'reasoning_consistency', 'rag_consistency', 'llm_selector'
    ])
    parser.add_argument("--enable_critique", action="store_true", help="")
    parser.add_argument("--enable_doc_generation", action="store_true", help="")
    parser.add_argument("--verbose", action="store_true", help="extra login")
    parser.add_argument("--mcts_discount_factor", type=float, default=1.0)
    parser.add_argument("--mcts_exploration_weight", type=float, default=2.0)
    parser.add_argument("--mcts_weight_scheduler", choices=["exp", "lin", "const"], default="const")
    parser.add_argument("--save_tree", action="store_true")
    parser.add_argument("--num_rollouts", type=int, default=4)
    parser.add_argument("--max_depth_allowed", type=int, default=6)
    parser.add_argument("--num_votes", type=int, default=2)
    parser.add_argument("--mcts_num_last_votes", type=int, default=2)
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
    args.output_dir = f"run_output/{args.run}/{model_}/{args.dataset}_{args.subsec}/{args.retriever_name}"
    args.generation_trees_results_dir = f'{args.output_dir}/generation_trees'
    args.discrimination_results_file = f"{args.output_dir}/discrimination_results_{args.discriminator_method}.jsonl"
    os.makedirs(args.generation_trees_results_dir, exist_ok=True)
    
    # === Prompt files =============
    args.query_decomposition_prompt_file = "run_mcts/prompts/query_decomposition_prompt_template.txt"
    args.semantic_equivalence_prompt_file = "run_mcts/prompts/semantic_equivalence_prompt_template.txt"
    
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
    # mcts_generation(args)
    subsample_generation(args)
    
    
    # python run_mcts/mcts_generator.py
    # accelerate launch --multi_gpu  run_mcts/mcts_generator.py
    
























    # dataset_ = BaseDataset(args.dataset, args.subsec, args.fraction_of_data_to_use)
    # dataset = dataset_.dataset
    # sample_index = 0
    # print(f"Dataset example {sample_index}:")
    # print(f"Id:             {dataset[sample_index]['qid']}")
    # print(f"Question:       {dataset[sample_index]['question']}")
    # print(f"Answers:        {dataset[sample_index]['ground_truths']}")
    # print(f"Reasoning Steps:{dataset[sample_index]['reasoning_steps']}")
    # print(f"Gold Context: \n{dataset[sample_index]['positive_ctxs'][0]}\n\n")