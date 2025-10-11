#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import glob
import json
import torch
import random
import argparse
import transformers
from tqdm import tqdm
from math import sqrt, erf
from accelerate import Accelerator
from sklearn.metrics import roc_auc_score
from MLstatkit import Delong_test

from utils.general_utils import set_seed, passages2string
from run_rag_methods.src.rag_methods import *
from run_uncertainty_estimation.consistency_methods import *
from run_rag_methods.src.correctness import em_score, subem_score, f1_score
from run_uncertainty_estimation.src.uncertainty_estimator import UncertaintyEstimator
from run_mcts.run_mcts_two_actions.src.models.semantic_equivalence import SemanticEquivalenceGenerator
from run_uncertainty_estimation.ue_methods import MajorityVoting
from run_rag_methods.src.retrievers_local import load_docs


def ue_generation(args):
    are_traces_generated = False # when yo generated the paths and want to add UE results
    # === MultiGPU setup =========================
    accelerator = Accelerator()
    device = accelerator.device
    if accelerator.is_main_process:
        print("\n== UE Generation ...")
        print(f"""
            Model name:    {args.model_name_or_path}
            Secondary M.:  {args.secondary_model_name_or_path}
            Dataset:       {args.dataset} / {args.subsec} ({args.fraction_of_data_to_use})
            Retriever:     {args.retriever_name} / ({args.retrieval_model_path})
            RAG Method:    {args.rag_method}
            Con. Method:   {args.consistency_method}
            Action Set:    {args.action_set}
            Run:           {args.run}
            Seed:          {args.seed}
        """.replace('            ', ''))
        
        # --- Define CUDA device
        if torch.cuda.is_available():
            print(f"Number of available GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA is not available. No GPUs detected.")
        print('\n')

    # === Read input data ========================
    query_ids, rag_generations = [], {}
    if os.path.exists(args.inference_results_file):
        with open(args.inference_results_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'qid' in data:
                    query_ids.append(data['qid'])
                    rag_generations[data['qid']] = data
    sorted_query_ids = sorted(query_ids, key=lambda x: int(x.split('_')[1]))
    
    # === Read existing (generated) samples ======
    generated_qids = []
    if not are_traces_generated:
        if os.path.exists(args.consistency_results_file):
            with open(args.consistency_results_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    if 'qid' in data:
                        generated_qids.append(data['qid'])
        generated_qids = set(generated_qids)
    filtered_sorted_query_ids = [id_ for id_ in sorted_query_ids if id_ not in generated_qids]
    
    # === Read traces (generated) samples =======
    if are_traces_generated:
        generated_traces_obj = {}
        if os.path.exists(args.masked_traces_results_file):
            with open(args.masked_traces_results_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    if 'qid' in data:
                        generated_traces_obj[data['qid']] = data['masked_traces']
                    
    # === Read Models ============================
    generation_model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path, dtype=torch.bfloat16).to(device) # attn_implementation="eager" # Disable this for searchR1
    generation_tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    secondary_model = transformers.AutoModelForCausalLM.from_pretrained(args.secondary_model_name_or_path, dtype=torch.bfloat16).to(device)
    secondary_tokenizer = transformers.AutoTokenizer.from_pretrained(args.secondary_model_name_or_path)
    
    # -
    if args.rag_method == 'direct_inference':
        rag_model = DirectInference(generation_model, generation_tokenizer, device, args)
    elif args.rag_method == 'cot_inference':
        rag_model = CoTInference(generation_model, generation_tokenizer, device, args)
    elif args.rag_method == "cot_single_retrieval":
        rag_model = CoTSingleRAG(generation_model, generation_tokenizer, device, args)
    elif args.rag_method == "fix_sentence_retrieval":
        rag_model = FixSentenceRAG(args, device)
    elif args.rag_method == "fix_length_retrieval":
        rag_model = FixLengthRAG(args, device)
    elif args.rag_method == 'ircot':
        rag_model = IRCOT_RAG(generation_model, generation_tokenizer, device, args)
    elif args.rag_method == 'flare':
        rag_model = FLARE_RAG_V1(generation_model, generation_tokenizer, device, args)
    elif args.rag_method == 'dragin':
        rag_model = DRAGIN_RAG(generation_model, generation_tokenizer, device, args)
    elif args.rag_method == 'self_ask':
        rag_model = SelfAsk_RAG(generation_model, generation_tokenizer, device, args)
    elif args.rag_method == 'react':
        rag_model = ReAct_RAG(generation_model, generation_tokenizer, device, args)
    elif args.rag_method == 'search_o1':
        rag_model = SearchO1_RAG(generation_model, generation_tokenizer, device, args)
    elif args.rag_method == 'research':
        rag_model = ReSearch_RAG(generation_model, generation_tokenizer, device, args)
    elif args.rag_method == 'search_r1':
        rag_model = SearchR1_RAG(generation_model, generation_tokenizer, device, args)
    else:
        raise NotImplementedError
    
    # -
    if args.consistency_method == 'fa_consistency':
        consistency_model = FAConsistency(rag_model, args)
    elif args.consistency_method == 'rrr_consistency':       # retrieval-retained reasoning consistency
        consistency_model = RRRConsistency(rag_model, args)
    elif args.consistency_method == 'reasoning_consistency': # reasoning consistency
        consistency_model = ReasoningConsistency(rag_model, args)
    elif args.consistency_method == 'self_consistency':
        consistency_model = SelfConsistency(rag_model, args)
    elif args.consistency_method == 'rag_consistency':
        consistency_model = RagConsistency(rag_model, secondary_model, secondary_tokenizer, device, args)
    else:
        raise NotImplementedError
    
    # -
    uncertainty_estimator_model = UncertaintyEstimator(
        model=secondary_model,
        tokenizer=secondary_tokenizer,
        device=device,
        args=args,
        generated_output_template=rag_model.answer_template
    )
    
    # === Functions ==============================
    def get_unique_docs(traces):
        docs_lst = [doc for trace in traces for step in trace for doc in step.get('docs', [])]        
        return list({doc['id']: doc for doc in docs_lst}.values()) 
    
    # === Main Loop ==============================
    accelerator.wait_for_everyone()
    with accelerator.split_between_processes(filtered_sorted_query_ids) as sorted_query_ids_shard:
        
        # --- Consistency file -----------
        if args.rag_method in ['flare', 'dragin']:
            if args.consistency_method == "rag_consistency":
                consistency_results_file_ranked = f"{args.output_dir}/{args.consistency_method}_{args.action_set}_results_th{args.hallucination_threshold}_rank{accelerator.process_index}.jsonl"
            else:
                consistency_results_file_ranked = f"{args.output_dir}/{args.consistency_method}_results_th{args.hallucination_threshold}_rank{accelerator.process_index}.jsonl"
        else:
            if args.consistency_method == "rag_consistency":
                consistency_results_file_ranked = f"{args.output_dir}/{args.consistency_method}_{args.action_set}_results_rank{accelerator.process_index}.jsonl"
            else:
                consistency_results_file_ranked = f"{args.output_dir}/{args.consistency_method}_results_rank{accelerator.process_index}.jsonl"
        cons_f = open(consistency_results_file_ranked, 'w', encoding='utf-8')

        trace_f = None
        write_traces = args.consistency_method != 'fa_consistency'
        if write_traces and not are_traces_generated:
            if args.rag_method in ['flare', 'dragin']:
                if args.consistency_method == "rag_consistency":
                    masked_traces_results_file_ranked = f"{args.output_dir}/{args.consistency_method}_{args.action_set}_masked_traces_results_th{args.hallucination_threshold}_rank{accelerator.process_index}.jsonl"
                else:
                    masked_traces_results_file_ranked = f"{args.output_dir}/{args.consistency_method}_masked_traces_results_th{args.hallucination_threshold}_rank{accelerator.process_index}.jsonl"
            else:
                if args.consistency_method == "rag_consistency":
                    masked_traces_results_file_ranked = f"{args.output_dir}/{args.consistency_method}_{args.action_set}_masked_traces_results_rank{accelerator.process_index}.jsonl"
                else:
                    masked_traces_results_file_ranked = f"{args.output_dir}/{args.consistency_method}_masked_traces_results_rank{accelerator.process_index}.jsonl"
                
            trace_f = open(masked_traces_results_file_ranked, 'w', encoding='utf-8')

        try:
            for i, qid in enumerate(tqdm(sorted_query_ids_shard, desc=f"[Rank {accelerator.process_index}]")):
                # if i == 3:
                #     break
                sample = rag_generations[qid]
                user_query, prediction, trace = sample['query'], sample['pred_answer'], sample['path']
                prediction = prediction.strip() if prediction else prediction 
                
                ### --- 1.1) Generate traces list
                if not are_traces_generated:
                    masked_traces, masked_traces_text, final_answer_list = consistency_model.get_masked_traces(qid, user_query, prediction, trace)
                    context = passages2string(get_unique_docs(masked_traces))
                
                ### --- 1.2) Read traces list
                elif are_traces_generated:
                    ## -- Read docs
                    generated_masked_traces_wo_docs = generated_traces_obj[qid]
                    generated_masked_traces_with_docs = []
                    for generated_masked_trace_wo_docs in generated_masked_traces_wo_docs:
                        generated_masked_trace_with_docs = []
                        for step in generated_masked_trace_wo_docs:
                            if 'docs' in step:
                                docs_with_context = []
                                for doc_item in step["docs"]:
                                    if doc_item['id'] == '-1':
                                        docs_with_context.append(doc_item)
                                    else:
                                        docs_with_context.append({
                                            'id': doc_item['id'],
                                            'contents': load_docs(rag_model.retriever.corpus, [doc_item['id']])[0]['contents']
                                        })
                                step_with_docs = {
                                    **step,
                                    "docs": docs_with_context
                                }
                                generated_masked_trace_with_docs.append(step_with_docs)
                            else:
                                generated_masked_trace_with_docs.append(step)
                        generated_masked_traces_with_docs.append(generated_masked_trace_with_docs)
                        
                    ## -- Convert masked trace to text
                    masked_traces_text = [
                        rag_model.get_input_prompt_without_final_answer(user_query, masked_trace)
                        for masked_trace in generated_masked_traces_with_docs
                    ]
                    final_answer_list = [
                        masked_trace[-1]['answer'].strip() if masked_trace[-1]['answer'] else None
                        for masked_trace in generated_masked_traces_with_docs
                    ]
                    context = passages2string(get_unique_docs(generated_masked_traces_with_docs))

                ### --- 2) Calculate UE scores
                ue_scores = uncertainty_estimator_model.estimate(
                    qid, user_query,
                    prediction,
                    context=context,
                    input_prompt_texts = masked_traces_text,
                    generated_output_texts = final_answer_list
                )
                
                ### --- 3) Print in output files 
                cons_item = {
                    "qid": qid,
                    "query": user_query,
                    "gt_answers": sample['gt_answers'],
                    "pred_answer": prediction,
                    "em": sample['em'],
                    "final_answer_list": final_answer_list,
                    "ue_scores": ue_scores
                }
                cons_f.write(json.dumps(cons_item) + "\n")
                
                if trace_f and not are_traces_generated:
                    new_masked_traces = [
                        [
                            {
                                **step,
                                "docs": [{"id": doc["id"]} if doc["id"] != "-1" else doc for doc in step["docs"]]
                            } if "docs" in step else step
                            for step in masked_trace
                        ]
                        for masked_trace in masked_traces
                    ]
                    trace_item = {
                        "qid": qid,
                        "query": user_query,
                        "masked_traces": new_masked_traces
                    }
                    trace_f.write(json.dumps(trace_item) + '\n')

        finally:
            cons_f.close()
            if trace_f and not are_traces_generated:
                trace_f.close()

def merge_result_files(args):
    
    if args.rag_method in ['flare', 'dragin']:
        if args.consistency_method == "rag_consistency":
            consistency_results_file_ranked = f"{args.output_dir}/{args.consistency_method}_{args.action_set}_results_th{args.hallucination_threshold}_rank*.jsonl"
        else:
            consistency_results_file_ranked = f"{args.output_dir}/{args.consistency_method}_results_th{args.hallucination_threshold}_rank*.jsonl"
    else:
        if args.consistency_method == "rag_consistency":
            consistency_results_file_ranked = f"{args.output_dir}/{args.consistency_method}_{args.action_set}_results_rank*.jsonl"
        else:
            consistency_results_file_ranked = f"{args.output_dir}/{args.consistency_method}_results_rank*.jsonl"
    
    # consistency_results_file_ranked = (
    #     f"{args.output_dir}/{args.consistency_method}_results_th{args.hallucination_threshold}_rank*.jsonl"
    #     if args.rag_method in ['flare', 'dragin'] 
    #     else f"{args.output_dir}/{args.consistency_method}_results_rank*.jsonl"
    # )
    consistency_results_shard_files = sorted(glob.glob(consistency_results_file_ranked))
    with open(args.consistency_results_file, "a") as fout:
        for shard_file in consistency_results_shard_files:
            if shard_file == args.consistency_results_file:
                continue
            with open(shard_file, "r") as fin:
                for line in fin:
                    fout.write(line)
            os.remove(shard_file)
            print(f"Deleted shard file: {shard_file}")
         
    write_traces = args.consistency_method != 'fa_consistency'   
    if write_traces:
        # masked_traces_results_file_ranked = (
        #     f"{args.output_dir}/{args.consistency_method}_masked_traces_results_th{args.hallucination_threshold}_rank*.jsonl"
        #     if args.rag_method in ['flare', 'dragin'] 
        #     else f"{args.output_dir}/{args.consistency_method}_masked_traces_results_rank*.jsonl"
        # )
        if args.rag_method in ['flare', 'dragin']:
            if args.consistency_method == "rag_consistency":
                masked_traces_results_file_ranked = f"{args.output_dir}/{args.consistency_method}_{args.action_set}_masked_traces_results_th{args.hallucination_threshold}_rank*.jsonl"
            else:
                masked_traces_results_file_ranked = f"{args.output_dir}/{args.consistency_method}_masked_traces_results_th{args.hallucination_threshold}_rank*.jsonl"
        else:
            if args.consistency_method == "rag_consistency":
                masked_traces_results_file_ranked = f"{args.output_dir}/{args.consistency_method}_{args.action_set}_masked_traces_results_rank*.jsonl"
            else:
                masked_traces_results_file_ranked = f"{args.output_dir}/{args.consistency_method}_masked_traces_results_rank*.jsonl"
        
        masked_traces_results_shard_files = sorted(glob.glob(masked_traces_results_file_ranked))
        with open(args.masked_traces_results_file, "a") as fout:
            for shard_file in masked_traces_results_shard_files:
                if shard_file == args.masked_traces_results_file:
                    continue
                with open(shard_file, "r") as fin:
                    for line in fin:
                        fout.write(line)
                os.remove(shard_file)
                print(f"Deleted shard file: {shard_file}")

def get_auroc(correctness, confidence):
    try:
        auroc = roc_auc_score(correctness, confidence)
    except:
        print("Auroc couldn't be calculated because there is only one class. Returning 0.5 as auroc.")
        auroc = 0.5
    return auroc

def evaluation_correlation(args):
    print("\n== Correlation Evaluation ...")
    print(f"""
        Model name:    {args.model_name_or_path}
        Secondary M.:  {args.secondary_model_name_or_path}
        Dataset:       {args.dataset} / {args.subsec} ({args.fraction_of_data_to_use})
        Retriever:     {args.retriever_name} / ({args.retrieval_model_path})
        RAG Method:    {args.rag_method}
        Con. Method:   {args.consistency_method}
        Run:           {args.run}
        Seed:          {args.seed}
    """.replace('            ', ''))
    
    correctness_list, uncertainty_obj = [], {}
    with open(args.consistency_results_file, 'r') as infile:
        for line in infile:
            data = json.loads(line)
            correctness = data['em']
            ue_scores = data['ue_scores']
            correctness_list.append(correctness)
            
            for ue_metric, ue_value in ue_scores.items():
                if args.consistency_method == 'rag_consistency':
                    conf_score = ue_value['confidence']
                else:
                    conf_score = ue_value['most_confident_answer'][1] if ue_metric == "majority_voting" else ue_value['confidence']
                
                if ue_metric in uncertainty_obj.keys():
                    uncertainty_obj[ue_metric].append(conf_score)
                else:
                    uncertainty_obj[ue_metric] = [conf_score]

        for ue_metric, conf_list in uncertainty_obj.items():
            print(f"{ue_metric}: {get_auroc(correctness_list, conf_list)}")

def evaluation_correlation_num_generations(args):
    print("\n== Correlation Evaluation ...")
    print(f"""
        Model name:    {args.model_name_or_path}
        Secondary M.:  {args.secondary_model_name_or_path}
        Dataset:       {args.dataset} / {args.subsec} ({args.fraction_of_data_to_use})
        Retriever:     {args.retriever_name} / ({args.retrieval_model_path})
        RAG Method:    {args.rag_method}
        Con. Method:   {args.consistency_method}
        Run:           {args.run}
        Seed:          {args.seed}
    """.replace('            ', ''))
    accelerator = Accelerator()
    device = accelerator.device
    secondary_model = transformers.AutoModelForCausalLM.from_pretrained(args.secondary_model_name_or_path, dtype=torch.bfloat16).to(device)
    secondary_tokenizer = transformers.AutoTokenizer.from_pretrained(args.secondary_model_name_or_path)
    se_model = SemanticEquivalenceGenerator(args, device, secondary_model, secondary_tokenizer)
    mv_model =  MajorityVoting(se_model)
    
    num_generations = 2
    correctness_list, confidece_list = [], []
    with open(args.consistency_results_file, 'r') as infile:
        for idx, line in enumerate(tqdm(infile)):
            # if idx == 5:
            #     break
            data = json.loads(line)
            question, correctness = data['query'], data['em']
            prediction = data['pred_answer'].strip() if data['pred_answer'] else ''
            generated_texts = random.sample(data['final_answer_list'], min(len(data['final_answer_list']), num_generations))
            sampled_gen_dict = {'question': question, 'generated_texts': generated_texts}
            
            ue_value = mv_model(sampled_gen_dict, prediction, '')
            if args.consistency_method == 'rag_consistency':
                conf_score = ue_value['confidence']
            else:
                conf_score = ue_value['most_confident_answer'][1]
            
            correctness_list.append(correctness)
            confidece_list.append(conf_score)
            
    print(f"With n={num_generations}: {get_auroc(correctness_list, confidece_list)}")

def get_stat_testing_delong(args):
    print("\n=== De long ===============")
    file1 = f"{args.output_dir}/rag_consistency_{args.action_set}_results.jsonl"
    file2 = args.consistency_results_file = f"{args.output_dir}/{args.consistency_method}_results.jsonl"
        
    # === Functions 
    def rag_extractor(obj):
        qid = obj["qid"]
        score = obj["ue_scores"]["majority_voting"]["confidence"]
        label = obj["em"]
        return qid, score, label
    
    def others_extractor(obj):
        qid = obj["qid"]
        score = obj["ue_scores"]["majority_voting"]['most_confident_answer'][1]
        # score = obj["ue_scores"]["p_true"]["confidence"]
        label = obj["em"]
        return qid, score, label
    
    def load_scores_labels_with_qid(path, extract_fn):
        """
        Load scores, labels, and qids from a JSONL file.
        extract_fn must return (qid, score, label).
        """
        data = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                qid, score, label = extract_fn(obj)
                data[qid] = (float(score), int(label))
        return data

    def align_by_qid(file1, file2, extract_fn1, extract_fn2):
        """
        Load two JSONL files and align them by qid.
        Returns: scores1, scores2, labels (np.arrays)
        """
        d1 = load_scores_labels_with_qid(file1, extract_fn1)
        d2 = load_scores_labels_with_qid(file2, extract_fn2)

        # intersection of qids
        common_qids = sorted(set(d1.keys()) & set(d2.keys()))
        if not common_qids:
            raise ValueError("No overlapping qids between the two files.")

        scores1, scores2, labels = [], [], []
        for qid in common_qids:
            s1, y1 = d1[qid]
            s2, y2 = d2[qid]
            if y1 != y2:
                raise ValueError(f"Label mismatch for qid={qid}: {y1} vs {y2}")
            scores1.append(s1)
            scores2.append(s2)
            labels.append(y1)

        return np.array(scores1), np.array(scores2), np.array(labels)

    def _compute_midrank(x):
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=float)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5*(i + j - 1) + 1  # +1 for 1-based ranks
            i = j
        T2 = np.empty(N, dtype=float)
        T2[J] = T
        return T2

    def _fast_delong(predictions, labels):
        """
        predictions: shape (K, N) where K models, N examples (same labels for all)
        labels:     shape (N,) with 0/1
        Returns:
        aucs: shape (K,)
        cov:  shape (K, K)
        """
        labels = np.asarray(labels, dtype=int)
        assert set(np.unique(labels)) <= {0, 1}, "labels must be 0/1"
        order = np.argsort(-predictions[0, :])  # not essential which model, but stable order helps
        predictions = predictions[:, order]
        labels = labels[order]

        # Split pos/neg
        pos_idx = labels == 1
        neg_idx = labels == 0
        m = np.sum(pos_idx)
        n = np.sum(neg_idx)
        if m == 0 or n == 0:
            raise ValueError("Need at least one positive and one negative sample.")

        preds_pos = predictions[:, pos_idx]
        preds_neg = predictions[:, neg_idx]

        # Midranks within positives and within negatives
        tx = np.apply_along_axis(_compute_midrank, 1, preds_pos)
        ty = np.apply_along_axis(_compute_midrank, 1, preds_neg)

        # Midranks on pooled (pos+neg) for each model
        preds_all = np.concatenate([preds_pos, preds_neg], axis=1)
        tz = np.apply_along_axis(_compute_midrank, 1, preds_all)

        aucs = (tz[:, :m].sum(axis=1) - m*(m+1)/2) / (m*n)

        v01 = (tz[:, :m] - tx) / n
        v10 = 1.0 - (tz[:, m:] - ty) / m

        S01 = np.cov(v01)
        S10 = np.cov(v10)
        # If single model, np.cov returns scalar; coerce to 2D
        if np.ndim(S01) == 0:
            S01 = S01.reshape(1, 1)
            S10 = S10.reshape(1, 1)

        delong_cov = (S01 / m) + (S10 / n)
        return aucs, delong_cov

    def _normal_cdf(z):
        return 0.5 * (1.0 + erf(z / sqrt(2.0)))

    def delong_2sample_auc_test(scores1, labels1, scores2, labels2):
        """
        DeLong test comparing ROC-AUC of two models.

        If labels1 and labels2 refer to the SAME examples (same length and identical label vector),
        this uses the correlated two-sample DeLong. Otherwise it falls back to the
        independent-sample case by constructing a block-diagonal covariance.

        Returns dict with auc1, auc2, diff, var, z, p_value (two-sided).
        """
        scores1 = np.asarray(scores1, dtype=float)
        labels1 = np.asarray(labels1, dtype=int)
        scores2 = np.asarray(scores2, dtype=float)
        labels2 = np.asarray(labels2, dtype=int)

        same_data = (len(labels1) == len(labels2)) and np.all(labels1 == labels2)

        if same_data:
            preds = np.vstack([scores1, scores2])  # shape (2, N)
            aucs, cov = _fast_delong(preds, labels1)
            auc1, auc2 = aucs[0], aucs[1]
            var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
        else:
            # Independent samples: compute variance of each separately, assume cov=0
            auc1, cov1 = _fast_delong(scores1[None, :], labels1)
            auc2, cov2 = _fast_delong(scores2[None, :], labels2)
            auc1, auc2 = auc1.item(), auc2.item()
            var = cov1.item() + cov2.item()

        diff = auc1 - auc2
        if var <= 0:
            # Numerical safety
            var = max(var, 1e-12)

        z = diff / sqrt(var)
        # two-sided p-value
        p = 2 * (1 - _normal_cdf(abs(z)))

        return {
            "auc1": float(auc1),
            "auc2": float(auc2),
            "diff": float(diff),
            "var": float(var),
            "z": float(z),
            "p_value": float(p),
        }

    # === Main
    scores1, scores2, labels = align_by_qid(file1, file2, rag_extractor, others_extractor)

    # --- V1: handy implementation
    result = delong_2sample_auc_test(scores1, labels, scores2, labels)
    print(f"AUC1: {result['auc1']:.6f}")
    print(f"AUC2: {result['auc2']:.6f}")
    print(f"Diff (AUC1 - AUC2): {result['diff']:.6f}")
    print(f"Var(diff): {result['var']:.6e}")
    print(f"z: {result['z']:.4f}")
    print(f"p-value (two-sided): {result['p_value']:.6g}")

    alpha = 0.05
    if result["p_value"] < alpha:
        print(f"Result: Significant at alpha={alpha}")
    else:
        print(f"Result: Not significant at alpha={alpha}")

    # --- V2: with library
    z, p_value = Delong_test(labels, scores1, scores2, return_ci=False, return_auc=False, verbose=0)
    print(f"z = {z:.6f}, p = {p_value:.3e}")
    if p_value < alpha:
        print(f"Result: Significant at alpha={alpha}")
    else:
        print(f"Result: Not significant at alpha={alpha}")

def get_stat_testing_bootstrapping(args, n_bootstrap=10000):
    print("\n=== bootstrapping ==========")
    file1 = f"{args.output_dir}/rag_consistency_{args.action_set}_results.jsonl"
    file2 = args.consistency_results_file = f"{args.output_dir}/{args.consistency_method}_results.jsonl"
    
    # === Functions ===========
    def rag_extractor(obj):
        qid = obj["qid"]
        score = obj["ue_scores"]["majority_voting"]["confidence"]
        label = obj["em"]
        return qid, score, label
    
    def others_extractor(obj):
        qid = obj["qid"]
        # score = obj["ue_scores"]["majority_voting"]['most_confident_answer'][1]
        score = obj["ue_scores"]["p_true"]["confidence"]
        label = obj["em"]
        return qid, score, label
    
    def load_scores_labels_with_qid(path, extract_fn):
        """
        Load scores, labels, and qids from a JSONL file.
        extract_fn must return (qid, score, label).
        """
        data = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                qid, score, label = extract_fn(obj)
                data[qid] = (float(score), int(label))
        return data

    def align_by_qid(file1, file2, extract_fn1, extract_fn2):
        """
        Load two JSONL files and align them by qid.
        Returns: scores1, scores2, labels (np.arrays)
        """
        d1 = load_scores_labels_with_qid(file1, extract_fn1)
        d2 = load_scores_labels_with_qid(file2, extract_fn2)

        # intersection of qids
        common_qids = sorted(set(d1.keys()) & set(d2.keys()))
        if not common_qids:
            raise ValueError("No overlapping qids between the two files.")

        scores1, scores2, labels = [], [], []
        for qid in common_qids:
            s1, y1 = d1[qid]
            s2, y2 = d2[qid]
            if y1 != y2:
                raise ValueError(f"Label mismatch for qid={qid}: {y1} vs {y2}")
            scores1.append(s1)
            scores2.append(s2)
            labels.append(y1)

        return np.array(scores1), np.array(scores2), np.array(labels)

    # === Main ================
    rng = np.random.default_rng(args.seed)
    scores1, scores2, labels = align_by_qid(file1, file2, rag_extractor, others_extractor)
    
    auroc1 = roc_auc_score(labels, scores1)
    auroc2 = roc_auc_score(labels, scores2)
    obs_diff = auroc1 - auroc2
    
    # bootstrap
    diffs = []
    n = len(labels)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)  # sample with replacement
        if len(np.unique(labels[idx])) < 2:  # skip if resample has only one class
            continue
        boot_auroc1 = roc_auc_score(labels[idx], np.array(scores1)[idx])
        boot_auroc2 = roc_auc_score(labels[idx], np.array(scores2)[idx])
        diffs.append(boot_auroc1 - boot_auroc2)

    diffs = np.array(diffs)
    p_value = np.mean(np.abs(diffs) >= np.abs(obs_diff))
    ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])
    
    alpha = 0.05
    print(f"AUROC (Model 1): {auroc1:.4f}")
    print(f"AUROC (Model 2): {auroc2:.4f}")
    print(f"Observed Difference: {obs_diff:.4f}")
    print(f"p-value (two-sided): {p_value:.4f}")
    print(f"95% CI for difference: ({ci_lower:.4f}, {ci_upper:.4f})")
    if p_value < alpha:
        print(f"Result: Significant at alpha={alpha}")
    else:
        print(f"Result: Not significant at alpha={alpha}")

    return {
        "auroc1": auroc1,
        "auroc2": auroc2,
        "obs_diff": obs_diff,
        "p_value": p_value,
        "ci": (ci_lower, ci_upper)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model_name_or_path', type=str, default='PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo')
    # parser.add_argument('--model_name_or_path', type=str, default="agentrl/ReSearch-Qwen-7B-Instruct")
    # parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--secondary_model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='hotpotqa', choices=[
        'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'
    ])
    parser.add_argument('--subsec', type=str, default='dev', choices=['train', 'dev', 'test', 'validation'])
    parser.add_argument('--fraction_of_data_to_use', type=float, default=2000.0)
    parser.add_argument("--enable_fewshot_examples", action="store_true", help="")
    
    # Retriever
    parser.add_argument('--retriever_name', type=str, default='rerank_l6', choices=[
        'bm25', 'contriever', 'rerank_l6', 'rerank_l12', 'e5', 'bge', 'reasonir'
    ])
    parser.add_argument('--corpus_path', type=str, default='data/search_r1_files/wiki-18.jsonl')
    parser.add_argument('--index_path', type=str, default='data/search_r1_files/bm25', choices=[
        'data/search_r1_files/bm25',                # For BM25 & Rerank
        'data/search_r1_files/e5_Flat.index',       # For E5
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
    
    # RAG methods (input)
    parser.add_argument('--rag_method', type=str, default='search_r1', choices=[
        'direct_inference', 'cot_inference', 'cot_single_retrieval',
        'fix_length_retrieval', 'fix_sentence_retrieval',
        'ircot', 'flare', 'dragin',
        'self_ask', 'react', 'search_o1',
        'research', 'search_r1'
    ])
    parser.add_argument('--generate_fix_length', type=int, default=25)
    parser.add_argument('--modifier_method', type=str, default='token', choices=['token', 'entity'])          # for FLARE
    parser.add_argument('--query_formulation', type=str, default='direct', choices=[                      # for FLARE & DRAGIN
        'direct', 'forward_all',
        'real_words', 'current', 'current_wo_wrong', 'last_sentence', 'last_n_tokens',
    ])
    parser.add_argument('--sentence_solver', type=str, default='avg', choices=['avg', 'max', 'min'])          # for FLARE
    parser.add_argument('--hallucination_threshold', type=float, default=0.08)                                 # for FLARE & DRAGIN
    parser.add_argument('--retrieve_keep_top_k', type=int, default=25)                                        # for DRAGIN
    parser.add_argument('--check_real_words', action='store_false')                                           # for DRAGIN
    parser.add_argument('--max_iter', type=int, default=5)
    
    # Consistency Generation Methods (answer list)
    parser.add_argument('--consistency_method', type=str, default='reasoning_consistency', choices=[
        'fa_consistency', 'rrr_consistency', 'reasoning_consistency', 'self_consistency', 'rag_consistency'
    ])
    parser.add_argument("--action_set", type=str, default='qp_ct_av', choices=[
        'qp', 'ct', 'av', 'qp_ct', 'qp_av', 'ct_av', 'qp_ct_av'
    ])
    parser.add_argument("--n_generations", type=int, default=10)
    parser.add_argument("--mask_left_boundary", type=float, default=0.1)
    parser.add_argument("--mask_right_boundary", type=float, default=0.4)
    parser.add_argument("--consistency_temperature", type=float, default=1.0)
    
    # Others
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--run', type=str, default='run_3 (rag_methods_500)')
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
        
        if args.consistency_method == "rag_consistency":
            args.consistency_results_file = f"{args.output_dir}/{args.consistency_method}_{args.action_set}_results_th{args.hallucination_threshold}.jsonl"
        else:
            args.consistency_results_file = f"{args.output_dir}/{args.consistency_method}_results_th{args.hallucination_threshold}.jsonl"
        
        if args.consistency_method != "fa_consistency":
            if args.consistency_method == "rag_consistency":
                args.masked_traces_results_file = f"{args.output_dir}/{args.consistency_method}_{args.action_set}_masked_traces_th{args.hallucination_threshold}.jsonl"
            else:
                args.masked_traces_results_file = f"{args.output_dir}/{args.consistency_method}_masked_traces_th{args.hallucination_threshold}.jsonl"
    else:
        args.inference_results_file = f"{args.output_dir}/inference_results.jsonl"
        
        if args.consistency_method == "rag_consistency":
            args.consistency_results_file = f"{args.output_dir}/{args.consistency_method}_{args.action_set}_results.jsonl"
        else:
            args.consistency_results_file = f"{args.output_dir}/{args.consistency_method}_results.jsonl"
        
        if args.consistency_method != "fa_consistency":
            if args.consistency_method == "rag_consistency":
                args.masked_traces_results_file = f"{args.output_dir}/{args.consistency_method}_{args.action_set}_masked_traces.jsonl"
            else:
                args.masked_traces_results_file = f"{args.output_dir}/{args.consistency_method}_masked_traces.jsonl"
        
    # === Prompt files =============
    args.query_decomposition_prompt_file = "run_mcts/run_mcts_two_actions/prompts/query_decomposition_prompt_template.txt"
    args.semantic_equivalence_prompt_file = "run_mcts/run_mcts_two_actions/prompts/semantic_equivalence_prompt_template.txt"
    
    # === Run Steps ================
    set_seed(args.seed)
    # ue_generation(args)
    # merge_result_files(args)
    # evaluation_correlation(args)
    # evaluation_correlation_num_generations(args)
    get_stat_testing_delong(args)
    # get_stat_testing_bootstrapping(args)
    
    # python run_uncertainty_estimation/ue_calculation.py
    # accelerate launch --multi_gpu run_uncertainty_estimation/ue_calculation.py


























# eps = 1e-12
# landa_1 = 0.7
# landa_2 = 0.3
# def evaluation_correlation_combined(args):
#     second_consistency = 'fa_consistency'
#     if args.rag_method in ['flare', 'dragin']:
#         second_consistency_results_file = f"{args.output_dir}/{second_consistency}_results_th{args.hallucination_threshold}.jsonl"
#     else:
#         second_consistency_results_file = f"{args.output_dir}/{second_consistency}_results.jsonl"
        
#     second_uncertainty_obj = {}    
#     with open(second_consistency_results_file, 'r') as infile:
#         for line in infile:
#             data = json.loads(line)
#             conf = data['ue_scores']['majority_voting']['confidence']
#             second_uncertainty_obj[data['qid']] = conf
    
#     correctness_list, conf_list = [], []
#     with open(args.consistency_results_file, 'r') as infile:
#         for line in infile:
#             data = json.loads(line)
#             correctness = data['em']
#             correctness_list.append(correctness)
#             conf = data['ue_scores']['majority_voting']['confidence']
#             agg_conf = landa_1 * conf + landa_2 * second_uncertainty_obj[data['qid']]
#             # agg_conf = 1.0 / ((1.0 / (conf+eps)) + (0.0 / (second_uncertainty_obj[data['qid']]+eps)))
#             conf_list.append(agg_conf)
    
#     print(f"AUROC: {get_auroc(correctness_list, conf_list)}")

# def correctness_evaluation_mv(args):
#     em_mv_full_evaluation, em_mv_sub_evaluation = [], []
#     em_org_full_evaluation, em_org_sub_evaluation = [], []
#     conf_list = []
#     with open(args.consistency_results_file, 'r') as infile:
#         for line in infile:
#             data = json.loads(line)
#             gt_answers = data['gt_answers']
            
#             pred_answer_mc = data['ue_scores']['majority_voting']['most_confident_answer'][0]
#             if pred_answer_mc:
#                 correctness_em = em_score(pred_answer_mc, gt_answers)
#                 correctness_em_sub = subem_score(pred_answer_mc, gt_answers)
#             else:
#                 correctness_em, correctness_em_sub = 0, 0
#             em_mv_full_evaluation.append(correctness_em)
#             em_mv_sub_evaluation.append(correctness_em_sub)
            
#             conf_mc = data['ue_scores']['majority_voting']['most_confident_answer'][1]
#             conf_list.append(conf_mc)
            
            
#             pred_answer_org = data['pred_answer']
#             if pred_answer_org:
#                 correctness_em = em_score(pred_answer_org, gt_answers)
#                 correctness_em_sub = subem_score(pred_answer_org, gt_answers)
#             else:
#                 correctness_em, correctness_em_sub = 0, 0
#             em_org_full_evaluation.append(correctness_em)
#             em_org_sub_evaluation.append(correctness_em_sub)
            
#     # === Print results ========================
#     print(f"\nEvaluation Result (MC) {args.consistency_method}:")
#     print(f"EM (full): {np.mean(em_mv_full_evaluation)*100}")
#     print(f"EM (sub): {np.mean(em_mv_sub_evaluation)*100}")
#     print(f"AUROC: {get_auroc(em_mv_full_evaluation, conf_list)}")
    
#     print(f"\nEvaluation Result (Org):")
#     print(f"EM (full): {np.mean(em_org_full_evaluation)*100}")
#     print(f"EM (sub): {np.mean(em_org_sub_evaluation)*100}")
