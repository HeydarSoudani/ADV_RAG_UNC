#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import ast
import json
import numpy as np
from functools import partial
from safetensors.torch import load_file
from transformers import AutoTokenizer, TrainingArguments

from run_rag_selector.data_preparation import get_prompt_template
import run_rag_selector.src.training_models.pairwise_confidence_in_input as PairwiseConfidenceInInput
import run_rag_selector.src.training_models.pairwise_confidence_in_representation as PairwiseConfidenceInRepresentation

def get_last_checkpoint(checkpoints_dir: str) -> str:
    entries = os.listdir(checkpoints_dir)
    checkpoint_dirs = []
    for e in entries:
        m = re.match(r"checkpoint-(\d+)", e)
        if m:
            checkpoint_dirs.append((int(m.group(1)), e))

    if not checkpoint_dirs:
        raise ValueError(f"No checkpoint found in {checkpoints_dir}")

    _, last_dir = max(checkpoint_dirs, key=lambda x: x[0])
    return os.path.join(checkpoints_dir, last_dir)

def write_results_in_file(pred_out, test_ds, args):
    preds = np.asarray(pred_out.predictions)
    labels = np.asarray(pred_out.label_ids)

    # Squeeze logits like (N,1)->(N,)
    if preds.ndim > 1 and preds.shape[-1] == 1:
        preds = preds.squeeze(-1)
    elif preds.ndim == 0:
        preds = np.array([preds], dtype=float)

    # Expect labels as (N,2): [group_id, is_correct]
    if labels.ndim == 1:
        labels = labels.reshape(-1, 2)
    if labels.ndim != 2 or labels.shape[1] != 2:
        raise ValueError(f"Expected labels shape (N,2) [group_id, is_correct], got {labels.shape}")
    if preds.shape[0] != labels.shape[0]:
        raise ValueError(f"Pred/label length mismatch: preds={preds.shape[0]} labels={labels.shape[0]}")

    # Keep only finite predictions (track original indices)
    N = len(preds)
    idx_all = np.arange(N)
    valid = np.isfinite(preds)
    preds = preds[valid]
    labels = labels[valid]
    idx_all = idx_all[valid]

    group_ids  = labels[:, 0].astype(np.int64, copy=False)
    is_correct = labels[:, 1].astype(np.int64, copy=False)

    # ---- select top-1 per group (same as compute_metrics) ----
    # Sort by group, then by pred (so last in each group has max pred)
    order = np.lexsort((preds, group_ids))
    g = group_ids[order]
    p = preds[order]
    y = is_correct[order]
    orig_idx = idx_all[order]

    # find group boundaries
    starts = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1]
    ends   = np.r_[starts[1:] - 1, len(g) - 1]   # last index of each group = top pred due to sort

    with open(args.save_results_path, "w") as f:
        for end_i in ends:
            i = int(orig_idx[end_i])      # original dataset index of the chosen sample
            record = {
                "qid":   test_ds[int(g[end_i])]["qid"],
                "query": test_ds[int(g[end_i])]["query"],
                "group_id": int(g[end_i]),
                "prediction": float(p[end_i]),
                "label": int(y[end_i]),            # 1 if the chosen sample is correct, else 0
                "correct": bool(y[end_i] == 1),    # convenience field
                "row_index": int(i),               # original row position (useful for debugging)
            }
            f.write(json.dumps(record) + "\n")

def inference(args, dataset):
    prompt_template = get_prompt_template(args.prompt_format)
    tokenizer = AutoTokenizer.from_pretrained(args.selector_model_name_or_path, cache_dir=args.cache_dir)
    
    if args.training_method == "pairwise" and args.confidence_score_injection == "in_input":
        trainer_model = PairwiseConfidenceInInput
    elif args.training_method == "pairwise" and args.confidence_score_injection == "in_representation":
        trainer_model = PairwiseConfidenceInRepresentation
    
    # === Printing Samples ======
    print('\n---\n')
    selected_test_sample = dataset['test'][0]
    selected_test_sample_tuple = ast.literal_eval(selected_test_sample['candidates'])[0]
    selected_test_sample_tuple_str = prompt_template.format(
            sep_token=tokenizer.sep_token,
            query=selected_test_sample['query'],
            answer=selected_test_sample_tuple[0],
            conf_score=selected_test_sample_tuple[1],
            search_queries=' '.join(str(g) for g in selected_test_sample_tuple[3] if g),
            thinks=' '.join(str(g) for g in selected_test_sample_tuple[4] if g),
            docs=' '.join(str(g) for g in selected_test_sample_tuple[5][:args.n_docs_prompt] if g),
            generations=' '.join(str(g) for g in selected_test_sample_tuple[6] if g),
        )
    print(f'Test Prompt:\n{selected_test_sample_tuple_str}')
    print('------\n')
    
    # === Inference ... ==========
    preproc_fn = partial(
        trainer_model.preprocess_function,
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        args=args,
    )
    tokenized_dataset = dataset.map(preproc_fn, with_indices=True)
    data_collator = trainer_model.DataCollator(tokenizer)
    
    model = trainer_model.RewardRanker(args.selector_model_name_or_path)
    weights_path = os.path.join(get_last_checkpoint(args.saved_model_name_or_path), "model.safetensors")
    state = load_file(weights_path)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:    print("[load] missing keys:", len(missing))
    if unexpected: print("[load] unexpected keys:", len(unexpected))
    model.eval()
    
    eval_args = TrainingArguments(
        output_dir='./',
        do_train=False,
        do_eval=True,
        per_device_eval_batch_size=5,
        remove_unused_columns=False,      # keep all features used by your custom collator/model
        report_to=[],                     # disable W&B etc.
        seed=args.seed
    )
    
    trainer = trainer_model.RewardTrainer(
        model=model,
        args=eval_args,
        train_dataset=None,
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=trainer_model.compute_metrics,  # returns acc@1
    )

    # - 
    test_ds = tokenized_dataset["test"]
    pred_out = trainer.predict(test_ds)

    metrics = pred_out.metrics
    print(json.dumps(metrics, indent=2))
    write_results_in_file(pred_out, test_ds, args)
