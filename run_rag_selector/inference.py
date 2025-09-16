#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import ast
import json
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
        per_device_eval_batch_size=4,
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
    
    # metrics = trainer.evaluate()
    # print("=== Evaluation Metrics ===")
    # print(json.dumps(metrics, indent=2))
    
    # - 
    predictions = trainer.predict(tokenized_dataset["test"])
    preds = predictions.predictions
    labels = predictions.label_ids
    metrics = predictions.metrics
    print(json.dumps(metrics, indent=2))
    with open(args.save_results_path, "w") as f:
        for i in range(len(preds)):
            record = {
                "qid": tokenized_dataset["test"][i]["qid"],
                "query": tokenized_dataset["test"][i]["query"],
                "prediction": preds[i].tolist() if hasattr(preds[i], "tolist") else preds[i],
                "label": labels[i].tolist() if hasattr(labels[i], "tolist") else labels[i],
            }
            f.write(json.dumps(record) + "\n")

    