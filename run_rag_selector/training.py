#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ast
from functools import partial
from transformers import AutoTokenizer, TrainingArguments

from run_rag_selector.data_preparation import get_prompt_template
import run_rag_selector.src.training_models.pairwise_confidence_in_input as PairwiseConfidenceInInput
import run_rag_selector.src.training_models.pairwise_confidence_in_representation as PairwiseConfidenceInRepresentation


def training(args, dataset):
    prompt_template = get_prompt_template(args.prompt_format)
    tokenizer = AutoTokenizer.from_pretrained(args.selector_model_name_or_path, cache_dir=args.cache_dir)
    
    if args.training_method == "pairwise" and args.confidence_score_injection == "in_input":
        trainer_model = PairwiseConfidenceInInput
    elif args.training_method == "pairwise" and args.confidence_score_injection == "in_representation":
        trainer_model = PairwiseConfidenceInRepresentation
    
    # === Printing Samples ======
    print('------')
    selected_train_sample = dataset['train'][0]
    selected_train_sample_pos_tuple = ast.literal_eval(selected_train_sample['positive_sample'])
    selected_train_sample_pos_str = prompt_template.format(
            sep_token=tokenizer.sep_token,
            query=selected_train_sample_pos_tuple[0],
            answer=selected_train_sample_pos_tuple[1],
            conf_score=selected_train_sample_pos_tuple[2],
            search_queries=' '.join(str(g) for g in selected_train_sample_pos_tuple[4] if g),
            thinks=' '.join(str(g) for g in selected_train_sample_pos_tuple[5] if g),
            docs=' '.join(str(g) for g in selected_train_sample_pos_tuple[6][:args.n_docs_prompt] if g),
            generations=' '.join(str(g) for g in selected_train_sample_pos_tuple[7] if g),
        )
    print(f'Train Prompt:\n{selected_train_sample_pos_str}')
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

    # === Training ... ==========
    preproc_fn = partial(
        trainer_model.preprocess_function,
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        args=args,
    )
    tokenized_dataset = dataset.map(preproc_fn, with_indices=True)
    data_collator = trainer_model.DataCollator(tokenizer)
    model = trainer_model.RewardRanker(args.selector_model_name_or_path)
    
    training_args = TrainingArguments(
        output_dir=args.saved_model_name_or_path,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.0,
        num_train_epochs=5,
        greater_is_better=True,
        load_best_model_at_end=True,
        lr_scheduler_type="linear",
        warmup_ratio=0.05,
        eval_strategy="epoch",
        save_strategy="epoch",
        eval_steps=2,
        logging_steps=100,
        remove_unused_columns=False,
        save_total_limit=2, 
        metric_for_best_model="acc@1",
        report_to=[],
        seed=args.seed
    )
    
    trainer = trainer_model.RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=trainer_model.compute_metrics,
    )
    trainer.train()
