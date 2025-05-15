import json
import torch
import random, os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from run_mcts.src.templates import ENTAILMENT_PROMPT, DEFAULT_SYSTEM_PROMPT

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def read_txt(file_path):
    assert str(file_path).endswith(".txt")
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    return data

def read_json(file_path):
    assert str(file_path).endswith(".json")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # skip empty lines
                data.append(json.loads(line))
    return data


def check_entailment(
    model_for_entailment: PreTrainedModel,
    tokenizer_for_entailment: PreTrainedTokenizer,
    context: str,
    seq1: str,
    seq2: str,
):
    with torch.no_grad():
        inputs = tokenizer_for_entailment.encode_plus(
            text=context + " " + seq1,
            text_pair=context + " " + seq2,
            return_tensors="pt",
            truncation=True,
            max_length=model_for_entailment.config.max_position_embeddings,
        ).to(model_for_entailment.device)
        outputs = model_for_entailment(**inputs)
        logits = outputs.logits.cpu()
        del inputs, outputs
        probs = torch.softmax(logits, dim=-1)
        out_class = torch.argmax(probs[0], dim=-1).item()

        return out_class

def check_entailment_with_generation(
    model,
    seq1: str = None,
    seq2: str = None,
    context: str = None,
    entailment_prompt: str = ENTAILMENT_PROMPT,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    tokenizer=None,
    max_new_tokens: int = 32,
):
    if (
        system_prompt is None
    ):  # for some models there is no system prompt in their chat template such as gemma
        chat = [
            {"role": "user", "content": entailment_prompt.format(
                question=input_text)}
        ]
    else:
        chat = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": entailment_prompt.format(
                    seq1=seq1, seq2=seq2, context=context
                ),
            },
        ]
    if type(model) != str:
        input_text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        generated_text = generate(
            input_text,
            model,
            tokenizer=tokenizer,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )["generated_text_skip_specials"]
    else:
        response = completion(model=model, messages=chat)
        generated_text = response.choices[0].message["content"]
    if "same" in generated_text.lower():
        return 2
    elif "contradicted" in generated_text.lower():
        return 0
    else:
        return 1

def calculate_jaccard_similarity(text1: str, text2: str) -> float:
    vectorizer = CountVectorizer().fit([text1, text2])
    vectors = vectorizer.transform([text1, text2]).toarray()

    intersection = (vectors[0] & vectors[1]).sum()
    union = (vectors[0] | vectors[1]).sum()
    return intersection / union if union != 0 else 0


def bidirectional_entailment_clustering(
    model_for_entailment: PreTrainedModel,
    tokenizer_for_entailment: PreTrainedTokenizer,
    context: str,
    sequences: list[str],
    method: str = "semantic",
    entailment_prompt: str = ENTAILMENT_PROMPT,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
):
    clusters = [[sequences[0]]]
    for s_m in sequences[1:]:
        added_to_class = False
        for c in clusters:
            s_c = c[0]  # Use the first sequence in the class for comparison
            if method == "semantic":
                left = check_entailment(
                    model_for_entailment, tokenizer_for_entailment, context, s_c, s_m
                )
                right = check_entailment(
                    model_for_entailment, tokenizer_for_entailment, context, s_m, s_c
                )

                if left != 0 and right != 0:  # it shows there is no contradiction
                    c.append(s_m)
                    added_to_class = True
                    break
            elif method == "jaccard":
                similarity = calculate_jaccard_similarity(s_c, s_m)
                if similarity >= 0.7:
                    c.append(s_m)
                    added_to_class = True
                    break
            elif method == "generation":
                check = check_entailment_with_generation(
                    model_for_entailment,
                    seq1=s_c,
                    seq2=s_m,
                    context=context,
                    entailment_prompt=entailment_prompt,
                    system_prompt=system_prompt,
                    tokenizer=tokenizer_for_entailment,
                )
                if check != 0:
                    c.append(s_m)
                    added_to_class = True
                    break
        if not added_to_class:
            clusters.append([s_m])

    return clusters