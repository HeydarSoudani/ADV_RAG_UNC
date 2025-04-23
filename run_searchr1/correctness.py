import re
import string
from collections import Counter


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_score(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def f1_score(prediction, ground_truth):
    ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        
    final_metric = {'f1': 0, 'precision': 0, 'recall': 0}
    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
            continue
        if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
            continue
        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        for k in ['f1', 'precision', 'recall']:
            final_metric[k] = max(eval(k), final_metric[k])
    return final_metric