import re
import string
import numpy as np
from collections import Counter
from typing import Dict, List, Callable, Tuple, Union, Callable


class CorrectnessEval:
    def __init__(self):
        pass
    
    def normalize_answer(self, s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    def exact_match_score(
        self,
        prediction: str,
        ground_truth: Union[str, List[str]]
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        # correct = np.max([int(self.normalize_answer(prediction) == self.normalize_answer(gt)) for gt in ground_truths])
        correct = int(any(self.normalize_answer(prediction) in self.normalize_answer(gt) for gt in ground_truths))
        
        return {'correct': correct, 'incorrect': 1 - correct}
    
    
    def f1_score(
        self,
        prediction: str,
        ground_truth: Union[str, List[str]],
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
            
        final_metric = {'f1': 0, 'precision': 0, 'recall': 0}
        for ground_truth in ground_truths:
            normalized_prediction = self.normalize_answer(prediction)
            normalized_ground_truth = self.normalize_answer(ground_truth)
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