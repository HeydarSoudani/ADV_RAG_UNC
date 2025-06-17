import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np

from run_uncertainty_estimation.src.scoring_methods import ScoringMethod, LengthNormalizedScoring

class PredictiveEntropy:
    def __init__(self, scoring_function : ScoringMethod = LengthNormalizedScoring()):
        self.scoring_function = scoring_function
    
    def _entropy(self, scores: list[float]):
        entropy = -np.sum(scores) / len(scores)
        return entropy
    
    def __call__(self, sampled_gen_dict, prediction, context):
        scores_ = [self.scoring_function(logp) for logp in sampled_gen_dict['logprobs']]
        score = self._entropy(scores_)
        
        return {
            "confidence": -score,
            "uncertainty": score
        } 
