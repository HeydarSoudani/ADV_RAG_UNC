from ...archive.run_mcts_src_.ue_methods.ue_method import UEMethod
from run_mcts.src.scoring_methods import ScoringMethod, LengthNormalizedScoring
import numpy as np


class PredictiveEntropy(UEMethod):
    def __init__(self, scoring_function : ScoringMethod = LengthNormalizedScoring()):#normalization, 
        super().__init__()
        self.scoring_function = scoring_function
    
    def _entropy(self, scores: list[float]):
        entropy = -np.sum(scores) / len(scores)
        return entropy
    
    def __call__(self, sampled_gen_dict):
        
        # Apply Score function
        scores_ = [self.scoring_function(logp) for logp in sampled_gen_dict['logprobs']]
        score = self._entropy(scores_)
        
        return {
            "confidence": -score,
            "uncertainty": score
        } 
