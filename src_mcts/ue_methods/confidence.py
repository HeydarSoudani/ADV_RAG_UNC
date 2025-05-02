from .ue_method import UEMethod
from src_mcts.scoring_methods import ScoringMethod, LengthNormalizedScoring

class Confidence(UEMethod):
    def __init__(self, scoring_function : ScoringMethod = LengthNormalizedScoring()):#normalization, 
        super().__init__()
        self.scoring_function = scoring_function
    
    def __call__(self, sampled_gen_dict):
        
        # Apply Score function
        scores_ = [self.scoring_function(logp) for logp in sampled_gen_dict['logprobs']]
        score = self.scoring_function(scores_)
        
        return {
            "confidence": score,
            "uncertainty": -score
        }