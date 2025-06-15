

class PTrue:
    def __init__(self, scoring_function : ScoringMethod = LengthNormalizedScoring()):
        super().__init__()
        self.scoring_function = scoring_function
    
    def __call__(self, sampled_gen_dict):
        
        prob_true = None
        
        return {
            "confidence": prob_true,
            "uncertainty": -prob_true
        }