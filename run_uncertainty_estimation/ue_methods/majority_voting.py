


class MajorityVoting:
    def __init__(self, se_model):
        self.se_model = se_model
    
    def __call__(self, sampled_gen_dict, prediction, context:str):
        generated_texts = sampled_gen_dict['generated_texts']
        len_generated_texts = len(generated_texts)
        question = sampled_gen_dict['question']
        num_consistent = sum(
            self.se_model.check_answers_equiv(question, prediction, ans)
            for ans in generated_texts
        )
        conf = num_consistent / len_generated_texts
        
        return {
            "confidence": conf,
            "uncertainty": -conf
        }
    
        