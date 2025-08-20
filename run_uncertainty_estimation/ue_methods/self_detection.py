class SelfDetection:
    def __init__(
        self,
    ):
        pass

    def __call__(self, sampled_gen_dict, prediction, context):
        question = sampled_gen_dict['question']
        generated_texts = sampled_gen_dict["generated_texts"]
        generated_tokens_texts = sampled_gen_dict["tokens_text"]
        logprobs = sampled_gen_dict["logprobs"]
        
        total_score = 0.0
        for i, generated_text in enumerate(generated_texts):
            pass
        
        score = total_score / len(generated_texts)
        return {
            "confidence": score,
            "uncertainty": -score
        } 