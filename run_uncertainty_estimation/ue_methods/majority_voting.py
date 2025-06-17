

class MajorityVoting:
    def __init__(self, se_model):
        self.se_model = se_model
    
    def cluster2prob(self, clusters):
        total = sum(len(inner) for inner in clusters)
        result = {inner[0]: len(inner) / total for inner in clusters}
        return result

    def __call__(self, sampled_gen_dict, prediction, context:str):
        question = sampled_gen_dict['question']
        generated_texts = sampled_gen_dict['generated_texts']
        len_generated_texts = len(generated_texts)
        
        num_consistent = sum(
            self.se_model.check_answers_equiv(question, prediction, ans)
            for ans in generated_texts
        )
        conf = num_consistent / len_generated_texts
        
        # Get most confident 
        clusters = self.se_model.cluster_by_meaning(question, generated_texts)
        candidates_with_prob = self.cluster2prob(clusters)
        most_confident = max(candidates_with_prob.items(), key=lambda x: x[1])
        return {
            "confidence": conf,
            "uncertainty": -conf,
            "most_confident_answer": most_confident
        }
    
        