import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class SelfConsistency:
    def __init__(self, device, args, rag_model):
        self.args = args
        self.rag_model = rag_model
        
    def get_masked_traces(self, qid, question, trace):
        
        # Convert trace to input prompt
        input_prompt_text = self.rag_model.get_input_prompt_self_consistency(question, trace)
        answer_output_list = self.rag_model.partial_inference_self_consistency(input_prompt_text)
        print(answer_output_list)
        masked_traces = []
        for answer_output in answer_output_list:
            new_trace = trace
            new_trace[-1]['answer'] = answer_output
            masked_traces.append(new_trace)
            
        return masked_traces