import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class SelfConsistency:
    def __init__(self, rag_model, args):
        self.args = args
        self.rag_model = rag_model
        
    def get_masked_traces(self, qid, question, prediction, trace):
        # Convert trace to input prompt
        answer_output_list = self.rag_model.partial_inference_self_consistency(question, trace)
        masked_traces = []
        for answer_output in answer_output_list:
            new_trace = trace
            new_trace[-1]['answer'] = answer_output
            masked_traces.append(new_trace)
        
        masked_traces_text = [
            self.rag_model.get_input_prompt_self_consistency(question, masked_trace)
            for masked_trace in masked_traces
        ]
            
        return masked_traces, masked_traces_text, answer_output_list