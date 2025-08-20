import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class SelfConsistency:
    def __init__(self, rag_model, args):
        self.args = args
        self.rag_model = rag_model
        
    def get_masked_traces(self, qid, question, prediction, trace):
        masked_traces, answer_output_list = [], []
        for _ in range(self.args.n_generations):
            final_ans, new_trace = self.rag_model.inference(question, generation_temp=self.args.consistency_temperature)
            masked_traces.append(new_trace)
            final_ans_ = final_ans.strip() if final_ans else final_ans
            answer_output_list.append(final_ans_)
        
        masked_traces_text = [
            self.rag_model.get_input_prompt_without_final_answer(question, masked_trace)
            for masked_trace in masked_traces
        ]    
        
        return masked_traces, masked_traces_text, answer_output_list