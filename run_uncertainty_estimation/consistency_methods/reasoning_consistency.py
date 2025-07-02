import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import math
import copy

class ReasoningConsistency:
    def __init__(self, rag_model, args):
        self.args = args
        self.rag_model = rag_model
    
    def get_masked_traces(self, qid, question, prediction, trace):
        
        if self.args.n_generations == 1:
            interval = 0
        else:
            assert self.args.n_generations > 1
            assert self.args.mask_right_boundary >= self.args.mask_left_boundary, f"right_boundary: {self.args.mask_right_boundary} < left_boundary: {self.args.mask_left_boundary}"
            interval = (self.args.mask_right_boundary - self.args.mask_left_boundary) / (self.args.n_generations - 1)
        
        #! 1) Create partial think
        last_think = trace[-1].get('think', '')
        if last_think:
            words_in_last_think = last_think.split(" ")
            mask_len = len(words_in_last_think)
        
            masked_last_thinks = []
            for i in range(self.args.n_generations):
                prefix_part_ratio = self.args.mask_left_boundary + i * interval
                prefix_part_num_words = math.ceil(mask_len * prefix_part_ratio) + 1
                prefix_part_str = " ".join(words_in_last_think[:prefix_part_num_words])
                masked_last_thinks.append(prefix_part_str)
        else:
            masked_last_thinks = [' ']*self.args.n_generations
        
        masked_traces_ = []
        for masked_last_think in masked_last_thinks:
            new_trace = copy.deepcopy(trace)
            new_trace[-1]['think'] = masked_last_think
            masked_traces_.append(new_trace)
        
        #! 2) Generate rest
        answer_output_list, masked_traces = [], []
        for partial_trace in masked_traces_: 
            last_think_first_part = partial_trace[-1].get('think', '')
            input_prompt_text = self.rag_model.get_input_prompt_reasoning_consistency(question, partial_trace)
            last_think_second_part, final_ans = self.rag_model.partial_inference_reasoning_consistency(input_prompt_text)
            
            new_trace = copy.deepcopy(trace)
            new_trace[-1]['think'] = f"{last_think_first_part.strip()} {last_think_second_part.strip()}".strip()
            new_trace[-1]['answer'] = final_ans
            masked_traces.append(new_trace)
            answer_output_list.append(final_ans)
        
        masked_traces_text = [
            self.rag_model.get_input_prompt_self_consistency(question, masked_trace) + f"{masked_trace[-1]['think']}"
            for masked_trace in masked_traces
        ]
        
        return masked_traces, masked_traces_text, answer_output_list
            

    
        