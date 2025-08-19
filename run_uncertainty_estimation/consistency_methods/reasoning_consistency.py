import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import copy
import random
from collections import Counter


class ReasoningConsistency:
    def __init__(self, rag_model, args):
        self.args = args
        self.rag_model = rag_model
        
    # this is like rag_consistency, but without applying actions
    def get_masked_traces(self, qid, question, prediction, trace):
        
        # -- Read search-query indices
        masked_traces, answer_output_list = [], []
        if self.args.rag_method == 'self_ask':
            has_search = len(trace) > 2
            think_search_indices = range(1, len(trace)-1)
        elif self.args.rag_method == 'react':
            think_search_indices = [idx for idx, step in enumerate(trace[:-1]) if step['action_type']=='search']
            has_search = len(think_search_indices) > 0
        elif self.args.rag_method in ['flare', 'dragin']:
            think_search_indices = [idx for idx, step in enumerate(trace[:-1]) if len(step['search_query']) > 0]
            has_search = len(think_search_indices) > 0
        else:
            has_search = len(trace) > 1
            think_search_indices = range(0, len(trace)-1)
            
        if has_search:
            random_points = [random.choice(think_search_indices) for _ in range(self.args.n_generations)]
            point_counts = Counter(random_points)
            selected_indices_group = [(index, repeat) for index, repeat in point_counts.items()]
            print(selected_indices_group)
            
            for (selected_index, repeat) in selected_indices_group:
                for i in range(repeat):
                    new_trace = copy.deepcopy(trace[:selected_index])
                    pred_answer, rest_of_trace = self.rag_model.partial_inference_middle_step(question, new_trace)
                    new_trace.extend(rest_of_trace)
                    masked_traces.append(new_trace)
                    answer_output_list.append(pred_answer.strip() if pred_answer else '')
        else:
            # ==============================
            # -- Do self-consistency -------
            # ==============================
            for _ in range(self.args.n_generations):
                final_ans, new_trace = self.rag_model.inference(question, generation_temp=self.args.consistency_temperature)
                masked_traces.append(new_trace)
                answer_output_list.append(final_ans)
        
        # Convert mased trace to text
        masked_traces_text = [
            self.rag_model.get_input_prompt_without_final_answer(question, masked_trace)
            for masked_trace in masked_traces
        ]
        
        return masked_traces, masked_traces_text, answer_output_list