import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class SelfConsistency:
    def __init__(self, device, args, rag_model):
        self.args = args
        self.rag_model = rag_model
        
    
    def get_masked_traces(self, qid, question, trace):
        masked_traces = []
        
        self.rag_model.inference_with_partial_trace(question, new_trace)