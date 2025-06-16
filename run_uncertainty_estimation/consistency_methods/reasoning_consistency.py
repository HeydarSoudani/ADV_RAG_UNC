import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class ReasoningConsistency:
    def __init__(self, rag_model, args):
        self.args = args
        self.rag_model = rag_model
    
    def get_masked_traces(self, qid, question, trace):
        pass