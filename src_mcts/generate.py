
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from typing import List, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.general_utils import read_txt

class Generator:
    """Generator generates children nodes"""

    def __init__(self, args, evaluator) -> None:
        self.args = args
        self.generation_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=False
        )
        self.evaluator = evaluator
        
        self.num_subquestions = args.num_subquestions
        self.num_votes = args.num_votes
        self.max_tokens = args.max_tokens
        self.enable_potential_score = args.enable_potential_score
        self.mcts_num_last_votes = args.mcts_num_last_votes
        
        # Actions' prompts
        self.a1_direct_answer_prompt = read_txt()
        self.a2_retrieve_doc_prompt = read_txt()
        self.a3_query_decomposition = read_txt()
        self.a4_query_reformulation = read_txt()
        
        # 
    
    def _extract_from_cache(self, subquestion_list: List[str]):
        pass
    
    def _get_most_likely_answer(self, io_output_list: List[str]) -> Tuple[str, float]:
        pass
    
    def _fewshot_cot_answer_question(self, question: str, paraphrased: bool, num_return: int, hint: str = None):
        pass
    
    def generate_direct_answers(self, user_question: str, paraphrased: bool, hint: str):
        pass
    
    def generate_retrieve_docs(self, user_question: str, paraphrased: bool, hint: str):
        pass
        
    def generate_query_decomposition(self, user_question: str, paraphrased: bool, hint: str):
        pass
    
    def generate_query_reformulation(self, user_question: str, paraphrased: bool, hint: str):
        pass
        