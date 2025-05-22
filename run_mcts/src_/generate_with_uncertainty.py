
import os
import sys
import copy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List, Dict, Tuple, Union
import torch
import TruthTorchLM as ttlm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


from abc import ABC, abstractmethod

class CorrectnessEvaluator(ABC):
    
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, question_text:str, generated_text: str,  ground_truth_text: list[str], seed:int = None) -> int:
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def __str__(self):
        raise NotImplementedError("Subclasses must implement this method")

class ExactMatch(CorrectnessEvaluator):
    def __init__(self):
        pass
    
    def __call__(self, question_text:str, generated_text: str,  ground_truths: list[str], seed:int = None) -> bool:
        is_correct = 0
        for pa in ground_truths:
            generated_text = generated_text.strip()
            if pa in generated_text or pa.lower() in generated_text or pa.capitalize() in generated_text:
                is_correct = 1
                break
        
        return is_correct
    
    def __str__(self):
        return f"EM"




class GeneratorUNC:
    """Generator generates children nodes"""
    
    def __init__(self, args) -> None:
        self.args = args
        
        # --- Define model ------------
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=False
        )
        
        # ---
        # White-box
        pt = ttlm.truth_methods.PTrue()
        cnf = ttlm.truth_methods.Confidence()
        pe = ttlm.truth_methods.Entropy(number_of_generations=args.mcts_num_last_votes)
        se = ttlm.truth_methods.SemanticEntropy()
        mars = ttlm.truth_methods.MARS()
        lars_co = ttlm.truth_methods.LARS(ue_type='confidence')
        sar = ttlm.truth_methods.SAR()
        # Black-box
        nums = ttlm.truth_methods.NumSemanticSetUncertainty()
        eigv = ttlm.truth_methods.SumEigenUncertainty()
        ecc = ttlm.truth_methods.EccentricityUncertainty()
        deg = ttlm.truth_methods.MatrixDegreeUncertainty()
        verb = ttlm.truth_methods.VerbalizedConfidence()
        inside = ttlm.truth_methods.Inside()
        kere = ttlm.truth_methods.KernelLanguageEntropy()
    
        self.truth_methods_name = [
            'Pt', 'Conf', 'PE', 'SE', 'MARS', 'SAR', 'LARS_Co', 'INS',
            'NumS', 'EigV', 'ECC', 'Deg', 'Verb', 'KerE'
        ]
        # self.truth_methods = [
        #     pt, cnf, pe, se, mars, sar, lars_co, inside,
        #     nums, eigv, ecc, deg, verb, kere
        # ]
        
        self.truth_methods_name = ['PE', 'EigV']
        self.truth_methods = [pe, eigv]
        
          
    def get_prompt_text(self, solution_trace: Dict[int, Dict[str, str]]):
        user_query = solution_trace[str(0)]['user_question']
        
        docs, thinks, search_queries = [], [], []
        for item_idx, item_value in solution_trace.items():
            node_keys = list(item_value.keys())
            node_type = node_keys[0]
            
            if node_type in ['think_search', 'think_answer']:
                thinks.append(item_value[node_type]['think'])
                if node_type == 'think_search':
                    docs.extend(item_value[node_type]['retrieved_documents'])
                    search_queries.append(item_value[node_type]['search_query'])
                        
        # Instruction    
        input_text = 'Answer the given question.\n'
        if len(docs) > 0:
            input_text += "Below are some relevant documents that may help answer the question:\n"
            for i, doc in enumerate(docs):
                input_text += f"Doc [{i+1}]: {doc}\n"
            input_text += f"\n"
        # if len(thinks) > 0:
        #     input_text += "Below are some reasoning steps that may help answer the question:\n"
        #     for i, think in enumerate(thinks):
        #         input_text += f"[{i+1}]: {think}\n"
        #     input_text += f"\n"
        
        input_text += "Now, provide only SHORT form answers, NOT complete sentence, without any introductory phrases, explanations, or extra text.\n"
        input_text += f'Question: {user_query.strip()} Answer: '
        
        return input_text
    
    
    def generate(self, input_prompt, question):
        
        messages = [
            {"role": "system", "content": 'You are a helpful assistant. Provide only SHORT form answers, NOT complete sentence, without any additional text or explanation.'},
            {"role": "user", "content": input_prompt}
        ]
        output_hf_model = ttlm.generate_with_truth_value(
            model=self.model,
            tokenizer=self.tokenizer,
            messages=messages,
            question=question,
            truth_methods=self.truth_methods,
            generation_seed=self.args.seed
        )
        
        return output_hf_model
    
