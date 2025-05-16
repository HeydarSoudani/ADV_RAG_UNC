
# Src: https://github.com/Ybakman/TruthTorchLM/blob/main/src/TruthTorchLM/truth_methods/sum_eigen_uncertainty.py

from transformers import DebertaForSequenceClassification, DebertaTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from run_searchr1.src.ue_methods.common_utils import calculate_U_eigv, calculate_U_deg, calculate_U_num_set, calculate_U_ecc

class BlackBoxUncertainty:
    def __init__(
        self,
        method_for_similarity: str = "semantic",
        model_for_entailment: PreTrainedModel = None,
        tokenizer_for_entailment: PreTrainedTokenizer = None,
        entailment_model_device="cuda:0",
        temperature=3.0,
        eigen_threshold=0.9,
    ):
        self.temperature = temperature
        self.eigen_threshold = eigen_threshold
        self.method_for_similarity = method_for_similarity
        if (model_for_entailment is None or tokenizer_for_entailment is None) and method_for_similarity == "semantic":
            model_for_entailment = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to(entailment_model_device)
            tokenizer_for_entailment = DebertaTokenizer.from_pretrained("microsoft/deberta-large-mnli")
            if method_for_similarity not in ["semantic", "jaccard"]:
                raise ValueError("method_for_similarity should be either semantic or jaccard. Please refer to https://arxiv.org/pdf/2305.19187 for more information.")
            
            self.model_for_entailment = None
            self.tokenizer_for_entailment = None
            if method_for_similarity == "semantic":
                print('There are 2 methods for similarity: semantic similarity and jaccard score. The default method is semantic similarity. If you want to use jaccard score, please set method_for_similarity="jaccard". Please refer to https://arxiv.org/pdf/2305.19187 for more information.')
                self.tokenizer_for_entailment = tokenizer_for_entailment
                self.model_for_entailment = model_for_entailment
    
    def num_semantic_set_uncertainty(self, question, generated_texts):
        # torch.cuda.empty_cache()
        uncertainty_score = calculate_U_num_set(
            generated_texts,
            question,
            method_for_similarity=self.method_for_similarity,
            model_for_entailment=self.model_for_entailment,
            tokenizer_for_entailment=self.tokenizer_for_entailment,
        )
        return uncertainty_score
    
    def sum_eigen_uncertainty(self, question, generated_texts):
        uncertainty_score = calculate_U_eigv(
            generated_texts,
            question,
            method_for_similarity=self.method_for_similarity,
            temperature=self.temperature,
            model_for_entailment=self.model_for_entailment,
            tokenizer_for_entailment=self.tokenizer_for_entailment,
        )
        return uncertainty_score
    
    def eccentricity_uncertainty(self,  question, generated_texts):
        uncertainty_score = calculate_U_ecc(
            generated_texts,
            question,
            method_for_similarity=self.method_for_similarity,
            temperature=self.temperature,
            eigen_threshold=self.eigen_threshold,
            model_for_entailment=self.model_for_entailment,
            tokenizer_for_entailment=self.tokenizer_for_entailment,
        )
        return uncertainty_score
    
    def matrix_degree_uncertainty(self, question, generated_texts):
        uncertainty_score = calculate_U_deg(
            generated_texts,
            question,
            method_for_similarity=self.method_for_similarity,
            temperature=self.temperature,
            model_for_entailment=self.model_for_entailment,
            tokenizer_for_entailment=self.tokenizer_for_entailment,
        )
        return uncertainty_score
    