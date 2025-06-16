from transformers import DebertaForSequenceClassification, DebertaTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from utils.general_utils import calculate_U_eigv

class SumEigenUncertainty:
    def __init__(
        self,
        model_for_entailment: PreTrainedModel = None,
        tokenizer_for_entailment: PreTrainedTokenizer = None,
        method_for_similarity: str = "semantic",
        number_of_generations=5,
        temperature=3.0,
        entailment_model_device="cuda",
    ):
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

        self.number_of_generations = number_of_generations
        self.method_for_similarity = method_for_similarity  # jaccard or semantic
        self.temperature = temperature  # temperature for NLI model
    
    def __call__(self, sampled_gen_dict, prediction, context:str):
        output = calculate_U_eigv(
            sampled_gen_dict["generated_texts"],
            sampled_gen_dict['question'],
            method_for_similarity=self.method_for_similarity,
            temperature=self.temperature,
            model_for_entailment=self.model_for_entailment,
            tokenizer_for_entailment=self.tokenizer_for_entailment,
        )
        
        return {
            "confidence": -output,
            "uncertainty": output
        }