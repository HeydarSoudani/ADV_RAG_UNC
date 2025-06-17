import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    DebertaForSequenceClassification,
    DebertaTokenizer,
)

from utils.general_utils import bidirectional_entailment_clustering
from run_uncertainty_estimation.src.scoring_methods import ScoringMethod, LengthNormalizedScoring


def calculate_total_log(generated_outputs: list[str, float], clusters: list[set[str]]):
    total_output_for_log = 0
    for i, cluster in enumerate(clusters):
        score_list = []
        for elem in cluster:
            for output in generated_outputs:
                if elem == output[0]:
                    score_list.append(output[1])
        total_output_for_log -= torch.logsumexp(
            torch.tensor(score_list), dim=0).item()
    return total_output_for_log / len(clusters)


class SemanticEntropy:
    def __init__(
        self,
        model_for_entailment: PreTrainedModel = None,
        tokenizer_for_entailment: PreTrainedTokenizer = None,
        scoring_function : ScoringMethod = LengthNormalizedScoring(),
        entailment_model_device="cuda",
    ):
        if model_for_entailment is None or tokenizer_for_entailment is None:
            model_for_entailment = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to(entailment_model_device)
            tokenizer_for_entailment = DebertaTokenizer.from_pretrained("microsoft/deberta-large-mnli")

        self.model_for_entailment = model_for_entailment
        self.tokenizer_for_entailment = tokenizer_for_entailment
        self.scoring_function = scoring_function
    
    def _semantic_entropy(self, generated_texts: list[str], question: str, generated_outputs: list):
        clusters = bidirectional_entailment_clustering(
            self.model_for_entailment,
            self.tokenizer_for_entailment,
            question,
            generated_texts,
        )
        total_output_for_log = calculate_total_log(generated_outputs, clusters)
        return total_output_for_log
    
    def __call__(self, sampled_gen_dict, prediction, context):
        generated_texts = sampled_gen_dict["generated_texts"]
        generated_outputs = []
        for i, text in enumerate(generated_texts):
            score = self.scoring_function(sampled_gen_dict["logprobs"][i])
            generated_outputs.append((text, score))
        
        output = self._semantic_entropy(
            sampled_gen_dict["generated_texts"],
            sampled_gen_dict['question'],
            generated_outputs
        )
        
        return {
            "confidence": -output,
            "uncertainty": output
        } 

