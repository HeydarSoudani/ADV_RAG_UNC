
import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    DebertaForSequenceClassification,
    DebertaTokenizer,
)
from .ue_method import UEMethod
from run_mcts.src.scoring_methods import ScoringMethod, LengthNormalizedScoring
from utils.general_utils import bidirectional_entailment_clustering


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

class SemanticEntropy(UEMethod):
    def __init__(self,
        scoring_function : ScoringMethod = LengthNormalizedScoring(),
        model_for_entailment: PreTrainedModel = None,
        tokenizer_for_entailment: PreTrainedTokenizer = None,
        entailment_model_device="cuda",
    ):
        super().__init__()
        if model_for_entailment is None or tokenizer_for_entailment is None:
            self.model_for_entailment = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to(entailment_model_device)
            self.tokenizer_for_entailment = DebertaTokenizer.from_pretrained("microsoft/deberta-large-mnli")
        
        self.scoring_function = scoring_function
    
    def _semantic_entropy(
        self,
        generated_texts: list[str],
        question: str,
        generated_outputs: list,
    ):
        clusters = bidirectional_entailment_clustering(
            self.model_for_entailment,
            self.tokenizer_for_entailment,
            question,
            generated_texts,
        )
        total_output_for_log = calculate_total_log(generated_outputs, clusters)

        return {
            "clusters": clusters,
            "semantic_entropy": total_output_for_log,
        }
    
    def __call__(self, sampled_gen_dict):
        generated_outputs = []
        number_of_generations = len(sampled_gen_dict['logits'])
        for i in range(number_of_generations):
            text = sampled_gen_dict['generated_texts'][i]
            score = self.scoring_function(sampled_gen_dict["logprobs"][i])
            generated_outputs.append((text, score))

        se_dict = self._semantic_entropy(
            sampled_gen_dict['generated_texts'],
            sampled_gen_dict["question"],
            generated_outputs
        )
        
        # print(se_dict['clusters'])
        return {
            "confidence": -se_dict['semantic_entropy'],
            "uncertainty": se_dict['semantic_entropy'],
            "clusters": se_dict['clusters']
        } 

