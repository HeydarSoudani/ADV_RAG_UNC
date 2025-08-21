import torch
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from transformers import DebertaForSequenceClassification, DebertaTokenizer

from utils.general_utils import bidirectional_entailment_clustering
from .semantic_entropy import calculate_total_log

class LARS:
    def __init__(
        self,
        ue_type: str = "semantic_entropy",
        lars_model: PreTrainedModel = None,
        lars_tokenizer: PreTrainedTokenizer = None,
        device="cuda",
        model_for_entailment: PreTrainedModel = None,
        tokenizer_for_entailment: PreTrainedTokenizer = None,
        entailment_model_device="cuda",
    ):
        assert ue_type in [
            "confidence",
            "semantic_entropy",
            "se",
            "entropy",
        ], f"ue_type must be one of ['confidence', 'semantic_entropy', 'se', 'entropy'] but it is {ue_type}."
        self.ue_type = ue_type
        
        # lars model
        if lars_model is None or lars_tokenizer is None:
            lars_model = AutoModelForSequenceClassification.from_pretrained("duygunuryldz/LARS").to(device)
            lars_tokenizer = AutoTokenizer.from_pretrained("duygunuryldz/LARS")
        self.lars_model = lars_model
        self.lars_tokenizer = lars_tokenizer
        self.device = device
        self.number_of_bins = (lars_model.config.number_of_bins)
        self.edges = (lars_model.config.edges) 
        
        # params for semantic entropy
        if (ue_type == "se" or ue_type == "semantic_entropy") and (model_for_entailment is None or tokenizer_for_entailment is None):
            model_for_entailment = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to(entailment_model_device)
            tokenizer_for_entailment = DebertaTokenizer.from_pretrained("microsoft/deberta-large-mnli")

        self.model_for_entailment = model_for_entailment
        self.tokenizer_for_entailment = tokenizer_for_entailment

    @staticmethod
    def _find_bin(value, edges, number_of_bins):
        if edges is not None:
            bin_index = np.digitize(value, edges, right=False)
        else:
            bin_index = int(
                value * number_of_bins
            )  # discretize the probability space equally
        return min(bin_index, (number_of_bins - 1))

    @staticmethod
    def prepare_answer_text(probs, answer_tokens, edges, number_of_bins):
        a_text = ""
        if len(probs) != len(answer_tokens):
            # print("LARS: probs is not equal answer_tokens!!")
            min_len = min(len(probs), len(answer_tokens))
            probs, answer_tokens = probs[:min_len], answer_tokens[:min_len]
        
        for i, tkn_text in enumerate(answer_tokens):
            bin_id = LARS._find_bin(probs[i], edges, number_of_bins)
            a_text += tkn_text + f"[prob_token_{bin_id}]"
        
        return a_text

    @staticmethod
    def tokenize_input(tokenizer, question, answer_text):
        tokenized_input = tokenizer(
            question,
            answer_text,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            return_token_type_ids=True,
            is_split_into_words=False,  # ???
            truncation=True,
            max_length=None,
            padding="max_length",
        )
        return tokenized_input

    def _lars(self, question, generation_token_texts, probs):
        a_text = LARS.prepare_answer_text(probs, generation_token_texts, self.edges, self.number_of_bins)
        tokenized_input = LARS.tokenize_input(self.lars_tokenizer, question, a_text)
        input_ids = (torch.tensor(tokenized_input["input_ids"]).reshape(1, -1).to(self.device))
        attention_mask = (torch.tensor(tokenized_input["attention_mask"]).reshape(1, -1).to(self.device))
        token_type_ids = (torch.tensor(tokenized_input["token_type_ids"]).reshape(1, -1).to(self.device))
        with torch.no_grad():
            self.lars_model.eval()
            logits = self.lars_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits.detach()

        return torch.nn.functional.sigmoid(logits[:, 0]).item()
         
    def __call__(self, sampled_gen_dict, prediction, context):
        question = sampled_gen_dict['question']
        generated_texts = sampled_gen_dict["generated_texts"]
        generated_tokens_texts = sampled_gen_dict["tokens_text"]
        logprobs = sampled_gen_dict["logprobs"]
        
        if self.ue_type == "confidence":
            pass # TODO
        
        elif self.ue_type in ["semantic_entropy", "se", "entropy"]:
            scores, generated_outputs = [], []
            for i, generated_text in enumerate(generated_texts):
                tokens_text = generated_tokens_texts[i]
                probs = torch.exp(torch.tensor(logprobs[i]))
                probs = np.exp(np.array(logprobs[i]))
                score = torch.log(torch.tensor(self._lars(question, tokens_text, probs))).item()
                scores.append(score)
                generated_outputs.append((generated_text, score))
        
            if self.ue_type == "semantic_entropy" or self.ue_type == "se":
                clusters = bidirectional_entailment_clustering(
                    self.model_for_entailment,
                    self.tokenizer_for_entailment,
                    question,
                    generated_texts,
                )
                lars_score = -calculate_total_log(generated_outputs, clusters)

            elif self.ue_type == "entropy":
                lars_score = np.sum(scores) / len(scores)

        return {
            "confidence": lars_score,
            "uncertainty": -lars_score
        } 