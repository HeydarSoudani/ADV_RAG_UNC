import torch
import numpy as np
from typing import Union
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast


class SAR:
    def __init__(
        self,
        tokenizer,
        t=0.001,
        model_for_similarity=None,
        similarity_model_device="cuda",
        
    ):
        if model_for_similarity is None:
            self.model_for_similarity = CrossEncoder(
                "cross-encoder/stsb-roberta-large",
                num_labels=1,
                device=similarity_model_device,
            )
        else:
            self.model_for_similarity = model_for_similarity
        self.t = t
        self.tokenizer = tokenizer
    
    def _sentsar(
        self,
        generated_texts: list[str],
        question: str,
        scores: list[float],
    ):

        similarities = {}
        for i in range(len(generated_texts)):
            similarities[i] = []

        for i in range(len(generated_texts)):
            for j in range(i + 1, len(generated_texts)):
                gen_i = question + generated_texts[i]
                gen_j = question + generated_texts[j]
                similarity_i_j = self.model_for_similarity.predict(
                    [gen_i, gen_j])
                similarities[i].append(similarity_i_j)
                similarities[j].append(similarity_i_j)

        probs = torch.exp(torch.tensor(scores))
        assert len(probs) == len(similarities)

        sentence_scores = []
        for idx, prob in enumerate(probs):
            w_ent = -torch.log(
                prob
                + (
                    (torch.tensor(similarities[idx]) / self.t)
                    * torch.cat([probs[:idx], probs[idx + 1:]])
                ).sum()
            )
            sentence_scores.append(w_ent)
        sentence_scores = torch.tensor(sentence_scores)

        entropy = (
            torch.sum(sentence_scores, dim=0) /
            torch.tensor(sentence_scores.shape[0])
        ).item()
        
        return entropy, similarities
    
    def _tokensar(
        self,
        question: str,
        generated_text: str,
        tokens: list[int],
        logprobs: list[float],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ):
        importance_vector = []
        for i in range(len(tokens)):
            removed_answer_ids = tokens[:i] + tokens[i + 1:]
            removed_answer = tokenizer.decode(
                removed_answer_ids, skip_special_tokens=True
            )
            score = self.model_for_similarity.predict(
                [
                    (
                        question + " " + removed_answer,
                        question + " " + generated_text,
                    )
                ]
            )
            score = 1 - score[0]
            importance_vector.append(score)

        importance_vector = importance_vector / np.sum(importance_vector)
        return np.dot(importance_vector, logprobs)

    def __call__(self, sampled_gen_dict, prediction, context):
        generated_texts = sampled_gen_dict["generated_texts"]
        generated_tokens = sampled_gen_dict["tokens"]
        logprobs = sampled_gen_dict["logprobs"]
        scores = []
        for i, text in enumerate(generated_texts):
            score = self._tokensar(
                sampled_gen_dict['question'],
                text,
                generated_tokens[i],
                logprobs[i],
                self.tokenizer,
            )
            scores.append(score)  # scores are in log scale
        
        entropy, similarities = self._sentsar(generated_texts, sampled_gen_dict['question'], scores)
        
        return {
            "confidence": -entropy,
            "uncertainty": entropy
        }
        
        