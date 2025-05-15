
import torch
import numpy as np
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from .ue_method import UEMethod
from run_mcts.src.scoring_methods import ScoringMethod, LengthNormalizedScoring


class Entropy(UEMethod):
    def __init__(self, aggregation_function="max", scoring_function : ScoringMethod = LengthNormalizedScoring()):#normalization, 
        super().__init__()
        self.scoring_function = scoring_function
        self.aggregation_function = aggregation_function

    def __call__(self, sampled_gen_dict):
        
        # if generated_text == None: # based on multiple generations
        #     pass
        # else:
        #     # Input preparation
        #     input_text = self.get_prompt_text(context, question)
        #     if tokenizer.chat_template:
        #         input_prompt_text = tokenizer.apply_chat_template(
        #             [
        #                 {"role": "system", "content": self.system_prompt},
        #                 {"role": "user", "content": input_text},
        #                 {"role": "assistant", "content": generated_text}
        #             ],
        #             add_generation_prompt=True,
        #             tokenize=False
        #         )
            
        #     # Generation
        #     generated_text_ids = tokenizer.encode(generated_text, add_special_tokens=False)
        #     generated_text_tokens_str = [tokenizer.decode([answer_id]) for answer_id in generated_text_ids]
        #     input_prompt_tokens = tokenizer.encode(input_prompt_text, return_tensors="pt").to(model.device)
        #     indices, texts = find_token_indices(input_prompt_tokens[0], tokenizer, generated_text)
            
        #     with torch.no_grad():
        #         outputs = model(input_prompt_tokens)
        #         logits = outputs.logits
         
        entropies = []
        number_of_generations = len(sampled_gen_dict['logits'])
        for i in range(number_of_generations):
            logits = sampled_gen_dict['logits'][i]
            logprobs = torch.log_softmax(logits, dim=-1)
        
            p_log_p = torch.exp(logprobs) * logprobs 
            tokens_entropy = -torch.sum(p_log_p, axis=1, keepdims=True) 
        
            if self.aggregation_function == 'min':
                entropy = torch.min(tokens_entropy).item()
            elif self.aggregation_function == 'max':
                entropy = torch.max(tokens_entropy).item()
            elif self.aggregation_function == 'mean':
                entropy = torch.mean(tokens_entropy).item()
            else:
                raise NotImplementedError("aggregation function is not implemented!")
    
            entropies.append(entropy)
        
        entropy_mean = sum(entropies) / len(entropies)

        return {
            "confidence": -entropy_mean,
            "uncertainty": entropy_mean,
        }