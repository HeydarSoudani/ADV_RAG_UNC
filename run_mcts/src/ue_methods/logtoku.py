# Ref: https://github.com/MaHuanAAA/logtoku/blob/main/SenU/metrics.py

import torch
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from .ue_method import UEMethod
from src_mcts.scoring_methods import ScoringMethod, LengthNormalizedScoring

class LogTokU(UEMethod):
    def __init__(self, scoring_function : ScoringMethod = LengthNormalizedScoring()):#normalization, 
        super().__init__()
        self.scoring_function = scoring_function
    
    def __call__(self, 
        model: PreTrainedModel,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        context: list[dict],
        question: str,
        generated_text: str=None
    ):
        # === For existing generation
        if generated_text == None: # based on multiple generations
            pass
        else:
            # Input preparation
            input_text = self.get_prompt_text(context, question)
            if tokenizer.chat_template:
                input_prompt_text = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": input_text},
                        {"role": "assistant", "content": generated_text}
                    ],
                    add_generation_prompt=True,
                    tokenize=False
                )

            # Generation
            generated_text_ids = tokenizer.encode(generated_text, add_special_tokens=False)
            generated_text_tokens_str = [tokenizer.decode([answer_id]) for answer_id in generated_text_ids]
            input_prompt_tokens = tokenizer.encode(input_prompt_text, return_tensors="pt").to(model.device)
            indices, texts = find_token_indices(input_prompt_tokens[0], tokenizer, generated_text)
            
            with torch.no_grad():
                outputs = model(input_prompt_tokens)
                logits = outputs.logits
            # print(logits.shape)
            
            # ...
            
            