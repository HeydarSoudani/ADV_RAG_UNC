import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import numpy as np

from utils.general_utils import find_token_indices
from run_uncertainty_estimation.src.ue_templates import (
    PTRUE_SYSTEM_PROMPT,
    PTRUE_USER_PROMPT,
    PTRUE_USER_PROMPT_WITH_CONTEXT,
    PTRUE_MODEL_OUTPUT
)


class PTrue:
    def __init__(
        self,
        model,
        tokenizer,
        number_of_ideas: int = 5,
        system_prompt: str = PTRUE_SYSTEM_PROMPT,
        user_prompt: str = PTRUE_USER_PROMPT,
        model_output: str = PTRUE_MODEL_OUTPUT,
        with_context: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.number_of_ideas = number_of_ideas
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.model_output = model_output
        self.with_context = with_context
        if with_context:
            print("Context field is required in user prompt for with_context=True, swithing to the default user prompt with context")
            self.user_prompt = PTRUE_USER_PROMPT_WITH_CONTEXT
    
    def __call__(self, sampled_gen_dict, prediction, context:str):
        ideas = sampled_gen_dict.get("generated_texts", []) or []
        ideas = [str(idea) for idea in ideas if idea is not None]  # Filter out None values and ensure strings
        ideas = "\n".join(ideas)
        
        if self.with_context == False:
            chat = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": self.user_prompt.format(
                        question=sampled_gen_dict['question'],
                        ideas=ideas,
                        generated_text=prediction,
                    ),
                },
                {"role": "assistant", "content": self.model_output},
            ]
        else:
            chat = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": self.user_prompt.format(
                        question=sampled_gen_dict['question'],
                        ideas=ideas,
                        generated_text=prediction,
                        context=context,
                    ),
                },
                {"role": "assistant", "content": self.model_output},
            ]
        
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False)
        prompt_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        self.model.eval()
        with torch.inference_mode():
            outputs = self.model(prompt_tokens, use_cache=False, output_attentions=False, output_hidden_states=False)
            logits = outputs.logits
            logits = outputs.logits.detach().cpu()   # <<< move off GPU before softmax
            del outputs
            torch.cuda.empty_cache() 

        idx = prompt_tokens[0][1:].cpu().view(-1, 1)  # just for gather()
        logprobs = torch.log_softmax(logits, dim=-1)
        logprobs = logprobs[0, :-1, :]
        logprobs = torch.gather(logprobs, dim=1, index=idx)
        logprobs = logprobs.view(-1).tolist()
        indices, texts = find_token_indices(prompt_tokens[0][1:], self.tokenizer, "true")

        loss_true = 0
        for index in indices[-1]:  # only look at the last occurence of the word true
            loss_true += logprobs[index]

        loss_true = loss_true / len(indices[-1])  # length normalization
        prob_true = np.exp(loss_true).item()
        
        return {
            "confidence": prob_true,
            "uncertainty": -prob_true
        }