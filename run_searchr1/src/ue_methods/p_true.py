import torch
import numpy as np
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from run_searchr1.src.ue_methods.common_utils import find_token_indices

PTRUE_SYSTEM_PROMPT = 'You are a helpful, respectful and honest question-answer evaluator. You will be given a question, some brainstormed ideas and a generated answer. Evaluate the generate answer as true or false considering the question and brainstormed ideas. Output "The generated answer is true" or "The generated answer is false".'
PTRUE_USER_PROMPT = "Question:{question}\nHere are some ideas that were brainstormed:{ideas}\nGenerated answer:{generated_text}"
PTRUE_USER_PROMPT_WITH_CONTEXT = "Context:{context}\nQuestion:{question}\nHere are some ideas that were brainstormed:{ideas}\nGenerated answer:{generated_text}"
PTRUE_MODEL_OUTPUT = "The generated answer is true"


class PTrue:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
        system_prompt: str = PTRUE_SYSTEM_PROMPT,
        user_prompt: str = PTRUE_USER_PROMPT,
        model_output: str = PTRUE_MODEL_OUTPUT,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.model_output = model_output
    
    
    def ptrue_uncertainty(self, question, ideas, prediction):
        ideas = "\n".join(ideas)
        chat = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt.format(question=question,ideas=ideas, generated_text=prediction)},
            {"role": "assistant", "content": self.model_output},
        ]
        
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False)
        prompt_tokens = self.tokenizer.encode(
            prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(prompt_tokens)
            logits = outputs.logits

        logprobs = torch.log_softmax(logits, dim=-1)
        logprobs = logprobs[0, :-1, :]
        logprobs = torch.gather(logprobs, dim=1, index=prompt_tokens[0][1:].view(-1, 1))
        logprobs = logprobs.view(-1).tolist()
        indices, texts = find_token_indices(prompt_tokens[0][1:], self.tokenizer, "true")

        loss_true = 0
        for index in indices[-1]:
            loss_true += logprobs[index]
        loss_true = loss_true / len(indices[-1])
        prob_true = np.exp(loss_true).item()
        uncertainty_score = -prob_true
        
        return uncertainty_score
        