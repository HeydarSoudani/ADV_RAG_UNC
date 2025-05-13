
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import spacy
import torch
import numpy as np
from math import exp
from scipy.special import softmax
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from src.templetes import SYSTEM_PROMPT_LONGFORM, SYSTEM_PROMPT_SHORTFORM, SYSTEM_PROMPT_REGENERATE

nlp = spacy.load("en_core_web_sm")


class BasicGenerator:
    def __init__(self, args):
        self.args = args
        self.generator = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map='auto'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=False
        )
        
        self.eos_token_ids = [self.generator.config.eos_token_id] + [self.tokenizer.encode(_)[-1] for _ in ['.', '\n']]
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_config = AutoConfig.from_pretrained(args.model_name_or_path)
        if self.model_config.model_type == "llama":
            self.space_token = "▁"
        else:
            self.space_token = self.tokenizer.tokenize(' ')[0]
    
    
    def generate(self,
        input_text,
        max_new_tokens, 
        system_prompt:str = SYSTEM_PROMPT_LONGFORM,
        return_logprobs=False
    ):
        print(f"{input_text}\n")
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(self.generator.device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        if return_logprobs:
            outputs = self.generator.generate(
                input_ids = input_ids, 
                attention_mask = attention_mask,
                max_new_tokens = max_new_tokens, 
                return_dict_in_generate = True, 
                output_scores = True,
            )
            transition_scores = self.generator.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )

            generated_tokens = outputs.sequences[:, input_length:]
            text = self.tokenizer.decode(generated_tokens[0]) # text = "".join(tokens)
            tokens = [self.tokenizer.decode(t) for t in generated_tokens[0]]
            logprobs = transition_scores[0]
            logprobs = [p.cpu().numpy() for p in logprobs]
            assert len(tokens) == len(logprobs)
            return text, tokens, logprobs
        
        else:
            outputs = self.generator.generate(
                input_ids = input_ids, 
                max_new_tokens = max_new_tokens, 
                attention_mask = attention_mask,
                eos_token_id=self.eos_token_ids
            )
            generated_tokens = outputs[:, input_length:]
            text = self.tokenizer.decode(generated_tokens[0])
            
            return text, None, None
        
    def generate_attn(self,
        input_text,
        max_new_tokens,
        system_prompt:str = SYSTEM_PROMPT_LONGFORM,
        solver="max", use_entropy = False, use_logprob = False
    ):        
        with torch.no_grad():
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
            input_ids = input_ids.to(self.generator.device)
            input_length = input_ids.shape[1]
            attention_mask = torch.ones_like(input_ids)     
            
            model_output = self.generator.generate(
                input_ids = input_ids, 
                attention_mask = attention_mask,
                max_new_tokens = max_new_tokens, 
                return_dict_in_generate = True, 
                output_scores = True,
                eos_token_id=self.eos_token_ids
            )
            generated_tokens = model_output.sequences[:, input_length:]
            tokens = self.tokenizer.convert_ids_to_tokens(generated_tokens[0])
            text = self.tokenizer.decode(generated_tokens[0])
            
            # merge tokens
            range_ = []
            for i, t in enumerate(tokens):
                if i == 0 or t.startswith(self.space_token) or generated_tokens[0][i] == 13 or tokens[i-1] == '</s>':
                    range_.append([i, i])
                else:
                    range_[-1][-1] += 1
        
            # attention
            # (batch_size, num_heads, 1, sequence_length)
            # atten = self.generator.generate(generated_tokens, output_attentions=True).attentions[-1][0]
            atten = self.generator.generate(generated_tokens, return_dict_in_generate=True, output_attentions=True).attentions[-1][0]
            # atten = self.generator.generate(generated_tokens, return_dict_in_generate=True, output_attentions=True).attentions[-1][-1]
            
            if solver == "max": 
                mean_atten, _ = torch.max(atten, dim=1)
                mean_atten = torch.mean(mean_atten, dim=0)
            elif solver == "avg":
                mean_atten = torch.sum(atten, dim=1)
                mean_atten = torch.mean(mean_atten, dim=0)
                for i in range(mean_atten.shape[0]):
                    mean_atten[i] /= (mean_atten.shape[0] - i)
            elif solver == "last_token":
                mean_atten = torch.mean(atten[:, -1], dim=0)
            else:
                raise NotImplementedError
            if mean_atten.shape[0] > 1 and tokens[0] == '</s>':
                mean_atten = mean_atten / sum(mean_atten[1:]).item()
        
            # regular tokens
            seqlist = []
            attns = []
            for r in range_:
                tokenseq = "".join(tokens[r[0]: r[1]+1]).replace(self.space_token, "")
                # value = mean_atten[r[0]: r[1]+1].sum().item()    # Original 
                value = mean_atten[0][r[0]: r[1]+1].sum().item() # Mine
                seqlist.append(tokenseq)
                attns.append(value)

            # -log prob
            if use_logprob:
                transition_scores = self.generator.compute_transition_scores(
                    model_output.sequences, model_output.scores, normalize_logits=True
                )
                logprobs = transition_scores[0]
                logprobs = [p.cpu().numpy() for p in logprobs]
                assert len(tokens) == len(logprobs)
                seqlogprobs = []
                for r in range_:
                    logprobseq = sum(logprobs[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                    seqlogprobs.append(logprobseq)
            else:
                seqlogprobs = None

            # entropy
            if use_entropy:
                tmp = []
                for v in model_output.scores:
                    tmp.append(v.cpu())
                softmax_probs = softmax(tmp, axis=-1)
                entropies = -np.sum(softmax_probs * np.log(softmax_probs + 1e-10), axis=-1)
                entropies = [v[0] for v in entropies]
                seqentropies = []
                for r in range_:
                    entropyseq = sum(entropies[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                    seqentropies.append(entropyseq) 
            else:
                seqentropies = None 
        
        return text, seqlist, attns, seqlogprobs, seqentropies
    
    def format_longform(self, question, fewshot_examplers, docs, add_case=True):
        prompt = ""
        for exp in fewshot_examplers:
            prompt += f"Question: {exp['question']}\n"
            prompt += f"Answer: {exp['cot']} So, the answer is {exp['answer']}.\n"
        prompt += "\n"
        
        if len(docs) > 0:
            prompt += "Context:\n"
            for i, doc in enumerate(docs):
                prompt += f"[{i+1}] {doc}\n"
            prompt += "\n"
        
        prompt += "Answer in the same format as before.\n" 
        # prompt += "Answer the following question by reasoning step-by-step, following the examples above.\n"
        if add_case:
            prompt += f"Question: {question}\nAnswer:"
        
        return prompt

    def format_regenerate(self, question, pred):
        prompt = "Answer the following question.\n"
        prompt += "DO NOT add any introductory phrases, explanations, or extra text.\n\n"
        prompt += f"Question: {question}\nAnswer: {pred}. So, the answer is"
        return prompt
