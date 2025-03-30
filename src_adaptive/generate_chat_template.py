import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
from math import exp
from scipy.special import softmax
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from utils.adaptive_utils import fix_tokenizer_chat
from src_adaptive.templetes import SYSTEM_PROMPT_LONGFORM, SYSTEM_PROMPT_SHORTFORM, SYSTEM_PROMPT_REGENERATE


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
        self.eos_token_ids = self.generator.config.eos_token_id
        
        if args.model_name_or_path in ["Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"]:
            self.space_token = "Ġ"
        elif args.model_name_or_path == "meta-llama/Llama-2-7b-chat-hf":
            self.space_token = "▁"
        else:
            self.space_token = self.tokenizer.tokenize(' ')[0]
    
    
    def generate(self,
        input_text,
        max_new_tokens,
        system_prompt:str = SYSTEM_PROMPT_LONGFORM,
        return_logprobs=True,
        return_text=True,
        add_generation_prompt=True,
        continue_final_message=False
    ):
        # === Add template
        messages = [{'role': 'system', 'content': system_prompt}]
        messages.append({'role': 'user', 'content': input_text})
        tokenizer, messages = fix_tokenizer_chat(self.tokenizer, messages)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message
        )
        # print(f"{text}\n")
        
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt").to(self.generator.device)
            input_ids = inputs['input_ids']
            model_output = self.generator.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_logits=True,
                output_scores = True,
                eos_token_id=self.eos_token_ids
            )

            model_output.past_key_values=None
            model_output.sequences = model_output.sequences.cpu()
            if type(self.eos_token_ids) == list:
                temp = torch.stack([
                    torch.argmax((model_output.sequences[:, len(input_ids[0]):] == eos).to(dtype=torch.int), dim=-1) 
                    for eos in self.eos_token_ids
                ]).T
                # indices = [torch.min(temp[i][temp[i]>0]).item() for i in range(len(temp))]
                # ------------------------------
                # Mine: Llama 3 generates error
                # ------------------------------
                indices = []
                for i in range(len(temp)):
                    non_zero_elements = temp[i][temp[i] > 0]
                    if non_zero_elements.numel() > 0:
                        indices.append(torch.min(non_zero_elements).item())
                    else:
                        indices.append(0)  # Handle the case where no EOS token is found
                # ------------------------------
            else:
                indices = torch.argmax((model_output.sequences[:, len(input_ids[0]):] == self.eos_token_ids).to(dtype=torch.int), dim=-1)
            indices[indices==0] = model_output.sequences.shape[1] - len(input_ids[0]) -1
            
            if return_text:
                tokens = [seq[len(input_ids[0]):indices[i] + len(input_ids[0])+1].tolist() for i, seq in enumerate(model_output.sequences)]
                tokens_text = [[tokenizer.decode(token) for token in tokens_] for tokens_ in tokens]
                generated_texts = tokenizer.batch_decode(tokens, skip_special_tokens=True)
            
            if return_logprobs:
                logits_list = torch.stack(model_output.logits).cpu().permute(1, 0, 2)
                model_output.logits = None
                logprobs = torch.log_softmax(logits_list, dim=-1) #logprobs for each token
                logprobs = torch.gather(logprobs, dim=-1, index = model_output.sequences[:, len(input_ids[0]):].unsqueeze(-1))#logprobs for each token in the generated text
                logprobs = logprobs.squeeze(-1).tolist()#convert to list
                logprobs = [logprobs[i][:indices[i]+1] for i in range(len(logprobs))]
                assert len(tokens_text[0]) == len(logprobs[0])
                
        return generated_texts[0], tokens_text[0], logprobs[0]
    
    
    def generate_attn(self,
        input_text,
        max_new_tokens,
        system_prompt:str = SYSTEM_PROMPT_LONGFORM,
        solver="max",
        use_entropy = False,
        use_logprob = False,
        add_generation_prompt=True,
        continue_final_message=False
    ):
        # === Add template
        messages = [{'role': 'system', 'content': system_prompt}]
        messages.append({'role': 'user', 'content': input_text})
        tokenizer, messages = fix_tokenizer_chat(self.tokenizer, messages)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message
        )
        
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt").to(self.generator.device)
            input_ids = inputs['input_ids']
            model_output = self.generator.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_logits=True,
                output_scores = True,
                output_attentions=True,
                eos_token_id=self.eos_token_ids
            )

            model_output.past_key_values=None
            # model_output.sequences = model_output.sequences.cpu()
            if type(self.eos_token_ids) == list:
                temp = torch.stack([
                    torch.argmax((model_output.sequences[:, len(input_ids[0]):] == eos).to(dtype=torch.int), dim=-1) 
                    for eos in self.eos_token_ids
                ]).T
                # indices = [torch.min(temp[i][temp[i]>0]).item() for i in range(len(temp))]
                # ------------------------------
                # Mine: Llama 3 generates error
                # ------------------------------
                indices = []
                for i in range(len(temp)):
                    non_zero_elements = temp[i][temp[i] > 0]
                    if non_zero_elements.numel() > 0:
                        indices.append(torch.min(non_zero_elements).item())
                    else:
                        indices.append(0)  # Handle the case where no EOS token is found
                # ------------------------------
            else:
                indices = torch.argmax((model_output.sequences[:, len(input_ids[0]):] == self.eos_token_ids).to(dtype=torch.int), dim=-1)
            indices[indices==0] = model_output.sequences.shape[1] - len(input_ids[0]) -1
            
        
            generation_id = 0 # We have only one generation
            token_ids = [seq[len(input_ids[0]):indices[i] + len(input_ids[0])+1].tolist() for i, seq in enumerate(model_output.sequences)]
            tokens = [tokenizer.convert_ids_to_tokens(tokens_) for tokens_ in token_ids][generation_id]
            text = tokenizer.batch_decode(token_ids, skip_special_tokens=True)[generation_id]
            
            range_ = [] # Convert tokens to entities
            for i, t in enumerate(tokens):
                if i == 0 or t.startswith(self.space_token) or token_ids[generation_id][i] == 13 or tokens[i-1] in ['</s>', '<|eot_id|>']:
                    range_.append([i, i])
                else:
                    range_[-1][-1] += 1
            
            # Get attention
            attentions_last_token_last_layer = model_output.attentions[-1][0]
            if solver == "max": 
                mean_atten, _ = torch.max(attentions_last_token_last_layer, dim=1)
                mean_atten = torch.mean(mean_atten, dim=0)
            elif solver == "avg":
                mean_atten = torch.sum(attentions_last_token_last_layer, dim=1)
                mean_atten = torch.mean(mean_atten, dim=0)
                for i in range(mean_atten.shape[0]):
                    mean_atten[i] /= (mean_atten.shape[0] - i)
            elif solver == "last_token":
                mean_atten = torch.mean(attentions_last_token_last_layer[:, -1], dim=0)
            else:
                raise NotImplementedError
            if mean_atten.shape[0] > 1 and tokens[0] in ['</s>', '<|eot_id|>']:
                mean_atten = mean_atten / sum(mean_atten[1:]).item()
            
            seqlist = []
            attns = []
            for r in range_:
                tokenseq = "".join(tokens[r[0]: r[1]+1]).replace(self.space_token, "")
                # value = mean_atten[r[0]: r[1]+1].sum().item()    # Original 
                value = mean_atten[0][r[0]: r[1]+1].sum().item() # Mine
                seqlist.append(tokenseq)
                attns.append(value)
            
            
            # Both methods output same results
            if use_logprob:
                # Method 1
                # transition_scores = self.generator.compute_transition_scores(
                #     model_output.sequences, model_output.scores, normalize_logits=True
                # )
                # logprobs = transition_scores[0]
                # logprobs = [p.cpu().numpy() for p in logprobs]
                # assert len(tokens) == len(logprobs)
                # seqlogprobs = []
                # for r in range_:
                #     logprobseq = sum(logprobs[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                #     seqlogprobs.append(logprobseq)
                
                # Method 2
                logits = model_output.scores
                log_probs = [torch.nn.functional.log_softmax(step_logits, dim=-1) for step_logits in logits]
                generated_tokens = model_output.sequences[:, -len(log_probs):]  # Only the new tokens
                token_log_probs = [
                    log_probs[i][torch.arange(generated_tokens.shape[0]), generated_tokens[:, i]]
                    for i in range(len(log_probs))
                ]
                token_log_probs = [p.cpu().numpy() for p in token_log_probs]
                assert len(token_log_probs) == generated_tokens.shape[1]
                seqlogprobs = []
                for r in range_:
                    logprobseq = sum(token_log_probs[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                    seqlogprobs.append(logprobseq)
                seqlogprobs = [np.float32(p[0]) for p in seqlogprobs]
            else:
                seqlogprobs = None
            

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
        if len(fewshot_examplers) > 0:
            prompt = "Here are several examples of how to answer similar questions:\n\n"
            for exp in fewshot_examplers:
                prompt += f"Question: {exp['question']}\n"
                prompt += f"Answer: {exp['cot']} So, the answer is {exp['answer']}.\n"
            prompt += "\n"
        
        if len(docs) > 0:
            prompt += "Below are some relevant documents that may help answer the question:\n"
            for i, doc in enumerate(docs):
                prompt += f"[{i+1}] {doc}\n"
            prompt += "\n"
        
        prompt += "Now, answer the following question EXACTLY in the format of the examples above.\n"
        prompt += "DO NOT add any introductory phrases, explanations, or extra text.\n\n"
        if add_case:
            prompt += f"Question: {question}\nAnswer:"
        
        return prompt
    
    
    
    
    
    # def format_shortform(self, question, fewshot_examplers, docs, add_case=True):
    #     prompt = ""
    #     if len(docs) > 0:
    #         prompt += "Context:\n"
    #         for i, doc in enumerate(docs):
    #             prompt += f"[{i+1}] {doc}\n"
    #         prompt += "\n"
    #     if add_case:
    #         prompt += f"Question: {question}\nAnswer:"
        
    #     return prompt
    
    