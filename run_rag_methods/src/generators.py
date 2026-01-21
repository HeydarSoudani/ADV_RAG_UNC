import os
import sys
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import transformers
from openai import OpenAI
from scipy.special import softmax


class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False

class LLMGenerator_api:
    def __init__(self, generation_model, generation_tokenizer):
        self.generator_model = generation_model
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENAI_API_KEY"))
        self.tokenizer = generation_tokenizer

        # Check if model is from Gemma family (doesn't support system role)
        self.is_gemma = 'gemma' in generation_model.lower()

        # IRCoT
        self.ircot_stopping_criteria = [".", "\n"]
        # FLARE
        self.flare_stopping_criteria = list("!@#$%^&*()\n\n)(*&^%$#@!")
        # SelfAsk - simplified for API (max 4 stop sequences)
        self.selfask_stopping_criteria = ["Intermediate answer:", "\nIntermediate answer:"]
        # SearchR1
        self.searchr1_stopping_criteria = ["</search>"]
        self.searchr1_answer_stopping_criteria = ["</answer>"]
        # SearchO1
        self.searcho1_stopping_criteria = ["<|end_search_query|>"]
        self.searcho1_answer_stopping_criteria = ["<|end_search_result|>"]

    def _process_messages(self, messages):
        """Process messages for models that don't support system role (e.g., Gemma).
        Merges system message content into the first user message."""
        if not self.is_gemma:
            return messages

        processed = []
        system_content = ""

        for msg in messages:
            if msg["role"] == "system":
                if msg["content"]:
                    system_content += msg["content"] + "\n"
            else:
                if msg["role"] == "user" and system_content:
                    # Merge system content into the first user message
                    processed.append({
                        "role": "user",
                        "content": system_content + msg["content"]
                    })
                    system_content = ""
                else:
                    processed.append(msg)

        return processed

    def generate(self,
        messages,
        stopping_criteria=None,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7
    ):
        processed_messages = self._process_messages(messages)

        # OpenAI API limits stop sequences to 4. Truncate if necessary and deduplicate.
        stop_sequences = None
        if stopping_criteria:
            # Deduplicate by stripping whitespace variants and taking unique core sequences
            unique_stops = []
            seen = set()
            for seq in stopping_criteria:
                core = seq.strip()
                if core and core not in seen:
                    seen.add(core)
                    unique_stops.append(core)
            stop_sequences = unique_stops[:4] if len(unique_stops) > 4 else unique_stops

        max_retries = 3
        retry_delay = 2  # seconds
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.generator_model,
                    messages=processed_messages,
                    temperature=temperature,
                    stop=stop_sequences,
                    max_tokens=max_new_tokens
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise

        # Handle empty or None choices (can happen with API errors or content filtering)
        if completion.choices is None or len(completion.choices) == 0:
            print(f"Warning: API returned empty choices. Completion object: {completion}")
            return "error", ""

        output_text = completion.choices[0].message.content
        # Handle None content (can happen with some models)
        if output_text is None:
            output_text = ""
        # For API: return finish_reason to check if generation ended naturally ("stop") or hit stop sequence
        # finish_reason: "stop" = natural end or stop sequence, "length" = max_tokens reached
        finish_reason = completion.choices[0].finish_reason

        return finish_reason, output_text

class LLMGenerator:
    def __init__(self, generation_model, generation_tokenizer, device, args):
        self.generator = generation_model
        self.tokenizer = generation_tokenizer
        self.device = device
        self.args = args

        self.eos_token_ids = self.generator.config.eos_token_id

        # Check if model is from Gemma family (doesn't support system role)
        model_name_lower = args.model_name_or_path.lower()
        self.is_gemma = 'gemma' in model_name_lower

        # For 'fix_length_retrieval', 'fix_sentence_retrieval', 'flare', 'dragin'
        if args.model_name_or_path in ["Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"]:
            self.space_token = "Ġ"
        elif args.model_name_or_path == "meta-llama/Llama-2-7b-chat-hf":
            self.space_token = "▁"
        else:
            self.space_token = self.tokenizer.tokenize(' ')[0]

        self.curr_eos = [151645, 151643] # for Qwen2.5 series models

        # IRCoT
        ircot_target_sequences = [".", " .", ".\n", " .\n", ".\n\n", " .\n\n", "\n", " \n", "\n\n", " \n\n"]
        self.ircot_stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(ircot_target_sequences, self.tokenizer)])

        # FLARE
        flare_target_sequences = list("!@#$%^&*()\n\n)(*&^%$#@!")
        self.flare_stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(flare_target_sequences, self.tokenizer)])

        # SelfAsk
        selfask_sequences = ["Context:", "#", "Intermediate answer:" , "Intermediate answer: ", "\nIntermediate answer:"]
        self.selfask_stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(selfask_sequences, self.tokenizer)])

        # SearchR1
        searchr1_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
        searchr1_answer_sequences = ["</answer>", " </answer>", "</answer>\n", " </answer>\n", "</answer>\n\n", " </answer>\n\n"]
        self.searchr1_stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(searchr1_sequences, self.tokenizer)])
        self.searchr1_answer_stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(searchr1_answer_sequences, self.tokenizer)])

        # SearchO1
        searcho1_sequences = ["<|end_search_query|>", " <|end_search_query|>", "<|end_search_query|>\n", " <|end_search_query|>\n", "<|end_search_query|>\n\n", " <|end_search_query|>\n\n"]
        searcho1_res_sequences = ["<|end_search_result|>", " <|end_search_result|>", "<|end_search_result|>\n", " <|end_search_result|>\n", "<|end_search_result|>\n\n", " <|end_search_result|>\n\n"]
        self.searcho1_stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(searcho1_sequences, self.tokenizer)])

        # for 'react'

    def _process_messages(self, messages):
        """Process messages for models that don't support system role (e.g., Gemma).
        Merges system message content into the first user message."""
        if not self.is_gemma:
            return messages

        processed = []
        system_content = ""

        for msg in messages:
            if msg["role"] == "system":
                if msg["content"]:
                    system_content += msg["content"] + "\n"
            else:
                if msg["role"] == "user" and system_content:
                    # Merge system content into the first user message
                    processed.append({
                        "role": "user",
                        "content": system_content + msg["content"]
                    })
                    system_content = ""
                else:
                    processed.append(msg)

        return processed
    
    def generate(self,
        messages,
        stopping_criteria=None,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
    ):
        processed_messages = self._process_messages(messages)
        if self.tokenizer.chat_template:
            input_prompt = self.tokenizer.apply_chat_template(
                processed_messages,
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            # Fallback: concatenate message contents if no chat template
            input_prompt = "\n".join(msg.get("content", "") for msg in processed_messages)

        input_ids = self.tokenizer.encode(input_prompt, return_tensors='pt').to(self.device)
        attention_mask = torch.ones_like(input_ids)
        outputs = self.generator.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=do_sample,
            temperature=temperature
        )
        output_ = outputs[0]
        generated_tokens = output_[input_ids.shape[1]:]
        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Debug: print generated tokens count and text for Gemma
        # if self.is_gemma and len(generated_tokens) < 10:
        #     print(f"[DEBUG Gemma] Generated {len(generated_tokens)} tokens")
        #     print(f"[DEBUG Gemma] Raw output (with special tokens): {self.tokenizer.decode(generated_tokens, skip_special_tokens=False)}")
        #     print(f"[DEBUG Gemma] Clean output: {output_text[:200] if output_text else 'EMPTY'}")

        return output_, output_text
    
    def generate_batch(self,
        messages,
        num_return:int = 5,
        max_new_tokens = 1024,
        temperature:float = 1.0,
        do_sample:bool = True
    ):
        processed_messages = self._process_messages(messages)
        if self.tokenizer.chat_template:
            input_prompt = self.tokenizer.apply_chat_template(
                processed_messages,
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            # Fallback: concatenate message contents if no chat template
            input_prompt = "\n".join(msg.get("content", "") for msg in processed_messages)

        inputs = self.tokenizer(input_prompt, return_tensors='pt').to(self.generator.device)
        input_ids = inputs["input_ids"]
        batch_size = input_ids.shape[0]
        
        with torch.no_grad():
            model_output = self.generator.generate(
                **inputs,
                return_dict_in_generate=True,
                output_logits=True,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=temperature,
                do_sample=do_sample,
            )
            all_generated = model_output.sequences[:, input_ids.shape[1]:]  # strip prompt

            # Group into batches
            grouped = all_generated.view(batch_size, num_return, -1)
            generated_texts = [
                self.tokenizer.batch_decode(group, skip_special_tokens=True)
                for group in grouped
            ][0]
        
        return generated_texts
    
    def generate_with_scores(self,
        messages,
        stopping_criteria=None,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        return_logprobs=True,
        return_text=True
    ):
        processed_messages = self._process_messages(messages)
        if self.tokenizer.chat_template:
            input_prompt = self.tokenizer.apply_chat_template(
                processed_messages,
                add_generation_prompt=True,
                tokenize=False
            )

        with torch.no_grad():
            inputs = self.tokenizer(input_prompt, return_tensors="pt").to(self.device)
            input_ids = inputs['input_ids']
            model_output = self.generator.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stopping_criteria,
                do_sample=do_sample,
                temperature=temperature,
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
                tokens_text = [[self.tokenizer.decode(token) for token in tokens_] for tokens_ in tokens]
                generated_texts = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
            
            if return_logprobs:
                logits_list = torch.stack(model_output.logits).cpu().permute(1, 0, 2)
                model_output.logits = None
                logprobs = torch.log_softmax(logits_list, dim=-1) #logprobs for each token
                logprobs = torch.gather(logprobs, dim=-1, index = model_output.sequences[:, len(input_ids[0]):].unsqueeze(-1))#logprobs for each token in the generated text
                logprobs = logprobs.squeeze(-1).tolist()#convert to list
                logprobs = [logprobs[i][:indices[i]+1] for i in range(len(logprobs))]
                assert len(tokens_text[0]) == len(logprobs[0])
        
        return generated_texts[0], tokens_text[0], logprobs[0]    
            
        #     logits = torch.stack(model_output.scores, dim=1).softmax(-1).cpu()
        #     generated_ids = model_output.sequences[:, len(input_ids[0]):]
        #     gen_score = torch.gather(logits, 2, generated_ids[:, :, None]).squeeze(-1).cpu().tolist()
        # return generated_texts[0], gen_score[0]

    def generate_attn(self,
        messages,
        max_new_tokens,
        solver="max",
        use_entropy = False,
        use_logprob = False,
        add_generation_prompt=True,
        continue_final_message=False
    ):
        processed_messages = self._process_messages(messages)
        text = self.tokenizer.apply_chat_template(
            processed_messages,
            tokenize = False,
            add_generation_prompt=True,
            # continue_final_message=continue_final_message
        )
        
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt").to(self.generator.device)
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
            tokens = [self.tokenizer.convert_ids_to_tokens(tokens_) for tokens_ in token_ids][generation_id]
            text = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)[generation_id]
            
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

