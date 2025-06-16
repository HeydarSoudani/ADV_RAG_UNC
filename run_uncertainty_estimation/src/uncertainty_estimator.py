import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
from transformers import DebertaForSequenceClassification, DebertaTokenizer

from utils.general_utils import find_token_indices
from run_uncertainty_estimation.ue_methods import *

class UncertaintyEstimator:
    def __init__(self, model, tokenizer, args):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        model_for_entailment = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to(entailment_model_device)
        tokenizer_for_entailment = DebertaTokenizer.from_pretrained("microsoft/deberta-large-mnli")

        self.ue_methods_ = {
            "p_true": PTrue(self.model, self.tokenizer),
            "num_ss": NumSemanticSet(model_for_entailment, tokenizer_for_entailment),
            "sum_eigen": SumEigenUncertainty(model_for_entailment, tokenizer_for_entailment),
            "eccentricity": EccentricityUncertainty(model_for_entailment, tokenizer_for_entailment),
            "matrix_degree": MatrixDegreeUncertainty(model_for_entailment, tokenizer_for_entailment),
            # "confidence": Confidence(),
            # "entropy": Entropy(),
            # "PE": PredictiveEntropy(),
            # "SE": SemanticEntropy(),
        }
    
    def estimate(self,
        question, prediction, context:str,
        input_prompt_texts, generated_output_texts,
        generation_type:str = "existing_generations",
    ):
        
        if "entropy" in list(self.ue_methods_.keys()):
            # = Generation
            if generation_type == "new_generations":
                sampled_gen_dict = self.sample_generations_batch_hf_local(question, input_prompt_texts)
            elif generation_type == "existing_generations": 
                sampled_gen_dict = self.dict_generations_batch_hf_local(question, input_prompt_texts, generated_output_texts)
            else:
                raise NotImplementedError("Generation type is not defined!")
        
        else:
            sampled_gen_dict = {
                "question": question,
                "generated_texts": generated_output_texts
            }

    
        # = Uncertainty Estimation
        ue_scores = {}
        for ue_title, ue_function in self.ue_methods_.items():
            ue_scores[ue_title] = ue_function(sampled_gen_dict, prediction, context)
    
        return ue_scores
    
    
    def sample_generations_batch_hf_local(self, context, question):
        
        eos_token_id = self.model.config.eos_token_id
        if type(eos_token_id) == list:
            pad_token_id = eos_token_id[0]
        else:
            pad_token_id = eos_token_id
        
        # Input preparation
        input_text = self.get_prompt_text(context, question)
        if self.tokenizer.chat_template:
            input_prompt_text = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": input_text},
                ],
                add_generation_prompt=True,
                tokenize=False
            )
        inputs = self.tokenizer(input_prompt_text, return_tensors='pt').to(self.model.device)
        input_ids = inputs["input_ids"]
    
        with torch.no_grad():
            model_output = self.model.generate(
                **inputs,
                num_return_sequences=self.number_of_generations,
                return_dict_in_generate=True,
                output_logits=True,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=True,
                temperature=1.0
            )
            model_output.past_key_values = None
            model_output.sequences = model_output.sequences.cpu()
            
            if type(eos_token_id) == list:
                temp = torch.stack([torch.argmax((model_output.sequences[:, len(input_ids[0]):] == eos).to(dtype=torch.int), dim=-1,) for eos in eos_token_id]).T
                for i in range(len(temp)):
                    if_eos = False
                    for j in range(len(temp[i])):
                        if temp[i][j] != 0:
                            if_eos = True
                            break
                    if if_eos == False:#if it doesn't contain eos token
                        temp[i][-1] = model_output.sequences.shape[1] - len(input_ids[0])  - 1
                indices = [torch.min(temp[i][temp[i] > 0]).item() for i in range(len(temp))]
            else:
                indices = torch.argmax((model_output.sequences[:, len(input_ids[0]):] == eos_token_id).to(dtype=torch.int), dim=-1,)
            indices[indices == 0] = model_output.sequences.shape[1] - len(input_ids[0]) - 1
            
            tokens = [seq[len(input_ids[0]): indices[i] + len(input_ids[0])].tolist() for i, seq in enumerate(model_output.sequences)]
            tokens_text = [[self.tokenizer.decode(token) for token in tokens_] for tokens_ in tokens]
            generated_texts = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
            
            logits_list = torch.stack(model_output.logits).cpu().permute(1, 0, 2)
            model_output.logits = None
            
            logprobs = torch.log_softmax(logits_list, dim=-1)  # logprobs for each token
            logprobs = torch.gather(logprobs, dim=-1, index=model_output.sequences[:, len(input_ids[0]):].unsqueeze(-1),)  # logprobs for each token in the generated text
            logprobs = logprobs.squeeze(-1).tolist()  # convert to list
            logprobs = [logprobs[i][: indices[i]] for i in range(len(logprobs))]
            
            logits_list = [logits_list[i][: indices[i]] for i in range(len(logits_list))]
        
        return {
            "question": question,
            "generated_texts": generated_texts,
            "tokens": tokens,
            "tokens_text": tokens_text,
            "logits": logits_list,
            "logprobs": logprobs,
        }
    
    def dict_generations_batch_hf_local(self, question, input_prompt_texts, generated_output_texts):
        logprobs_list, logits_list, tokens_list, tokens_text_list = [], [], [], []
        
    
        for idx, generated_output_text in enumerate(generated_output_texts):
            if self.tokenizer.chat_template:
                input_prompt_text = self.tokenizer.apply_chat_template(
                    [
                        # {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": input_prompt_texts[idx]},
                        {"role": "assistant", "content": generated_output_text}
                    ],
                    add_generation_prompt=True,
                    tokenize=False
                )
        
            # Generation
            generated_text_ids = self.tokenizer.encode(generated_output_text, add_special_tokens=False)
            tokens_list.append(generated_text_ids)
            tokens_text_list.append([self.tokenizer.decode(answer_id) for answer_id in generated_text_ids])
            input_prompt_tokens = self.tokenizer.encode(input_prompt_text, return_tensors="pt").to(self.model.device)
            indices, texts = find_token_indices(input_prompt_tokens[0], self.tokenizer, generated_output_text)
            
            with torch.no_grad():
                outputs = self.model(input_prompt_tokens)
                logits = outputs.logits

            logits_list.append(logits[0, indices[-1][0]-1:indices[-1][-1], :])
            
            logprobs = torch.log_softmax(logits, dim=-1)
            logprobs = logprobs[0, :-1, :]
            logprobs = torch.gather(logprobs, dim=1, index=input_prompt_tokens[0][1:].view(-1, 1)) # (len(input)-1, 1)
            logprobs = logprobs.view(-1).tolist()
            logprobs = [logprobs[index-1] for index in indices[-1]]
            logprobs_list.append(logprobs)
        
        return {
            "question": question,
            "generated_texts": generated_output_texts,
            "tokens": tokens_list,
            "tokens_text": tokens_text_list,
            "logits": logits_list,
            "logprobs": logprobs_list,
        }
