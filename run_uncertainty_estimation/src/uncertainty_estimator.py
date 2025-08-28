import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
from transformers import DebertaForSequenceClassification, DebertaTokenizer

from utils.general_utils import find_token_indices
from run_uncertainty_estimation.ue_methods import *
from run_mcts_two_actions.src.models.semantic_equivalence import SemanticEquivalenceGenerator

def check_lengths(qid, tokens_list, tokens_text_list, logits_list, logprobs_list):
    for i, (tokens, tokens_text, logits, logprobs) in enumerate(
        zip(tokens_list, tokens_text_list, logits_list, logprobs_list)
    ):
        lengths = [len(tokens), len(tokens_text), len(logits), len(logprobs)]
        if len(set(lengths)) != 1:
            print(f"for {qid}, there is a mismatch at index {i}: {lengths}")

class UncertaintyEstimator:
    def __init__(self, model, tokenizer, device, args, generated_output_template=None):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        entailment_model_device = device
        model_for_entailment = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to(entailment_model_device)
        tokenizer_for_entailment = DebertaTokenizer.from_pretrained("microsoft/deberta-large-mnli")
        self.generated_output_template = generated_output_template
        self.se_model = SemanticEquivalenceGenerator(args, device, self.model, self.tokenizer)
        
        if args.consistency_method == 'self_consistency':
            self.ue_methods_ = {
                "predictive_entropy": PredictiveEntropy(),
                "semantic_entropy": SemanticEntropy(model_for_entailment, tokenizer_for_entailment),
                'mars': MARS(),
                'lars': LARS(
                    ue_type="semantic_entropy",
                    model_for_entailment=model_for_entailment,
                    tokenizer_for_entailment=tokenizer_for_entailment,
                    entailment_model_device=entailment_model_device
                ),
                "sar": SAR(self.tokenizer),
                "num_ss": NumSemanticSet(model_for_entailment, tokenizer_for_entailment),
                "sum_eigen": SumEigenUncertainty(model_for_entailment, tokenizer_for_entailment),
                "eccentricity": EccentricityUncertainty(model_for_entailment, tokenizer_for_entailment),
                "matrix_degree": MatrixDegreeUncertainty(model_for_entailment, tokenizer_for_entailment),
                "p_true": PTrue(self.model, self.tokenizer),
                "majority_voting": MajorityVoting(self.se_model),
            }
        else:
            self.ue_methods_ = {
                "majority_voting": MajorityVoting(self.se_model),
            }
            
        self.white_box_ue_methods = ['predictive_entropy', 'semantic_entropy', 'mars', 'lars', 'sar']
        self.wanted_ue_methods = list(self.ue_methods_.keys())
    
    def estimate(self,
        qid, question, prediction, context:str,
        input_prompt_texts,
        generated_output_texts,
        generation_type:str = "existing_generations",
    ):
        
        if any(item in self.white_box_ue_methods for item in self.wanted_ue_methods):
            # = Generation
            if generation_type == "new_generations":
                sampled_gen_dict = self.sample_generations_batch_hf_local(qid, question, input_prompt_texts)
            elif generation_type == "existing_generations": 
                sampled_gen_dict = self.dict_generations_batch_hf_local(qid, question, input_prompt_texts, generated_output_texts)
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
    
    def dict_generations_batch_hf_local(self, qid, question, input_prompt_texts, generated_output_texts):
        generated_output_texts_, logprobs_list, logits_list, tokens_list, tokens_text_list = [], [], [], [], []
        
        for idx, generated_output_text in enumerate(generated_output_texts):
            if not generated_output_text:
                continue
            try:
                if self.tokenizer.chat_template:
                    # If you sometimes wrap the model output, keep that here
                    generated_output_text_ = (
                        self.generated_output_template.format(answer=generated_output_text)
                        if getattr(self, "generated_output_template", None)
                        else generated_output_text
                    )

                    # Build only the prompt up to assistant "generation start"
                    prompt_text = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": input_prompt_texts[idx]}],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
                else:
                    # Fallback: plain prompt (no chat template)
                    prompt_text = input_prompt_texts[idx]
                    # Optionally add a delimiter that your model expects before assistant text
                    prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True, return_tensors="pt")

                prompt_ids = prompt_ids.to(self.model.device)

                # 2) Encode the known generated text
                gen_ids = self.tokenizer.encode(generated_output_text, add_special_tokens=False)
                if len(gen_ids) == 0:
                    # Nothing generated; keep lists aligned by skipping all
                    continue
                gen_ids_tensor = torch.tensor(gen_ids, device=self.model.device).unsqueeze(0)  # shape (1, L_gen)

                # 3) Concatenate: [prompt || generated]
                input_ids_full = torch.cat([prompt_ids, gen_ids_tensor], dim=1)  # shape (1, L_prompt+L_gen)

                # 4) Forward pass
                with torch.no_grad():
                    outputs = self.model(input_ids_full)
                    logits = outputs.logits  # (1, L_total, V)

                # 5) Extract logits for the generated tokens only.
                #    Each generated token at position t is predicted by logits at position t-1.
                L_prompt = prompt_ids.size(1)
                L_gen = gen_ids_tensor.size(1)

                gen_logits = logits[0, L_prompt-1 : L_prompt-1 + L_gen, :]        # shape (L_gen, V)

                # 6) Logprobs for the actual generated tokens
                logprobs_all = torch.log_softmax(gen_logits, dim=-1)              # (L_gen, V)
                gen_token_ids = gen_ids_tensor[0]                                  # (L_gen,)
                gen_logprobs = logprobs_all.gather(1, gen_token_ids.view(-1,1))    # (L_gen, 1)
                gen_logprobs = gen_logprobs.view(-1).tolist()

                # 7) Append ONLY after all computations succeed
                generated_output_texts_.append(generated_output_text)
                tokens_list.append(gen_ids)                                        # list[int]
                tokens_text_list.append([self.tokenizer.decode([tid]) for tid in gen_ids])
                logits_list.append(gen_logits)                                     # torch.Tensor (L_gen, V)
                logprobs_list.append(gen_logprobs)                                 # list[float]

            except Exception as e:
                # If one sample fails, skip it consistently (or record None placeholders if you prefer strict alignment)
                # Example with placeholders to keep exact alignment across lists:
                # generated_output_texts_.append(generated_output_text)
                # tokens_list.append([])
                # tokens_text_list.append([])
                # logits_list.append(None)
                # logprobs_list.append([])
                # For most pipelines, skipping is cleaner:
                continue
        
        # print(tokens_list)
        # print(tokens_text_list)
        # print(logprobs_list)
        # print('----')
        check_lengths(qid, tokens_list, tokens_text_list, logits_list, logprobs_list)
        return {
            "question": question,
            "generated_texts": generated_output_texts_,
            "tokens": tokens_list,
            "tokens_text": tokens_text_list,
            "logits": logits_list,
            "logprobs": logprobs_list,
        }






# for idx, generated_output_text in enumerate(generated_output_texts):
#     if generated_output_text:
#         generated_output_texts_.append(generated_output_text)
#         if self.tokenizer.chat_template:
#             generated_output_text_ = self.generated_output_template.format(answer=generated_output_text) if self.generated_output_template else generated_output_text
#             input_prompt_text = self.tokenizer.apply_chat_template(
#                 [
#                     # {"role": "system", "content": self.system_prompt},
#                     {"role": "user", "content": input_prompt_texts[idx]},
#                     {"role": "assistant", "content": generated_output_text_}
#                 ],
#                 add_generation_prompt=True,
#                 tokenize=False
#             )
    
#         # Generation
#         generated_text_ids = self.tokenizer.encode(generated_output_text, add_special_tokens=False)
#         tokens_list.append(generated_text_ids)
#         tokens_text_list.append([self.tokenizer.decode(answer_id) for answer_id in generated_text_ids])
#         input_prompt_tokens = self.tokenizer.encode(input_prompt_text, return_tensors="pt").to(self.model.device)
#         indices, texts = find_token_indices(input_prompt_tokens[0], self.tokenizer, generated_output_text)
        
#         with torch.no_grad():
#             outputs = self.model(input_prompt_tokens)
#             logits = outputs.logits
#         logits_list.append(logits[0, indices[-1][0]-1:indices[-1][-1], :])
#         logprobs = torch.log_softmax(logits, dim=-1)
#         logprobs = logprobs[0, :-1, :]
#         logprobs = torch.gather(logprobs, dim=1, index=input_prompt_tokens[0][1:].view(-1, 1)) # (len(input)-1, 1)
#         logprobs = logprobs.view(-1).tolist()
#         logprobs = [logprobs[index-1] for index in indices[-1]]
#         logprobs_list.append(logprobs)  