
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import spacy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.common_utils import fix_tokenizer_chat
from src.templetes import DEFAULT_SYSTEM_PROMPT

nlp = spacy.load("en_core_web_sm")

class Counter:
    def __init__(self):
        self.retrieve = 0
        self.generate = 0
        self.hallucinated = 0
        self.token = 0
        self.sentence = 0

    def add_generate(self, text, tokenizer):
        self.generate += 1
        ids = tokenizer(text, return_tensors="pt")['input_ids'][0].tolist()
        self.token += len(ids)
        sentences = [sent.text for sent in nlp(text).sents]
        self.sentence += len(sentences)

    def calc(self, other_counter):
        return {
            "retrieve_count": self.retrieve - other_counter.retrieve, 
            "generate_count": self.generate - other_counter.generate,
            "hallucinated_count": self.hallucinated - other_counter.hallucinated, 
            "token_count": self.token - other_counter.token, 
            "sentence_count": self.sentence - other_counter.sentence 
        }

class BasicGenerator:
    def __init__(self, args):
        self.args = args
        self.generator = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16, device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False) 
        self.eos_token_id = self.generator.config.eos_token_id
        
        
    def generate(self, input_text, return_text=True, return_logprobs=True, add_generation_prompt=True, continue_final_message=False):
        messages = [{'role': 'system', 'content': DEFAULT_SYSTEM_PROMPT}]
        messages.append({'role': 'user', 'content': input_text})
        tokenizer, messages = fix_tokenizer_chat(self.tokenizer, messages)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message
        )
        print(f"{text}\n")
        
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt").to(self.generator.device)
            input_ids = inputs['input_ids']
            model_output = self.generator.generate(
                **inputs,
                max_new_tokens=self.args.generate_max_length,
                return_dict_in_generate=True,
                output_logits=True,
                eos_token_id=self.eos_token_id
            )
            # tokens = model_output[0][len(input_ids[0]):]
            # generated_text = self.tokenizer.decode(tokens, skip_special_tokens = False)
            # generated_text_return = self.tokenizer.decode(tokens, skip_special_tokens = True)

            model_output.past_key_values=None
            model_output.sequences = model_output.sequences.cpu()
            if type(self.eos_token_id) == list:
                temp = torch.stack([torch.argmax((model_output.sequences[:, len(input_ids[0]):] == eos).to(dtype=torch.int), dim=-1) for eos in self.eos_token_id]).T
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
                indices = torch.argmax((model_output.sequences[:, len(input_ids[0]):] == self.eos_token_id).to(dtype=torch.int), dim=-1)
            indices[indices==0] = model_output.sequences.shape[1] - len(input_ids[0]) -1
            
            if return_text:
                tokens = [seq[len(input_ids[0]):indices[i] + len(input_ids[0])+1].tolist() for i, seq in enumerate(model_output.sequences)]
                generated_texts = tokenizer.batch_decode(tokens, skip_special_tokens=True)
            
            if return_logprobs:
                logits_list = torch.stack(model_output.logits).cpu().permute(1, 0, 2)
                model_output.logits = None
                logprobs = torch.log_softmax(logits_list, dim=-1) #logprobs for each token
                logprobs = torch.gather(logprobs, dim=-1, index = model_output.sequences[:, len(input_ids[0]):].unsqueeze(-1))#logprobs for each token in the generated text
                logprobs = logprobs.squeeze(-1).tolist()#convert to list
                logprobs = [logprobs[i][:indices[i]+1] for i in range(len(logprobs))]
        
        # print(tokens[0])
        # print(logprobs[0])  
        # print(generated_texts[0])
          
        assert len(tokens[0]) == len(logprobs[0])
        return generated_texts[0], tokens[0], logprobs[0]
    
    
    def generate_attn(self, input_text, max_length, solver="max", use_entropy = False, use_logprob = False):
        pass

    
    # TODO: ...
    def regenerate_answer(self):
        pass
    


class BasicRAG:
    def __init__(self, args):
        # args = args.__dict__ 
        self.args = args
        self.generator = BasicGenerator(args)
        self.retriever = None
        self.counter = Counter()
        # prompt templates
        self.cot_answer_template = lambda cot, ans: f'{cot} So the answer is {ans}.'
        self.query_long_version_template = lambda ques: f'Answer the following question by reasoning step-by-step, following the example above.\nQuestion: {ques}\nAnswer:' 
        self.query_short_version_template = lambda ques: f'Question: {ques}\nAnswer:'

    def retrieve(self, query, topk=1, max_query_length=64):
        self.counter.retrieve += 1
        docs = self.retriever.retrieve(queries=[query], topk=topk)
        return docs[0]
        
    def get_top_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[0] if len(sentences) > 0 else ""

    def get_last_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[-1] if len(sentences) > 0 else "" 
   
    def format(self, question, fewshot_examplers, docs, with_rag=False):
        prompt = ""
        for exp in fewshot_examplers:
            prompt += f"Question: {exp['question']}\n"
            prompt += f"Answer: {exp['cot']} So, the answer is {exp['answer']}.\n"
        prompt += "\n"
        
        if with_rag:
            prompt += "Context:"
            for i, doc in enumerate(docs):
                prompt += f"[{i+1}] {doc}\n"
            prompt += "\n"
        
        prompt += "Answer in the same format as before.\n" 
        prompt += f"Question: {question}\nAnswer:"
        return prompt
    
    
class NoRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def inference(self, question, fewshot_examplers):
        prompt = self.format(question, fewshot_examplers, [], with_rag=False)
        text, _, _ = self.generator.generate(prompt, self.args.generate_max_length)
        # if self.args.use_counter == True:
        #     self.counter.add_generate(text, self.generator.tokenizer)
        return text
    

class SingleRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def inference(self, question, fewshot_examplers):
        docs = self.retrieve(question, topk=self.retrieve_topk)
        prompt = self.rag_format(question, fewshot_examplers, docs, with_rag=True)
        text, _, _ = self.generator.generate(prompt, self.generate_max_length)
        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text


class FixLengthRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def inference(self, question, demo, case):
        pass
