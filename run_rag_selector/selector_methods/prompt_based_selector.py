import re
import ast
import json
import torch
import numpy as np
import transformers
from tqdm import tqdm

from run_rag_methods.src.correctness import em_score


class RAGSelector:
    def __init__(self, generation_model, generation_tokenizer, device, args):
        self.system_prompt = 'You are a helpful assistant.'
        self.generator = generation_model
        self.tokenizer = generation_tokenizer
        self.device = device
        self.args = args

    # def get_instruction(self, question, candidates):
    #     input_text = ""
    #     input_text += "You are an expert in the final answer selection task.\n"
    #     input_text += "Your task is to identify and select the best answer to the question from a list of candidates, each with a given confidence score.\n"
    #     input_text += "You must select the final answer from the candidates provided. Do not generate a new one.\n\n"            
    #     input_text += "Your output must include:\n"
    #     input_text += "- Exactly one reasoning step explaining your choice, wrapped in a single pair of <think> and </think> tags.\n"
    #     input_text += "- The selected final answer, wrapped in a single pair of <prediction> and </prediction> tags.\n\n"
        
    #     input_text += f"<question>{question}</question>\n"
    #     input_text += "<candidates>\n"
    #     for i, c in enumerate(candidates):
    #         input_text += f"[{i+1}] {c[0]}, with confidence {c[1]}\n"    
    #     input_text += "</candidates>\n"
        
    #     return input_text

    def get_instruction(self, question, candidates):
        input_text = f"Here is a question and some external data from {len(candidates)} systems information:\n"
        for i, c in enumerate(candidates):
            input_text += f"[{i+1}] {c[0]}, with confidence {c[1]}\n"
        
        input_text += f"\nQuestion: {question}\n\n"
        input_text += "Your task is to answer the question based on the given information."
        input_text += "You should first output your reasoning process and then provide the final answer." 
        input_text += "The output format of reasoning process and final answer should be enclosed within <think> </think> and <answer> </answer> tags, respectively,"
        input_text += "i.e., <think> reasoning process here </think> <answer> a final answer here </answer>\n"
        input_text += "Only output your reasoning process in <think></think> and your answer in <answer></answer>, and do not output any other words."
        
        return input_text

    def generate(self,
        messages,
        stopping_criteria=None,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
    ):
        if self.tokenizer.chat_template:
            input_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
        
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
        
        return output_, output_text
    
    def get_think(self, text):
        pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            return matches[0]
        else:
            return None
    
    def get_prediction(self, text):
        pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            return matches[0]
        else:
            return None
    
    def inference(self, question, candidates, generation_temp=0.7):
        input_prompt = self.get_instruction(question, candidates)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_prompt}
        ]
        _, output_text = self.generate(messages, temperature=generation_temp)
        think = self.get_think(output_text)
        prediction = self.get_prediction(output_text)
        
        return think, prediction


def get_prompt_based_selector(args, dataset):
    generation_model = transformers.AutoModelForCausalLM.from_pretrained(args.secondary_model_name_or_path, dtype=torch.bfloat16).to(args.device)
    generation_tokenizer = transformers.AutoTokenizer.from_pretrained(args.secondary_model_name_or_path)
    rag_selector = RAGSelector(generation_model, generation_tokenizer, args.device, args)
    ds = dataset['test']
    
    em_evaluation = []
    for idx, sample in enumerate(tqdm(ds)):
        if idx == 10:
            break
        qid, query, gt_answers = sample['qid'], sample['query'], sample['gt_answers']
        candidates_str = sample.get("candidates", None)
        candidates = ast.literal_eval(candidates_str)
        candidates_ = [(c[0], c[1]) for c in candidates]
        
        think, prediction = rag_selector.inference(query, candidates_)
        
        if len(gt_answers) > 0:
            correctness_em = em_score(prediction, gt_answers)
        else:
            correctness_em = 0
        em_evaluation.append(correctness_em)
    
    print(f"prompt accuracy: {np.mean(em_evaluation)*100}")
    