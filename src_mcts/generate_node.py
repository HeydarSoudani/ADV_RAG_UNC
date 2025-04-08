
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import torch
from typing import List, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.general_utils import read_txt
from utils.adaptive_utils import fix_tokenizer_chat
from src_adaptive.templetes import SYSTEM_PROMPT_SHORTFORM
from src_mcts import examplers


class Generator:
    """Generator generates children nodes"""

    def __init__(self, args, retriever, evaluator) -> None:
        self.args = args
        self.retriever = retriever
        self.evaluator = evaluator
        
        # --- Define model ------------
        self.generation_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=False
        )
        self.eos_token_ids = self.generation_model.config.eos_token_id
        
        self.num_subquestions = args.num_subquestions
        self.num_votes = args.num_votes
        
        self.enable_potential_score = args.enable_potential_score
        self.mcts_num_last_votes = args.mcts_num_last_votes
        
        # Actions' prompts
        self.query_decomposition_prompt = read_txt(self.args.query_decomposition_prompt_file) # A3
        self.semantic_equivalence_prompt = read_txt(self.args.semantic_equivalence_prompt_file)
        
        try:
            self.fewshot_examplers = getattr(examplers, f'{args.dataset}_query_exps')
            self.fewshot_rephrased_examplers = getattr(examplers, 'rephrased_exps')
        except AttributeError:
            raise ValueError(f"The dataset '{args.dataset}' does not exist in the 'examplers' module.")
        
        # EoS tokens
        self.stop_tokens_direct_answer = ['\n</Answer>']
        self.stop_tokens_query_decomposition = ['\n4.', '\n</Subquestions>']
    
    def get_prompt_text(self, cur_node_type, solution_trace: Dict[int, Dict[str, str]], docs, prev_subqs=None, cur_subq=None):
        input_text = ''
        user_quesry = solution_trace[0]['user_question']
        
        # Generate Answer or Subanswer
        if cur_node_type in ['direct_answer', 'rag_answer']:
            sub_docs = []
            # Path
            for item_idx in solution_trace:
                solution_item = solution_trace[item_idx]
                node_keys = list(solution_item.keys())
                node_type = node_keys[0]
                if node_type == 'user_question':
                    user_quesry = solution_item[node_type]
                elif node_type == 'rephrased_query':
                    input_text += f'Given the question: {user_quesry}\n'
                    input_text += f'We rephrase the question, which can also be expressed as: \n{solution_item[node_type]}\n'
                elif node_type == 'subquestions':
                    input_text += f'We then decompose the question into several sub-questions with their corresponding answers:\n'
                elif node_type == 'subq_direct_answer':
                    input_text += f"{solution_item[node_type]['subquestion']} {solution_item[node_type]['subanswer']}\n"
                elif node_type == 'subq_rag_answer':
                    input_text += f"{solution_item[node_type]['subquestion']} {solution_item[node_type]['subanswer']}\n"
                    sub_docs.extend(solution_item[node_type]['documents'])
                elif node_type == 'rag_answer':
                    sub_docs.extend(solution_item[node_type]['documents'])
                
            # Docs
            if len(docs) > 0:
                input_text += "Below are some relevant documents that may help answer the question:\n"
                for i, doc in enumerate(docs):
                    input_text += f"[{i+1}] {doc}\n"
                input_text += "\n"
            if len(sub_docs) > 0:
                for i, doc in enumerate(sub_docs):
                    input_text += f"[{len(docs)+i+1}] {doc}\n"
                input_text += "\n"
                
            # Few-shot examples
            if len(self.fewshot_examplers) > 0:
                input_text += "Here are several examples of how to answer similar questions:\n\n"
                for exp in self.fewshot_examplers:
                    input_text += f"Question: {exp['question']}\n"
                    input_text += f"Answer: {exp['answer']}\n"
                input_text += "\n"
            
            
            input_text += "Now, answer the following question EXACTLY in the format of the examples above.\n"
            input_text += "DO NOT add any introductory phrases, explanations, or extra text.\n\n"
            input_text += f"Question: {user_quesry}\nAnswer:"
        
        elif cur_node_type == 'subquestions':
            input_text += "Given an input question, decompose it into multiple smaller and indivisible sub-questions.\n"
            input_text += "Here are several examples of how to output the sub-questions:\n\n"
            for exp in self.fewshot_examplers:
                input_text += f"Original Question: {exp['question']}\n"
                input_text += "Subquestions:\n"
                for i, subq in enumerate(exp['subqa']):
                    input_text += f"{i+1}.{subq[0]}\n"
                input_text += "\n"
            
            input_text += "Now, generate sub-questions for the following question EXACTLY in the format of the examples above.\n"
            input_text += "DO NOT add any introductory phrases, explanations, or extra text.\n\n"
            input_text += f'Original Question: {user_quesry}\n'
            input_text += "Subquestions:\n"
        
        elif cur_node_type == 'rephrased_query':
            input_text += "Given an input question, rephrase it into a more intuitive and easier-to-understand version.\n"
            input_text += "Here are several examples of how to output the rephrased question:\n\n"
            for exp in self.fewshot_rephrased_examplers:
                input_text += f"Original Question: {exp['question']}\n"
                input_text += f"Rephrased Question: {exp['Rephrased']}\n\n"
            
            input_text += "Now, generate the rephrased version for the following question EXACTLY in the format of the examples above.\n"
            input_text += "DO NOT add any introductory phrases, explanations, or extra text.\n\n"
            input_text += f'Original Question: {user_quesry}\n'
            input_text += f'Rephrased Question: '
        
        elif cur_node_type in ['subq_direct_answer', 'subq_rag_answer']:
            sub_docs = []
            # Path
            for item_idx in solution_trace:
                solution_item = solution_trace[item_idx]
                node_keys = list(solution_item.keys())
                node_type = node_keys[0]
                if node_type == 'user_question':
                    input_text += f'Given the question: {solution_item[node_type]}\n'
                elif node_type == 'rephrased_query':
                    input_text += f'We rephrase the question, which can also be expressed as: \n{solution_item[node_type]}\n\n'

            if len(prev_subqs) > 0:
                input_text += f'We then decompose the question into several sub-questions, some of them are answered:\n'
                for psub in prev_subqs:
                    input_text += f"Sub-question: {psub[0]} sub-answer: {psub[1]}\n"
            else:
                input_text += f'We then decompose the question into several sub-questions, we want to answer one by one.\n'

            # Docs
            if len(docs) > 0:
                input_text += "Below are some relevant documents that may help answer the question:\n"
                for i, doc in enumerate(docs):
                    input_text += f"[{i+1}] {doc}\n"
                input_text += "\n"
        
            # Few-shot examples
            if len(self.fewshot_examplers) > 0:
                input_text += "Here are several examples of how to answer similar questions:\n\n"
                for exp in self.fewshot_examplers:
                    input_text += f"Question: {exp['question']}\n"
                    input_text += f"Answer: {exp['answer']}\n"
                input_text += "\n"
        
            input_text += "Summarizing the information above, now answer the following question EXACTLY in the format of the examples above.\n"
            input_text += "DO NOT add any introductory phrases, explanations, or extra text.\n\n"
            input_text += f"Question: {cur_subq}\nAnswer:"
        
        return input_text
        
    def generate(self, input_text, max_new_tokens, sys_prompt='', num_return:int = 1):
        messages = [{'role': 'system', 'content': sys_prompt}]
        messages.append({'role': 'user', 'content': input_text})
        tokenizer, messages = fix_tokenizer_chat(self.tokenizer, messages)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt=True,
            continue_final_message=False
        )
        generated_texts = []
        for i in range(num_return):
            with torch.no_grad():
                inputs = tokenizer(text, return_tensors="pt").to(self.generation_model.device)
                input_ids = inputs['input_ids']
                model_output = self.generation_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_logits=True,
                    output_scores = True,
                    eos_token_id=self.eos_token_ids,
                    pad_token_id=tokenizer.eos_token_id,
                    return_legacy_cache=True
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
                
                tokens = [seq[len(input_ids[0]):indices[i] + len(input_ids[0])+1].tolist() for i, seq in enumerate(model_output.sequences)]
                tokens_text = [[tokenizer.decode(token) for token in tokens_] for tokens_ in tokens]
                generated_text = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
                generated_texts.append(generated_text)
                
        return generated_texts

    def _get_most_likely_answer(self, user_query: str, output_list: List[str]):
        assert len(output_list) > 0

        def cluster_by_meaning(user_query, output_list):
            cluster = []

            for i, answer in enumerate(output_list):
                if i == 0:
                    cluster.append([answer])
                else:
                    prompt = self.semantic_equivalence_prompt
                    prompt += f'\n\nWe are evaluating answers to the question: {user_query}\n'
                    prompt += 'Here are two possible answers:\n'

                    for j, c in enumerate(cluster):
                        tmp_prompt = prompt + f'Possible Answer 1: {answer}\n'
                        tmp_prompt += f'Possible Answer 2: {c[0]}\n'
                        tmp_prompt += 'For this question, is Possible Answer 1 semantically equivalent to Possible Answer 2? Respond with Yes or No.\n'
                        tmp_prompt += 'Response: '
                        
                        response = self.generate(
                            tmp_prompt,
                            max_new_tokens=1,
                            num_return=1,
                            # temperature=0.01,
                        )[0]
                        if 'Yes' in response:
                            c.append(answer)
                            break
                        elif j == len(cluster) - 1:
                            cluster.append([answer])
                            break

            return cluster

        if len(output_list) == 1:
            most_confident_answer = output_list[0]
            confidence = 1
        else:
            cluster = cluster_by_meaning(user_query=user_query, output_list=output_list)
            most_confident_cluster = sorted(cluster, key=len, reverse=True)[0]
            most_confident_answer, confidence = most_confident_cluster[0], len(most_confident_cluster)/sum(map(len, cluster))
            assert confidence > 0 and confidence <= 1

        return most_confident_answer, confidence

    def generate_rephrased_question(self, solution_trace: Dict[int, Dict[str, str]]):
        input_prompt_text = self.get_prompt_text('rephrased_query', solution_trace, [], [])
        output = self.generate(
            input_prompt_text,
            max_new_tokens=128,
            num_return=1,
        )[0]
        return output
    
    def generate_direct_answer(self, solution_trace: Dict[int, Dict[str, str]]):
        subs = []
        user_question = solution_trace[0]['user_question']
        
        for _, cur_node in solution_trace.items():
            node_key = list(cur_node.keys())[0]
            if node_key in ['subq_direct_answer', 'subq_rag_answer']:
                subquestion = cur_node[node_key]['subquestion']
                subanswer = cur_node[node_key]['subanswer']
                subs.append((subquestion, subanswer))
            
            if node_key is "rephrased_query":
                user_question = cur_node[node_key]
        
        # = Do generation
        input_prompt_text = self.get_prompt_text('direct_answer', solution_trace, [], subs)
        output_list = self.generate(
            input_prompt_text,
            max_new_tokens=16,
            num_return=self.mcts_num_last_votes,
            sys_prompt=SYSTEM_PROMPT_SHORTFORM
        )
        answer, value = self._get_most_likely_answer(user_query=user_question, output_list=output_list)
        return answer, value

    def generate_rag_answer(self, solution_trace: Dict[int, Dict[str, str]]):
        # = Get query
        qid = solution_trace[0]['qid']
        user_question = solution_trace[0]['user_question']
        for _, cur_node in solution_trace.items():
            node_key = list(cur_node.keys())[0]
            if node_key is "rephrased_query":
                user_question = cur_node[node_key]

        # = Do retrieval
        docs, _, _ = self.retriever.retrieve([user_question], [qid], [], [])
        
        # = Do generation
        input_prompt_text = self.get_prompt_text('rag_answer', solution_trace, docs[0], [])
        output_list = self.generate(
            input_prompt_text,
            max_new_tokens=16,
            num_return=self.mcts_num_last_votes,
            sys_prompt=SYSTEM_PROMPT_SHORTFORM
        )
        answer, value = self._get_most_likely_answer(user_query=user_question, output_list=output_list)
        return docs, answer, value
        
    def generate_query_decomposition(self, solution_trace: Dict[int, Dict[str, str]]):
        input_prompt_text = self.get_prompt_text('subquestions', solution_trace, [], [])
        output = self.generate(
            input_prompt_text,
            max_new_tokens=128,
            num_return=1,
        )[0]
        subquestions = re.findall(r'\d+\.(.*)', output)
        subquestions = [s.strip() for s in subquestions]
        return subquestions
       
    def generate_subq_direct_answer(self, solution_trace: Dict[int, Dict[str, str]]):
        # Get subquestion
        last_node = solution_trace[list(solution_trace.keys())[-1]]
        node_type = list(last_node.keys())[0]
        subs = []
        
        if node_type == "subquestions":
            subquestion = last_node[node_type][0]
            subquestion_pointer = 1
            len_subqs = len(last_node["subquestions"])
        
        elif node_type in ["subq_direct_answer", "subq_rag_answer"]:
            cur_pointer = last_node[node_type]["subq_pointer"]
            subquestion_pointer = cur_pointer + 1
            
            for id, node in solution_trace.items():
                node_key = list(node.keys())[0]
                if node_key == "subquestions":
                    len_subqs = len(node["subquestions"])
                    subquestion = node["subquestions"][cur_pointer]
                    
                if node_key in ['subq_direct_answer', 'subq_rag_answer']:
                    subq = node[node_key]['subquestion']
                    suba = node[node_key]['subanswer']
                    subd = node[node_key]['documents'] if node_key=='subq_rag_answer' else []
                    subs.append((subq, suba, subd))    
            
        # Do generation
        input_prompt_text = self.get_prompt_text('subq_direct_answer', solution_trace, [], subs, subquestion)
        output_list = self.generate(
            input_prompt_text,
            max_new_tokens=32,
            num_return=self.mcts_num_last_votes,
        )
        subanswer, _ = self._get_most_likely_answer(user_query=subquestion, output_list=output_list)
        return subquestion, subanswer, subquestion_pointer, len_subqs
        
    def generate_subq_rag_answer(self, solution_trace: Dict[int, Dict[str, str]]):
        # Get subquestion
        qid = solution_trace[0]['qid']
        last_node = solution_trace[list(solution_trace.keys())[-1]]
        node_type = list(last_node.keys())[0]
        subs = []
        
        if node_type == "subquestions":
            subquestion = last_node[node_type][0]
            subquestion_pointer = 1
            len_subqs = len(last_node["subquestions"])
        
        elif node_type in ["subq_direct_answer", "subq_rag_answer"]:
            cur_pointer = last_node[node_type]["subq_pointer"]
            subquestion_pointer = cur_pointer + 1
            
            for id, node in solution_trace.items():
                node_key = list(node.keys())[0]
                if node_key == "subquestions":
                    len_subqs = len(node["subquestions"])
                    subquestion = node["subquestions"][cur_pointer]
                    
                if node_key in ['subq_direct_answer', 'subq_rag_answer']:
                    subq = node[node_key]['subquestion']
                    suba = node[node_key]['subanswer']
                    subd = node[node_key]['documents'] if node_key=='subq_rag_answer' else []
                    subs.append((subq, suba, subd))

        # Do retrieval
        ret_query = ', '.join([f"{sub[0]} {sub[1]}" for sub in subs])
        ret_query += f", {subquestion}"
        docs, _, _ = self.retriever.retrieve([ret_query], [qid], [], [])
        
        # Do generation
        input_prompt_text = self.get_prompt_text('subq_rag_answer', solution_trace, docs[0], subs, subquestion)
        output_list = self.generate(
            input_prompt_text,
            max_new_tokens=32,
            num_return=self.mcts_num_last_votes,
        )
        subanswer, _ = self._get_most_likely_answer(user_query=subquestion, output_list=output_list)
        
        return docs, subquestion, subanswer, subquestion_pointer, len_subqs