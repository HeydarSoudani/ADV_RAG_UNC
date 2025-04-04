
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import torch
from typing import List, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.general_utils import read_txt
from utils.adaptive_utils import fix_tokenizer_chat


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
        
        # EoS tokens
        self.stop_tokens_direct_answer = ['\n</Answer>']
        self.stop_tokens_query_decomposition = ['\n4.', '\n</Subquestions>']
        
    def generate(self, input_text, max_new_tokens, num_return:int = 1):
        messages = [{'role': 'system', 'content': ''}]
        messages.append({'role': 'user', 'content': input_text})
        tokenizer, messages = fix_tokenizer_chat(self.tokenizer, messages)
        text = tokenizer.apply_chat_template(messages, tokenize = False)
        
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

    def generate_direct_answers(self, solution_trace: Dict[int, Dict[str, str]]):
        # Following RASPberry ....
        input_text = ''
        node_type_trace = []
        retrieve_count = 0
        
        for _, parent_state in solution_trace.items():
            node_type = list(parent_state.keys())[0]
            node_type_trace.append(node_type)
            parent_gen = parent_state[node_type]
        
            if node_type == 'document':
                retrieve_count += 1

            if node_type == 'user_question':
                user_query = parent_gen
                input_text += f'Given the question: \n{parent_gen}\n'
            elif node_type == 'subquery':
                input_text += f'We then decompose the question into several sub-questions, namely: \n{parent_gen}\n'

            elif node_type == 'document':
                if retrieve_count == 1:
                    input_text += f'For this type of question, we retrieve a series of relevant documents, referred to as: \n{parent_gen}\n'
                else:
                    node_type_trace = node_type_trace[:-1]
                    node_type_trace.append(f'document-{retrieve_count}')
                    input_text += f'After further retrieving more relevant documents, we obtain: \n{parent_gen}\n'
            
        assert 'answer' not in node_type_trace
        node_type_trace.append('answer')
        
        input_text += 'Summarizing the information above, now we extract the answer, the answer is: ' #\n<Answer>\n

        # Get final answer with confidence
        output_list = self.generate(
            input_text,
            max_new_tokens=32,
            # eos_tokens=['\n</Answer>'],
            num_return=self.mcts_num_last_votes,
        )

        answer, value = self._get_most_likely_answer(user_query=user_query, output_list=output_list)
        return answer, value

    def generate_retrieve_docs(self, solution_trace: Dict[int, Dict[str, str]]):
        query = ''
        for _, parent_state in solution_trace.items():
            node_type = list(parent_state.keys())[0]
            parent_gen = parent_state[node_type]

            if node_type == 'user_question':
                qid = parent_state['qid']
                query += f'Given the question: \n{parent_gen}\n'
            elif node_type == 'subquery':
                query += f'We then decompose the question into several sub-questions, namely: \n{parent_gen}\n'

        docs, _, _ = self.retriever.retrieve([query], [qid], [], [])
        
        document = ''
        for i, doc in enumerate(docs[0]):
            document += f'\n\n<doc {i+1}>\n{doc}\n</doc {i+1}>\n\n'
        document = f'<document>{document}</document>'

        return document
        
    def generate_query_decomposition(self, query):
        input_text = self.query_decomposition_prompt
        input_text += f'\n\n<Original Question>\n{query}\n</Original Question>\n\n'
        input_text += f'<Subquestions>\n'

        output = self.generate(
            input_text,
            max_new_tokens=128,
            num_return=1,
            # stop_tokens=['\n4.', '\n</Subquestions>'],
        )[0]
        
        match = re.search(r"<Subquestions>\n(.*?)\n</Subquestions>", output, re.DOTALL)
        if match:
            subquestions_text = match.group(1)
            subquestions = re.findall(r'\d+\.(.*)', subquestions_text)
            subquestions = [s.strip() for s in subquestions]
            return subquestions
        else:
            print("No subquestions found.")
            return []
        


        