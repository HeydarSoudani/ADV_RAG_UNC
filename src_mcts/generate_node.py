
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import torch
import random
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
        
        # try:
        #     self.fewshot_examplers = getattr(examplers, f'{args.dataset}_query_exps')
        #     self.fewshot_rephrased_examplers = getattr(examplers, 'rephrased_exps')
        # except AttributeError:
        #     raise ValueError(f"The dataset '{args.dataset}' does not exist in the 'examplers' module.")
        
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
        
    def generate(self, 
        input_text,
        max_new_tokens,
        sys_prompt='You are a helpful assistant.',
        num_return:int = 1,
        temperature:float = 1.0,
        do_sample:bool = False,
    ):
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
                    return_legacy_cache=True,
                    temperature=temperature,
                    do_sample=do_sample
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
    
    
    # For V2
    def get_prompt_text_v2(self, curr_node, solution_trace: Dict[int, Dict[str, str]]):
        user_quesry = solution_trace[0]['user_question']
        
        # Intruction
        # input_text = "You are tasked with answering the following question in multiple steps. "
        # input_text += "At this point, you are responsible for executing one step of the process. "
        # input_text += "This could be the first, an intermediate, or the final step. "
        # if curr_node == 'think_search':
        #     input_text += 'You MUST produce output exclusively within the <think>...</think> and <search>...</search> tags. Do not include any content outside these tags or use any other tags.'
        #     input_text += ' To carry out one step, you MUST begin by reasoning inside <think> and </think>.'
        #     input_text += ' During reasoning, aim to identify one of the information that would help in answering the question.'
        #     input_text += ' Keep your reasoning concise and direct—no more than three to four sentences. Ensure that you close the reasoning properly with </think>.'
        #     input_text += ' After reasoning, you MUST call a search engine by <search> query </search>.'

        # elif curr_node == 'think_answer':
        #     input_text += 'You MUST produce output exclusively within the <think>...</think> and <answer>...</answer> tags. Do not include any content outside these tags or use any other tags.'
        #     input_text += ' During reasoning, draw upon your memorized knowledge and all previously retrieved external information to arrive at a final answer.'
        #     input_text += ' Keep your reasoning concise and direct—no more than three to four sentences. Ensure that you close the reasoning properly with </think>.'
        #     input_text += ' After reasoning, you MUST provide the final answer inside <answer> and </answer> without any further explanation. For example: <answer> Beijing </answer>.'

        # input_text += f' Question: {user_quesry}'

        # input_text = ''
        # # input_text += 'Answer the given question.'
        # input_text += 'You are a multi-step reasoner for the question-answering task. Your task is to generate ONLY one step forward.'
        # input_text += 'You MUST first conduct one step reasoning inside <think> and </think>.' 
        # input_text += 'Keep your reasoning concise and direct — no more than couple of sentences. Make sure to properly close the reasoning with </think>. '
        # if curr_node == 'think_search':
        #     input_text += " You are also equipped with a search engine and may use it as many times as needed." 
        #     input_text += " If you are uncertain or lack confidence in the information, you are encouraged to retrieve external knowledge."
        #     input_text += ' So, the reasoning should be concluded with a search query to find information helpful for answering the query.' # TODO 
        #     input_text += ' After reasoning, you MUST call a search engine by <search> query </search>.'
        # elif curr_node == 'think_answer':
        #     input_text += 'The reasoning should be concluded with a final answer for the user query.' # TODO
        #     # input_text += 'During reasoning, try to use your memorized knowledge and all external information provided so far to conclude and find the answer. '
        #     input_text += 'After reasoning, you MUST directly provide the answer inside <answer> and </answer> without detailed illustrations. For example, <answer> Beijing </answer>. '
        
        # input_text += f'Question: {user_quesry}'
        
        
        # input_text = ''
        # input_text += 'You are a multi-step reasoner for a question-answering task. '
        # input_text += 'At each step, your ONLY job is to generate one step of reasoning and an associated output. '
        # input_text += 'The reasoning MUST appear inside the tags <think> and </think>. '
        # # input_text += 'Do NOT include anything outside the required tags. No explanations, no formatting, no additional comments.\n\n'
        # # input_text += 'Keep your reasoning concise and direct — no more than couple of sentences. Make sure to properly close the reasoning with </think>. '
        # if curr_node == 'think_search':
        #     input_text += 'You are currently in the SEARCH stage.\n'
        #     input_text += 'At this stage, you do NOT answer the question. Instead, reason about the next information needed to answer part of the question.\n'
        #     input_text += 'You MAY use the search engine as many times as needed.\n'
        #     input_text += 'Your output MUST follow *exactly* this format:\n'
        #     input_text += '<think> one step of reasoning that leads to a search query </think>\n'
        #     input_text += '<search> your search query here </search>\n'
        #     input_text += 'Output ONLY these two tags and nothing else.\n'

        # elif curr_node == 'think_answer':
        #     input_text += 'You are currently in the ANSWER stage.\n'
        #     input_text += 'At this stage, you MUST conclude the task and provide the final answer to the main question.\n'
        #     input_text += 'Your output MUST follow *exactly* this format:\n'
        #     input_text += '<think> one step of reasoning that leads to the answer </think>\n'
        #     input_text += '<answer> your final answer here </answer>\n'
        #     input_text += 'Output ONLY these two tags and nothing else.\n'
        # input_text += f'\nQuestion: {user_quesry.strip()}'


        # input_text = ''
        # input_text += 'You are a multi-step reasoner for a question-answering task.\n'
        # input_text += 'At each step, you must reason one step forward toward answering the question.\n'
        # input_text += 'You have access to both your internal knowledge and a search engine.\n'
        # input_text += 'Use your internal knowledge whenever possible. Use the search engine only when additional information is needed.\n'
        # input_text += 'Your output must always begin with a concise reasoning step inside <think> and </think> tags.\n'
        # input_text += 'Output NOTHING outside the required tags.\n\n'

        # if curr_node == 'think_search':
        #     input_text += 'You are in the SEARCH stage.\n'
        #     input_text += 'Your goal is to reason about what needs to be known next — not to answer the full question yet.\n'
        #     input_text += 'After reasoning, if retrieval is needed, output a search query inside <search> and </search> tags.\n'
        #     input_text += 'If retrieval is not needed, still produce a plausible query that would help someone gather supporting information.\n'
        #     input_text += 'You must ONLY output the following two tags and nothing else:\n'
        #     input_text += '<think> your one-step reasoning here </think>\n'
        #     input_text += '<search> your search query here </search>\n'

        # elif curr_node == 'think_answer':
        #     input_text += 'You are in the ANSWER stage.\n'
        #     input_text += 'You may rely on both your internal knowledge and any previously retrieved information to answer the question.\n'
        #     input_text += 'After reasoning, output the final answer inside <answer> and </answer> tags.\n'
        #     input_text += 'You must ONLY output the following two tags and nothing else:\n'
        #     input_text += '<think> your final reasoning leading to the answer </think>\n'
        #     input_text += '<answer> your answer here </answer>\n'


        # input_text = ''
        # input_text += 'You are a multi-step reasoner in a question-answering task. '
        # input_text += 'At each step, generate only one step of reasoning toward answering the question. '
        # input_text += 'You may use both your internal knowledge and a search engine as needed. '
        # input_text += 'If any retrieved documents are provided, they will be enclosed in <information> and </information> tags. '
        # input_text += 'Treat <information> as read-only input — do NOT generate or modify anything inside <information> tags. '
        # input_text += 'Always begin by writing your reasoning inside <think> and </think>. '
        # input_text += 'Your output must contain ONLY the required tags. Do not include anything else.\n'

        # if curr_node == 'think_search':
        #     input_text += 'You are in the SEARCH stage. '
        #     input_text += 'Your goal is to identify the next piece of information needed — not to answer the question yet. '
        #     input_text += 'After reasoning, provide a helpful search query inside <search> and </search>. '
        #     input_text += 'Output only the following tags, in this exact order:\n'
        #     input_text += '<think> one step of reasoning </think>. '
        #     input_text += '<search> search query </search>.'

        # elif curr_node == 'think_answer':
        #     input_text += 'You are in the ANSWER stage. '
        #     input_text += 'Your goal is to answer the main question. '
        #     input_text += 'Use internal knowledge or prior information as needed. '
        #     input_text += 'After reasoning, provide the final answer inside <answer> and </answer>. '
        #     input_text += 'Output only the following tags, in this exact order:\n'
        #     input_text += '<think> final reasoning step </think>.'
        #     input_text += '<answer> final answer </answer>.'

        # input_text += f'\nQuestion: {user_quesry.strip()}'
    
    
    
        input_text = ''
        input_text += 'You are a multi-step reasoner in a question-answering task. '
        input_text += 'At each step, generate exactly one reasoning step toward answering the question.\n'
        input_text += 'You may use your internal knowledge or retrieved information if needed.\n'
        input_text += 'Retrieved documents, if any, will be provided inside <information> and </information> tags.\n'
        input_text += 'Treat <information> as read-only input. NEVER generate <information> tags yourself.\n'
        input_text += 'All reasoning must be enclosed in ONE and ONLY ONE pair of <think> and </think> tags.\n'
        input_text += 'Do NOT generate multiple <think> tags. Do NOT repeat or summarize information.\n'
        input_text += 'Your output must contain ONLY the required tags. No commentary, formatting, or extra output is allowed.\n\n'

        if curr_node == 'think_search':
            input_text += 'You are in the SEARCH stage.\n'
            input_text += 'Your goal is to reason about what information is needed next — not to answer the question.\n'
            input_text += 'After reasoning, provide a search query inside one <search> and </search> tag.\n'
            input_text += 'Your output must include only these two tags in this exact order:\n'
            input_text += '<think> one step of reasoning </think>\n'
            input_text += '<search> search query </search>\n'

        elif curr_node == 'think_answer':
            input_text += 'You are in the ANSWER stage.\n'
            input_text += 'You may use the question, your internal knowledge, and any input from <information>.\n'
            input_text += 'You MUST generate exactly ONE reasoning step in a single <think> block.\n'
            input_text += 'Do NOT generate multiple reasoning steps or multiple <think> tags.\n'
            input_text += 'Do NOT generate any <information> tags.\n'
            input_text += 'After the reasoning, output the final answer inside a single <answer> tag.\n'
            input_text += 'Your output must include only these two tags in this exact order:\n'
            input_text += '<think> one complete reasoning step leading to the final answer </think>\n'
            input_text += '<answer> final answer </answer>\n'

        input_text += f'\nQuestion: {user_quesry.strip()}'

        # Path so far
        for item_idx in solution_trace:
            solution_item = solution_trace[item_idx]
            node_keys = list(solution_item.keys())
            node_type = node_keys[0]
            
            if node_type == 'think_search':
                input_text += f"<think> {solution_item[node_type]['think']} </think>\n"
                input_text += f"<search> {solution_item[node_type]['search_query']} </search>\n"
                docs = solution_item[node_type]['retrieved_documents']
                if len(docs) > 0:
                    input_text += f"<information>"
                    for i, doc in enumerate(docs):
                        input_text += f"Doc [{i+1}]: {doc}\n"
                    input_text += f"<\information>\n"
            
        return input_text
    
    def generate_think_search(self, solution_trace: Dict[int, Dict[str, str]]):
        qid = solution_trace[0]['qid']
        user_question = solution_trace[0]['user_question']
        think_pattern = r'<think>(.*?)</think>'
        search_query_pattern = r'<search>(.*?)</search>'
        
        ### = Do generation
        input_prompt_text = self.get_prompt_text_v2('think_search', solution_trace)
        initial_output = self.generate(input_prompt_text, max_new_tokens=self.args.max_new_token, num_return=1)[0]
        # print('\n\n')
        # print(input_prompt_text)
        # print('\n-------')
        # print(initial_output)
        
        ### = Post-processing
        thinks = ', '.join(([t.strip() for t in re.findall(think_pattern, initial_output, re.DOTALL)]))
        search_query_match = re.search(search_query_pattern, initial_output, re.DOTALL)
        search_query = search_query_match.group(1).strip() if search_query_match else user_question
        
        ### = Check if regenerate needed
        if thinks == '':
            print(f"Think is not provided for query {qid}")
            for i in range(self.args.retry):
                print(f"Think, try {i+1} ...")
                output = self.generate(
                    input_prompt_text,
                    max_new_tokens=self.args.max_new_token,
                    num_return=1,
                    temperature=0.7,
                    do_sample=True
                )[0]
                
                thinks = ', '.join(([t.strip() for t in re.findall(think_pattern, output, re.DOTALL)]))
                if thinks != '':
                    search_query_match = re.search(search_query_pattern, output, re.DOTALL)
                    search_query = search_query_match.group(1).strip() if search_query_match else ''
                    break
            else:
                print(f"Failed to generate 'think' after all retries for query {qid}")
                   
        if search_query == '':
            print(f"Search Query is not provided for query {qid}")
            input_prompt_text_ = input_prompt_text + f'<think> {thinks} </think>\n'
            for i in range(self.args.retry):
                print(f"Search Query, try {i+1} ...")
                output = self.generate(
                    input_prompt_text_,
                    max_new_tokens=self.args.max_new_token,
                    num_return=1,
                    temperature=0.7,
                    do_sample=True
                )[0]
                
                search_query_match = re.search(search_query_pattern, output, re.DOTALL)
                search_query = search_query_match.group(1).strip() if search_query_match else ''
                if search_query != '':
                    break
            else:
                print(f"Failed to generate 'search query' after all retries for query {qid}")
        
        ### = Do retrieval
        if search_query != '':
            retrieved_docs_, _, _ = self.retriever.retrieve([search_query], [qid], [], [])
            retrieved_docs = retrieved_docs_[0]
        else:
            retrieved_docs = []
            
            
        return thinks, search_query, retrieved_docs
    
    
    def generate_think_answer(self, solution_trace: Dict[int, Dict[str, str]]):
        qid = solution_trace[0]['qid']
        user_question = solution_trace[0]['user_question']
        think_pattern = r'<think>(.*?)</think>'
        answer_pattern = r'<answer>(.*?)</answer>'
        
        ### = Do generation
        input_prompt_text = self.get_prompt_text_v2('think_answer', solution_trace)
        initial_output = self.generate(
            input_prompt_text,
            max_new_tokens=self.args.max_new_token,
            num_return=1
        )[0]        
        # print('\n\n')
        # print(input_prompt_text)
        # print('\n-------')
        # print(initial_output)
        
        ### = Post-processing
        thinks = ', '.join(([t.strip() for t in re.findall(think_pattern, initial_output, re.DOTALL)]))
        answer_match = re.search(answer_pattern, initial_output, re.DOTALL)
        most_likely_answer = answer_match.group(1).strip() if answer_match else ''
        
        ### = Check if regenerate needed
        if thinks == '':
            print(f"Think is not provided for query {qid}")
            for i in range(self.args.retry):
                print(f"Think, try {i+1} ...")
                output = self.generate(
                    input_prompt_text,
                    max_new_tokens=self.args.max_new_token,
                    num_return=1,
                    temperature=0.7,
                    do_sample=True
                )[0]
                
                thinks = ', '.join(([t.strip() for t in re.findall(think_pattern, output, re.DOTALL)]))
                if thinks != '':
                    answer_match = re.search(answer_pattern, output, re.DOTALL)
                    most_likely_answer = answer_match.group(1).strip() if answer_match else ''
                    break
            else:
                print(f"Failed to generate 'think' after all retries for query {qid}")
        
        input_prompt_text_ = input_prompt_text + f'<think> {thinks} </think>\n' if thinks != '' else input_prompt_text
        if most_likely_answer == '':
            print(f"The most-likely answer is not provided for query {qid}")
            
            for i in range(self.args.retry):
                print(f"The most-likely answer, try {i+1} ...")
                output = self.generate(
                    input_prompt_text_,
                    max_new_tokens=16,
                    num_return=1,
                    temperature=0.7,
                    do_sample=True
                )[0]
                answer_match = re.search(answer_pattern, output, re.DOTALL)
                most_likely_answer = answer_match.group(1).strip() if answer_match else ''
                
                if most_likely_answer != '':
                    break
            else:
                print(f"Failed to generate the 'most-likely answer' after all retries for query {qid}")
        
        ### = Generate more 
        output_list = self.generate(
            input_prompt_text, #input_prompt_text_,
            max_new_tokens=self.args.max_new_token,
            num_return=self.mcts_num_last_votes,
            temperature=0.7,
            do_sample=True
        )
        answer_list = []
        for output in output_list:
            answer_match = re.search(answer_pattern, output, re.DOTALL)
            answer = answer_match.group(1).strip() if answer_match else ''
            answer_list.append(answer)
        
        answer_list_ = [most_likely_answer] if len(answer_list) == 0 else [ans for ans in answer_list if ans]
        answer, value = self._get_most_likely_answer(user_query=user_question, output_list=answer_list_)

        return thinks, most_likely_answer, value