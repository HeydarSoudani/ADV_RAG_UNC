
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import spacy
import torch
import random
import transformers
from typing import List, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
# import TruthTorchLM as ttlm


from run_searchr1.inference import get_think, get_query, get_answer, get_critique, _passages2string, StopOnSequence
from utils.general_utils import read_txt
from utils.adaptive_utils import fix_tokenizer_chat
from src_adaptive.templetes import SYSTEM_PROMPT_SHORTFORM
from src_mcts import examplers

nlp = spacy.load("en_core_web_sm")

examples = [
    {   
        "dataset": "hotpotqa",
        "qid": "train_36021",
        "question": "Where was the team which drafted Brenden Blair Morrow based when it was founded?",
        "reasoning_path": [
            {
                "think": "I need to find out where the team which drafted Brenden Blair Morrow was based when it was founded. I'll search for it.",
                "search_query": "Brenden Blair Morrow"
            },
            {
                "think": "I found out that the team which drafted Brenden Blair Morrow is the Dallas Stars. Now I need to find out where the Dallas Stars were based when it was founded.",
                "search_query": "Dallas Stars founded"
            },
            {
                "think": "I found out that the Dallas Stars were founded in Bloomington, Minnesota. Now I can provide the answer.",
                "answer": "Bloomington, Minnesota"
            },
            
        ]
    },
    {
        "dataset": "hotpotqa",
        "qid": "train_74035",
        "question": "The sister of Britney Spears starred as what character in the show based off Zoey 101?",
        "reasoning_path": [
            {
                "think": "I need to find the sister of Britney Spears who starred as a character in a show based off Zoey 101. I'll search for it.",
                "search_query": "The sister of Britney Spears"
            },
            {
                "think": "I found out that the sister of Britney Spears is Jamie Lynn Spears. Now I need to find out if she starred as a character in a show based off Zoey 101.",
                "search_query": "starred as a character in a show based off Zoey 101",
            },
            {
                "think": "I found out that Jamie Lynn Spears starred as Zoey Brooks in the show based off Zoey 101.",
                "answer": "Zoey Brooks",
            }
        ]
    }
]

class Counter:
    def __init__(self):
        self.retrieve = 0
        self.generate = 0
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
            "token_count": self.token - other_counter.token, 
            "sentence_count": self.sentence - other_counter.sentence 
        }

class Generator:
    """Generator generates children nodes"""
    def __init__(self, args, retriever, mcts_type="generation") -> None:
        self.args = args
        self.retriever = retriever
        self.mcts_type = mcts_type
        self.counter = Counter()
                
        # --- Define model ------------
        self.generation_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            # use_fast=False
        )
        self.eos_token_ids = self.generation_model.config.eos_token_id
        self.mcts_num_last_votes = args.mcts_num_last_votes
        
        # Prompts
        self.semantic_equivalence_prompt = read_txt(self.args.semantic_equivalence_prompt_file)
        self.fewshot_examples = examples
          
        # EoS tokens
        search_target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
        answer_target_sequences = ["</answer>", " </answer>", "</answer>\n", " </answer>\n", "</answer>\n\n", " </answer>\n\n"]
        self.search_stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(search_target_sequences, self.tokenizer)])
        self.answer_stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(answer_target_sequences, self.tokenizer)])

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

    def generate_(self,
        input_text,
        stopping_criteria,
        max_new_tokens=1024,
        num_return:int = 1,
        temperature:float = 1.0,
        do_sample:bool = False
    ):
        if self.tokenizer.chat_template:
            input_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": input_text}],
                add_generation_prompt=True,
                tokenize=False
            )
        
        input_ids = self.tokenizer.encode(input_prompt, return_tensors='pt').to(self.generation_model.device)
        attention_mask = torch.ones_like(input_ids)
        
        generated_texts = []
        for i in range(num_return):
            with torch.no_grad():
                outputs = self.generation_model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7
                )
                generated_tokens = outputs[0][input_ids.shape[1]:]
                output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                generated_texts.append(output_text)
        
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
    
    # === Prompt ==================
    def get_instruction(self, node_type):
        # === V1
        # input_text = ''
        # input_text += 'You are a multi-step reasoner in a question-answering task. '
        # input_text += 'At each step, generate exactly one reasoning step toward answering the question.\n'
        # input_text += 'You may use your internal knowledge or retrieved information if needed.\n'
        # input_text += 'Retrieved documents, if any, will be provided inside <information> and </information> tags.\n'
        # input_text += 'Treat <information> as read-only input. NEVER generate <information> tags yourself.\n'
        # input_text += 'All reasoning must be enclosed in ONE and ONLY ONE pair of <think> and </think> tags.\n'
        # # input_text += 'Do NOT generate multiple <think> tags. Do NOT repeat or summarize information.\n'
        # input_text += 'Your output must contain ONLY the required tags. No commentary, formatting, or extra output is allowed.\n\n'

        # if curr_node == 'think_search':
        #     input_text += 'You are in the SEARCH stage.\n'
        #     input_text += 'Your goal is to reason about what information is needed next — not to answer the question.\n'
        #     input_text += 'After reasoning, provide a search query inside one <search> and </search> tag.\n'
        #     input_text += 'Your output must include only these two tags in this exact order:\n'
        #     input_text += '<think> one complete reasoning step leading to a search query </think>\n'
        #     input_text += '<search> search query </search>\n'
        # elif curr_node == 'think_answer':
        #     input_text += 'You are in the ANSWER stage.\n'
        #     input_text += 'You may use the question, your internal knowledge, and any input from <information>.\n'
        #     # input_text += 'You MUST generate exactly ONE reasoning step in a single <think> block.\n'
        #     # input_text += 'Do NOT generate multiple reasoning steps or multiple <think> tags.\n'
        #     input_text += 'Do NOT generate any <information> tags.\n'
        #     input_text += 'After the reasoning, output the final answer inside a single <answer> tag.\n'
        #     input_text += 'Your output must include only these two tags in this exact order:\n'
        #     input_text += '<think> one complete reasoning step leading to the final answer </think>\n'
        #     input_text += '<answer> final answer </answer>\n'
        # input_text += f'\nQuestion: {user_query.strip()}\n'
        # 
        # === V2
        input_text = ''
        input_text += 'You are a multi-step reasoner in a question-answering task. '
        input_text += 'Your overall task is to answer the question through a series of intermediate reasoning steps.\n'
        input_text += 'At each step, generate exactly one reasoning step toward answering the question.\n'
        input_text += 'You may use your internal knowledge or retrieved information if needed.\n'
        input_text += 'Retrieved documents, if any, will be provided inside <information> and </information> tags.\n'
        input_text += 'Treat <information> as read-only input. NEVER generate or alter <information> tags yourself.\n'
        input_text += 'NEVER include anything outside the required tags. DO NOT add explanations, introductions, or extra formatting.\n'
        input_text += 'Your output must not contain anything else beyond what is explicitly required.\n\n'

        if node_type == 'think_search':
            input_text += 'You are in the THINK-SEARCH stage.\n'
            input_text += 'Your goal is to identify what specific information is missing and required to move closer to the answer.\n'
            input_text += 'DO NOT attempt to answer the question yet.\n'
            input_text += 'The search query should be precise and focused.\n'
            input_text += 'All reasoning must be enclosed within ONE and ONLY ONE pair of <think> and </think> tags.\n'
            input_text += 'Only include the following tags in this exact order:\n'
            input_text += '<think> one complete reasoning step leading to a search query </think>\n'
            input_text += '<search> search query </search>\n'
        
        elif node_type == 'think_answer':
            input_text += 'You are in the THINK-ANSWER stage.\n'
            input_text += 'Use your internal knowledge and any available <information> content to reason toward the answer.\n'
            input_text += 'Do NOT generate or modify <information> tags in your output.\n'
            input_text += 'Ensure your reasoning is directly connected to the provided information and leads logically to the final answer.\n'
            input_text += 'The final answer must be short, concise, and to the point.\n'
            input_text += 'All reasoning must be enclosed within ONE and ONLY ONE pair of <think> and </think> tags.\n'
            input_text += 'Only include the following tags in this exact order:\n'
            input_text += '<think> one complete reasoning step leading to the final answer </think>\n'
            input_text += '<answer> final answer </answer>\n'
        
        elif node_type == 'critique_search':
            input_text += 'You are in the CRITIQUE-SEARCH stage.\n'
            input_text += 'Your goal is to critically assess both your internal knowledge and the content of the retrieved documents.\n'
            input_text += 'Consider the possibility that these documents may contain inaccuracies, biases, or outdated information.\n'
            input_text += 'Reflect on how these potential issues could affect the reliability of the information provided.\n'
            input_text += 'Based on this critical assessment, formulate a new search query aimed at retrieving alternative or more reliable information sources.\n'
            input_text += 'Formulate a new search query that explores the question from a fresh perspective, utilizing creative strategies like rephrasing, employing synonyms, or considering related concepts.\n'
            input_text += 'All reasoning must be enclosed within ONE and ONLY ONE pair of <critique> and </critique> tags.\n'
            input_text += 'Only include the following tags in this exact order:\n'
            input_text += '<critique> one complete critical assessment and reasoning leading to a new search query </critique>\n'
            input_text += '<search> new search query </search>\n'
        
        elif node_type == 'critique_answer':
            input_text += 'You are in the CRITIQUE-ANSWER stage.\n'
            input_text += 'Your goal is to critically evaluate both your internal knowledge and the content of the retrieved documents.\n'
            input_text += 'Consider the possibility that these documents may contain inaccuracies, biases, or outdated information.\n'
            input_text += 'Reflect on how these potential issues could affect the reliability of the information provided.\n'
            input_text += 'Compare the information from the documents with your internal knowledge to identify any discrepancies or confirmations.\n'
            input_text += 'Based on this critical evaluation, reason carefully toward a new and improved final answer.\n'
            input_text += 'The final answer must be short, concise, and to the point.\n'
            input_text += 'Use <information> content carefully without generating or modifying the tags.\n'
            input_text += 'All reasoning must be enclosed within ONE and ONLY ONE pair of <critique> and </critique> tags.\n'
            input_text += 'Only include the following tags in this exact order:\n'
            input_text += '<critique> one complete critical evaluation and reasoning leading to a new final answer </critique>\n'
            input_text += '<answer> new final answer </answer>\n'
    
        # Add Examplers
        if self.args.enable_fewshot_examples:
            input_text += '\nExamples:\n'
            for example in self.fewshot_examples:
                question = example['question'].strip()
                if question[-1] != '?':
                    question += '?'
                input_text += f"Question: {example['question']}\n"
                for reasoning_step in example["reasoning_path"][:-1]:
                    input_text += f'<think> {reasoning_step["think"]} </think>\n'
                    input_text += f'<search> {reasoning_step["search_query"]} </search>\n'
                if node_type == 'think_answer':
                    input_text += f'<think> {example["reasoning_path"][-1]["think"]} </think>\n'
                    input_text += f'<answer> {example["reasoning_path"][-1]["answer"]} </answer>\n'
                input_text += '\n'

        input_text += f'\nQuestion: '
        
        return input_text
        
    def get_prompt_text(self, curr_node, solution_trace: Dict[int, Dict[str, str]]):
        if self.mcts_type == "generation":
            return self.get_prompt_text_generation(curr_node, solution_trace)
        elif self.mcts_type == "discrimination":
            return self.get_prompt_text_discrimination(curr_node, solution_trace)
    
    def get_prompt_text_generation(self, cur_node_type, solution_trace: Dict[int, Dict[str, str]]):
        user_query = solution_trace[0]['user_question'] 
        input_text = self.get_instruction(cur_node_type)
        input_text += f"{user_query.strip()}\n"
        
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
                    input_text += f"<information> {_passages2string(docs)}</information>\n"
            elif node_type == 'critique_search':
                input_text += f"<critique> {solution_item[node_type]['critique']} </critique>\n"
                input_text += f"<search> {solution_item[node_type]['search_query']} </search>\n"
                docs = solution_item[node_type]['retrieved_documents']
                if len(docs) > 0:
                    input_text += f"<information> {_passages2string(docs)}</information>\n"
            
        return input_text
    
    def get_prompt_text_discrimination(self, cur_node_type, solution_trace: Dict[int, Dict[str, str]]):
        user_query = solution_trace[0]['user_question']
        answer_candidates = solution_trace[0]['answer_candidates']
        
        input_text = self.get_instruction(cur_node_type, user_query)
        input_text += 'Answer Candidates:\n'
        for idx, candidate in enumerate(answer_candidates, 1):
            input_text += f'- {candidate}\n'

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
                    input_text += f"<information> {_passages2string(docs)}</information>\n"
            
        return input_text
    
    def trace2text(self, solution_trace: Dict[int, Dict[str, str]]):
        input_text = ''
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
                    input_text += f"<information> {_passages2string(docs)}</information>\n"
            if node_type == 'critique_search':
                input_text += f"<critique> {solution_item[node_type]['critique']} </critique>\n"
                input_text += f"<search> {solution_item[node_type]['search_query']} </search>\n"
                docs = solution_item[node_type]['retrieved_documents']
                if len(docs) > 0:
                    input_text += f"<information> {_passages2string(docs)}</information>\n"
            if node_type == 'think_answer':
                input_text += f"<think> {solution_item[node_type]['think']} </think>\n"
                input_text += f"<answer> {solution_item[node_type]['answer']} </answer>\n"    
            if node_type == 'critique_answer':
                input_text += f"<critique> {solution_item[node_type]['critique']} </critique>\n"
                input_text += f"<answer> {solution_item[node_type]['answer']} </answer>\n"
            
        return input_text

    # === Actions ==================
    def generate_think_search(self, solution_trace: Dict[int, Dict[str, str]]):
        ### = Do generation
        input_prompt_text = self.get_prompt_text('think_search', solution_trace)
        initial_output = self.generate_(input_prompt_text, self.search_stopping_criteria, num_return=1)[0]        
        if self.args.use_counter:
            self.counter.add_generate(initial_output, self.tokenizer)
        # print('\n\n')
        # print(input_prompt_text)
        # print('\n-------')
        # print(initial_output)
        
        ### = Post-processing
        thinks, search_query = self.think_search_postprocessing(solution_trace, input_prompt_text, initial_output)
        
        ### = Do retrieval
        if search_query != '':
            retrieved_docs = self.retriever.search(search_query)
            if self.args.use_counter:
                self.counter.retrieve += 1
        else:
            retrieved_docs = []
              
        return thinks, search_query, retrieved_docs
    
    def generate_think_answer(self, solution_trace: Dict[int, Dict[str, str]]):
        ### = Do generation
        input_prompt_text = self.get_prompt_text('think_answer', solution_trace)
        initial_output = self.generate_(input_prompt_text, self.answer_stopping_criteria)[0] 
        if self.args.use_counter:
            self.counter.add_generate(initial_output, self.tokenizer)
        print('\n\n')
        print(input_prompt_text)
        print('\n-------')
        print(initial_output)
        
        ### = Post-processing
        think, most_likely_answer = self.think_answer_postprocessing(solution_trace, input_prompt_text, initial_output)
        # value = 0.9
        
        ### = Generate more 
        user_question = solution_trace[0]['user_question']
        input_prompt_text_ = input_prompt_text + f'<think> {think} </think>\n'
        output_list = self.generate_(input_prompt_text_, self.answer_stopping_criteria, num_return=self.mcts_num_last_votes)
        answer_list = [get_answer(output) for output in output_list]
        answer_list_ = [most_likely_answer] if len(answer_list) == 0 else [ans for ans in answer_list if ans]
        if len(answer_list_) > 0:
            answer, value = self._get_most_likely_answer(user_query=user_question, output_list=answer_list_)
        else:
            value = 0.001
        
        return think, most_likely_answer, value

    def generate_critique_search(self, solution_trace: Dict[int, Dict[str, str]]):
        ### = Do generation
        input_prompt_text = self.get_prompt_text('critique_search', solution_trace)
        initial_output = self.generate_(input_prompt_text, self.search_stopping_criteria, num_return=1)[0]        
        if self.args.use_counter:
            self.counter.add_generate(initial_output, self.tokenizer)
        # print('\n\n')
        # print(input_prompt_text)
        # print('\n-------')
        # print(initial_output)

        ### = Post-processing
        critiques, search_query = self.critique_search_postprocessing(solution_trace, input_prompt_text, initial_output)

        ### = Do retrieval
        if search_query != '':
            retrieved_docs = self.retriever.search(search_query)
            if self.args.use_counter:
                self.counter.retrieve += 1
        else:
            retrieved_docs = []
              
        return critiques, search_query, retrieved_docs

    def generate_critique_answer(self, solution_trace: Dict[int, Dict[str, str]]):
        ### = Do generation
        input_prompt_text = self.get_prompt_text('critique_answer', solution_trace)
        initial_output = self.generate_(input_prompt_text, self.answer_stopping_criteria)[0] 
        if self.args.use_counter:
            self.counter.add_generate(initial_output, self.tokenizer)
        # print('\n\n')
        # print(input_prompt_text)
        # print('\n-------')
        # print(initial_output)
        
        ### = Post-processing
        critiques, most_likely_answer = self.critique_answer_postprocessing(solution_trace, input_prompt_text, initial_output)
        # value = 0.9
        
        ### = Generate more 
        user_question = solution_trace[0]['user_question']
        input_prompt_text_ = input_prompt_text + f'<critique> {critiques} </critique>\n'
        output_list = self.generate_(input_prompt_text_, self.answer_stopping_criteria, num_return=self.mcts_num_last_votes)
        answer_list = [get_answer(output) for output in output_list]
        answer_list_ = [most_likely_answer] if len(answer_list) == 0 else [ans for ans in answer_list if ans]
        if len(answer_list_) > 0:
            answer, value = self._get_most_likely_answer(user_query=user_question, output_list=answer_list_)
        else:
            value = 0.001
        
        return critiques, most_likely_answer, value

    # ===
    def think_search_postprocessing(self, solution_trace, input_prompt_text, output):
        qid = solution_trace[0]['qid']
        think_pattern = r'<think>(.*?)</think>'
        thinks = ', '.join(([t.strip() for t in re.findall(think_pattern, output, re.DOTALL)]))
        search_query = get_query(output) 
        
        ### = Check if regenerate needed
        if thinks == '':
            print(f"Think is not provided for query {qid}")
            for i in range(self.args.retry):
                print(f"Think, try {i+1} ...")
                output = self.generate_(input_prompt_text, self.search_stopping_criteria, temperature=0.7, do_sample=True)[0]
                
                thinks = ', '.join(([t.strip() for t in re.findall(think_pattern, output, re.DOTALL)]))
                if thinks != '':
                    search_query = get_query(output) 
                    break
            else:
                print(f"Failed to generate 'think' after all retries for query {qid}")
                   
        if search_query == None:
            print(f"Search Query is not provided for query {qid}")
            input_prompt_text_ = input_prompt_text + f'<think> {thinks} </think>\n'
            for i in range(self.args.retry):
                print(f"Search Query, try {i+1} ...")
                output = self.generate_(input_prompt_text_, self.search_stopping_criteria)[0]
                
                search_query = get_query(output)
                if search_query != None:
                    break
            else:
                print(f"Failed to generate 'search query' after all retries for query {qid}")
        search_query = '' if search_query == None else search_query

        return thinks, search_query

    def think_answer_postprocessing(self, solution_trace, input_prompt_text, output):
        qid = solution_trace[0]['qid']
        think_pattern = r'<think>(.*?)</think>'
        thinks = ', '.join(([t.strip() for t in re.findall(think_pattern, output, re.DOTALL)]))
        most_likely_answer = get_answer(output)
        
        ### = Check if regenerate needed
        if thinks == '':
            print(f"Think is not provided for query {qid}")
            for i in range(self.args.retry):
                print(f"Think, try {i+1} ...")
                output = self.generate_(input_prompt_text, self.answer_stopping_criteria)[0]
                
                thinks = ', '.join(([t.strip() for t in re.findall(think_pattern, output, re.DOTALL)]))
                if thinks != '':
                    most_likely_answer = get_answer(output)
                    break
            else:
                print(f"Failed to generate 'think' after all retries for query {qid}")
        
        if most_likely_answer == None:
            print(f"The most-likely answer is not provided for query {qid}")
            input_prompt_text_ = input_prompt_text + f'<think> {thinks} </think>\n'
            for i in range(self.args.retry):
                print(f"The most-likely answer, try {i+1} ...")
                output = self.generate_(input_prompt_text_, self.answer_stopping_criteria)[0]
                most_likely_answer = get_answer(output)
                if most_likely_answer != None:
                    break
            else:
                print(f"Failed to generate the 'most-likely answer' after all retries for query {qid}")
        most_likely_answer = '' if most_likely_answer == None else most_likely_answer
        
        return thinks, most_likely_answer

    def critique_search_postprocessing(self, solution_trace, input_prompt_text, output):
        qid = solution_trace[0]['qid']
        critique_pattern = r'<critique>(.*?)</critique>'
        critiques = ', '.join(([t.strip() for t in re.findall(critique_pattern, output, re.DOTALL)]))
        search_query = get_query(output) 
        
        ### = Check if regenerate needed
        if critiques == '':
            print(f"Critique is not provided for query {qid}")
            for i in range(self.args.retry):
                print(f"Critique, try {i+1} ...")
                output = self.generate_(input_prompt_text, self.search_stopping_criteria, temperature=0.7, do_sample=True)[0]
                
                critiques = ', '.join(([t.strip() for t in re.findall(critique_pattern, output, re.DOTALL)]))
                if critiques != '':
                    search_query = get_query(output) 
                    break
            else:
                print(f"Failed to generate 'critique' after all retries for query {qid}")

        if search_query == None:
            print(f"Search Query is not provided for query {qid}")
            input_prompt_text_ = input_prompt_text + f'<critique> {critiques} </critique>\n'
            for i in range(self.args.retry):
                print(f"Search Query, try {i+1} ...")
                output = self.generate_(input_prompt_text_, self.search_stopping_criteria)[0]
                search_query = get_query(output)
                if search_query != None:
                    break
            else:
                print(f"Failed to generate 'search query' after all retries for query {qid}")
        search_query = '' if search_query == None else search_query

        return critiques, search_query

    def critique_answer_postprocessing(self, solution_trace, input_prompt_text, output):
        qid = solution_trace[0]['qid']
        critique_pattern = r'<critique>(.*?)</critique>'
        critiques = ', '.join(([t.strip() for t in re.findall(critique_pattern, output, re.DOTALL)]))
        most_likely_answer = get_answer(output)
        
        ### = Check if regenerate needed
        if critiques == '':
            print(f"Critique is not provided for query {qid}")
            for i in range(self.args.retry):
                print(f"Critique, try {i+1} ...")
                output = self.generate_(input_prompt_text, self.answer_stopping_criteria)[0]
                critiques = ', '.join(([t.strip() for t in re.findall(critique_pattern, output, re.DOTALL)]))
                if critiques != '':
                    most_likely_answer = get_answer(output)
                    break
            else:
                print(f"Failed to generate 'critique' after all retries for query {qid}")
        
        
        if most_likely_answer == None:
            print(f"The most-likely answer is not provided for query {qid}")
            input_prompt_text_ = input_prompt_text + f'<critique> {critiques} </critique>\n'
            for i in range(self.args.retry):
                print(f"The most-likely answer, try {i+1} ...")
                output = self.generate_(input_prompt_text_, self.answer_stopping_criteria)[0]
                most_likely_answer = get_answer(output)
                if most_likely_answer != None:
                    break
            else:
                print(f"Failed to generate the 'most-likely answer' after all retries for query {qid}")
        most_likely_answer = '' if most_likely_answer == None else most_likely_answer
        
        return critiques, most_likely_answer












    
    # def get_prompt_text(self, cur_node_type, solution_trace: Dict[int, Dict[str, str]], docs, prev_subqs=None, cur_subq=None):
    #     input_text = ''
    #     user_quesry = solution_trace[0]['user_question']
        
    #     # Generate Answer or Subanswer
    #     if cur_node_type in ['direct_answer', 'rag_answer']:
    #         sub_docs = []
    #         # Path
    #         for item_idx in solution_trace:
    #             solution_item = solution_trace[item_idx]
    #             node_keys = list(solution_item.keys())
    #             node_type = node_keys[0]
    #             if node_type == 'user_question':
    #                 user_quesry = solution_item[node_type]
    #             elif node_type == 'rephrased_query':
    #                 input_text += f'Given the question: {user_quesry}\n'
    #                 input_text += f'We rephrase the question, which can also be expressed as: \n{solution_item[node_type]}\n'
    #             elif node_type == 'subquestions':
    #                 input_text += f'We then decompose the question into several sub-questions with their corresponding answers:\n'
    #             elif node_type == 'subq_direct_answer':
    #                 input_text += f"{solution_item[node_type]['subquestion']} {solution_item[node_type]['subanswer']}\n"
    #             elif node_type == 'subq_rag_answer':
    #                 input_text += f"{solution_item[node_type]['subquestion']} {solution_item[node_type]['subanswer']}\n"
    #                 sub_docs.extend(solution_item[node_type]['documents'])
    #             elif node_type == 'rag_answer':
    #                 sub_docs.extend(solution_item[node_type]['documents'])
                
    #         # Docs
    #         if len(docs) > 0:
    #             input_text += "Below are some relevant documents that may help answer the question:\n"
    #             for i, doc in enumerate(docs):
    #                 input_text += f"[{i+1}] {doc}\n"
    #             input_text += "\n"
    #         if len(sub_docs) > 0:
    #             for i, doc in enumerate(sub_docs):
    #                 input_text += f"[{len(docs)+i+1}] {doc}\n"
    #             input_text += "\n"
                
    #         # Few-shot examples
    #         if len(self.fewshot_examplers) > 0:
    #             input_text += "Here are several examples of how to answer similar questions:\n\n"
    #             for exp in self.fewshot_examplers:
    #                 input_text += f"Question: {exp['question']}\n"
    #                 input_text += f"Answer: {exp['answer']}\n"
    #             input_text += "\n"
            
            
    #         input_text += "Now, answer the following question EXACTLY in the format of the examples above.\n"
    #         input_text += "DO NOT add any introductory phrases, explanations, or extra text.\n\n"
    #         input_text += f"Question: {user_quesry}\nAnswer:"
        
    #     elif cur_node_type == 'subquestions':
    #         input_text += "Given an input question, decompose it into multiple smaller and indivisible sub-questions.\n"
    #         input_text += "Here are several examples of how to output the sub-questions:\n\n"
    #         for exp in self.fewshot_examplers:
    #             input_text += f"Original Question: {exp['question']}\n"
    #             input_text += "Subquestions:\n"
    #             for i, subq in enumerate(exp['subqa']):
    #                 input_text += f"{i+1}.{subq[0]}\n"
    #             input_text += "\n"
            
    #         input_text += "Now, generate sub-questions for the following question EXACTLY in the format of the examples above.\n"
    #         input_text += "DO NOT add any introductory phrases, explanations, or extra text.\n\n"
    #         input_text += f'Original Question: {user_quesry}\n'
    #         input_text += "Subquestions:\n"
        
    #     elif cur_node_type == 'rephrased_query':
    #         input_text += "Given an input question, rephrase it into a more intuitive and easier-to-understand version.\n"
    #         input_text += "Here are several examples of how to output the rephrased question:\n\n"
    #         for exp in self.fewshot_rephrased_examplers:
    #             input_text += f"Original Question: {exp['question']}\n"
    #             input_text += f"Rephrased Question: {exp['Rephrased']}\n\n"
            
    #         input_text += "Now, generate the rephrased version for the following question EXACTLY in the format of the examples above.\n"
    #         input_text += "DO NOT add any introductory phrases, explanations, or extra text.\n\n"
    #         input_text += f'Original Question: {user_quesry}\n'
    #         input_text += f'Rephrased Question: '
        
    #     elif cur_node_type in ['subq_direct_answer', 'subq_rag_answer']:
    #         sub_docs = []
    #         # Path
    #         for item_idx in solution_trace:
    #             solution_item = solution_trace[item_idx]
    #             node_keys = list(solution_item.keys())
    #             node_type = node_keys[0]
    #             if node_type == 'user_question':
    #                 input_text += f'Given the question: {solution_item[node_type]}\n'
    #             elif node_type == 'rephrased_query':
    #                 input_text += f'We rephrase the question, which can also be expressed as: \n{solution_item[node_type]}\n\n'

    #         if len(prev_subqs) > 0:
    #             input_text += f'We then decompose the question into several sub-questions, some of them are answered:\n'
    #             for psub in prev_subqs:
    #                 input_text += f"Sub-question: {psub[0]} sub-answer: {psub[1]}\n"
    #         else:
    #             input_text += f'We then decompose the question into several sub-questions, we want to answer one by one.\n'

    #         # Docs
    #         if len(docs) > 0:
    #             input_text += "Below are some relevant documents that may help answer the question:\n"
    #             for i, doc in enumerate(docs):
    #                 input_text += f"[{i+1}] {doc}\n"
    #             input_text += "\n"
        
    #         # Few-shot examples
    #         if len(self.fewshot_examplers) > 0:
    #             input_text += "Here are several examples of how to answer similar questions:\n\n"
    #             for exp in self.fewshot_examplers:
    #                 input_text += f"Question: {exp['question']}\n"
    #                 input_text += f"Answer: {exp['answer']}\n"
    #             input_text += "\n"
        
    #         input_text += "Summarizing the information above, now answer the following question EXACTLY in the format of the examples above.\n"
    #         input_text += "DO NOT add any introductory phrases, explanations, or extra text.\n\n"
    #         input_text += f"Question: {cur_subq}\nAnswer:"
        
    #     return input_text
    
    # def generate_rephrased_question(self, solution_trace: Dict[int, Dict[str, str]]):
    #     input_prompt_text = self.get_prompt_text('rephrased_query', solution_trace, [], [])
    #     output = self.generate(
    #         input_prompt_text,
    #         max_new_tokens=128,
    #         num_return=1,
    #     )[0]
    #     return output
    
    # def generate_direct_answer(self, solution_trace: Dict[int, Dict[str, str]]):
    #     subs = []
    #     user_question = solution_trace[0]['user_question']
        
    #     for _, cur_node in solution_trace.items():
    #         node_key = list(cur_node.keys())[0]
    #         if node_key in ['subq_direct_answer', 'subq_rag_answer']:
    #             subquestion = cur_node[node_key]['subquestion']
    #             subanswer = cur_node[node_key]['subanswer']
    #             subs.append((subquestion, subanswer))
            
    #         if node_key is "rephrased_query":
    #             user_question = cur_node[node_key]
        
    #     # = Do generation
    #     input_prompt_text = self.get_prompt_text('direct_answer', solution_trace, [], subs)
    #     output_list = self.generate(
    #         input_prompt_text,
    #         max_new_tokens=16,
    #         num_return=self.mcts_num_last_votes,
    #         sys_prompt=SYSTEM_PROMPT_SHORTFORM
    #     )
    #     answer, value = self._get_most_likely_answer(user_query=user_question, output_list=output_list)
    #     return answer, value

    # def generate_rag_answer(self, solution_trace: Dict[int, Dict[str, str]]):
    #     # = Get query
    #     qid = solution_trace[0]['qid']
    #     user_question = solution_trace[0]['user_question']
    #     for _, cur_node in solution_trace.items():
    #         node_key = list(cur_node.keys())[0]
    #         if node_key is "rephrased_query":
    #             user_question = cur_node[node_key]

    #     # = Do retrieval
    #     docs, _, _ = self.retriever.retrieve([user_question], [qid], [], [])
        
    #     # = Do generation
    #     input_prompt_text = self.get_prompt_text('rag_answer', solution_trace, docs[0], [])
    #     output_list = self.generate(
    #         input_prompt_text,
    #         max_new_tokens=16,
    #         num_return=self.mcts_num_last_votes,
    #         sys_prompt=SYSTEM_PROMPT_SHORTFORM
    #     )
    #     answer, value = self._get_most_likely_answer(user_query=user_question, output_list=output_list)
    #     return docs, answer, value
        
    # def generate_query_decomposition(self, solution_trace: Dict[int, Dict[str, str]]):
    #     input_prompt_text = self.get_prompt_text('subquestions', solution_trace, [], [])
    #     output = self.generate(
    #         input_prompt_text,
    #         max_new_tokens=128,
    #         num_return=1,
    #     )[0]
    #     subquestions = re.findall(r'\d+\.(.*)', output)
    #     subquestions = [s.strip() for s in subquestions]
    #     return subquestions
       
    # def generate_subq_direct_answer(self, solution_trace: Dict[int, Dict[str, str]]):
    #     # Get subquestion
    #     last_node = solution_trace[list(solution_trace.keys())[-1]]
    #     node_type = list(last_node.keys())[0]
    #     subs = []
        
    #     if node_type == "subquestions":
    #         subquestion = last_node[node_type][0]
    #         subquestion_pointer = 1
    #         len_subqs = len(last_node["subquestions"])
        
    #     elif node_type in ["subq_direct_answer", "subq_rag_answer"]:
    #         cur_pointer = last_node[node_type]["subq_pointer"]
    #         subquestion_pointer = cur_pointer + 1
            
    #         for id, node in solution_trace.items():
    #             node_key = list(node.keys())[0]
    #             if node_key == "subquestions":
    #                 len_subqs = len(node["subquestions"])
    #                 subquestion = node["subquestions"][cur_pointer]
                    
    #             if node_key in ['subq_direct_answer', 'subq_rag_answer']:
    #                 subq = node[node_key]['subquestion']
    #                 suba = node[node_key]['subanswer']
    #                 subd = node[node_key]['documents'] if node_key=='subq_rag_answer' else []
    #                 subs.append((subq, suba, subd))    
            
    #     # Do generation
    #     input_prompt_text = self.get_prompt_text('subq_direct_answer', solution_trace, [], subs, subquestion)
    #     output_list = self.generate(
    #         input_prompt_text,
    #         max_new_tokens=32,
    #         num_return=self.mcts_num_last_votes,
    #     )
    #     subanswer, _ = self._get_most_likely_answer(user_query=subquestion, output_list=output_list)
    #     return subquestion, subanswer, subquestion_pointer, len_subqs
        
    # def generate_subq_rag_answer(self, solution_trace: Dict[int, Dict[str, str]]):
    #     # Get subquestion
    #     qid = solution_trace[0]['qid']
    #     last_node = solution_trace[list(solution_trace.keys())[-1]]
    #     node_type = list(last_node.keys())[0]
    #     subs = []
        
    #     if node_type == "subquestions":
    #         subquestion = last_node[node_type][0]
    #         subquestion_pointer = 1
    #         len_subqs = len(last_node["subquestions"])
        
    #     elif node_type in ["subq_direct_answer", "subq_rag_answer"]:
    #         cur_pointer = last_node[node_type]["subq_pointer"]
    #         subquestion_pointer = cur_pointer + 1
            
    #         for id, node in solution_trace.items():
    #             node_key = list(node.keys())[0]
    #             if node_key == "subquestions":
    #                 len_subqs = len(node["subquestions"])
    #                 subquestion = node["subquestions"][cur_pointer]
                    
    #             if node_key in ['subq_direct_answer', 'subq_rag_answer']:
    #                 subq = node[node_key]['subquestion']
    #                 suba = node[node_key]['subanswer']
    #                 subd = node[node_key]['documents'] if node_key=='subq_rag_answer' else []
    #                 subs.append((subq, suba, subd))

    #     # Do retrieval
    #     ret_query = ', '.join([f"{sub[0]} {sub[1]}" for sub in subs])
    #     ret_query += f", {subquestion}"
    #     docs, _, _ = self.retriever.retrieve([ret_query], [qid], [], [])
        
    #     # Do generation
    #     input_prompt_text = self.get_prompt_text('subq_rag_answer', solution_trace, docs[0], subs, subquestion)
    #     output_list = self.generate(
    #         input_prompt_text,
    #         max_new_tokens=32,
    #         num_return=self.mcts_num_last_votes,
    #     )
    #     subanswer, _ = self._get_most_likely_answer(user_query=subquestion, output_list=output_list)
        
    #     return docs, subquestion, subanswer, subquestion_pointer, len_subqs
    
    