import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import torch
import transformers
from typing import List, Dict

from run_rag_methods.src.generators import StopOnSequence

from utils.general_utils import read_txt, get_query, get_answer, passages2string


class Generator:
    """Generator generates children nodes"""
    def __init__(self, args, retriever, generator, tokenizer, mcts_type="generation") -> None:
        self.args = args
        self.retriever = retriever
        self.mcts_type = mcts_type
        
        # --- Define model ------------
        self.generation_model = generator
        self.tokenizer = tokenizer
        self.eos_token_ids = self.generation_model.config.eos_token_id
        self.mcts_num_last_votes = args.mcts_num_last_votes
        
        # --- Prompts -----------------
        self.semantic_equivalence_prompt = read_txt(self.args.semantic_equivalence_prompt_file)
        self.retrieve_documents_prompt = read_txt(self.args.retrieve_documents_prompt_file)
        self.documents_analysis_prompt = read_txt(self.args.documents_analysis_prompt_file)
        self.documents_rethinking_prompt = read_txt(self.args.documents_rethinking_prompt_file)
        self.answer_generation_prompt = read_txt(self.args.answer_generation_prompt_file)
        self.answer_generation_only_answer_prompt = read_txt(args.answer_generation_only_answer_prompt_file)
        self.answer_validation_prompt = read_txt(self.args.answer_validation_prompt_file)
        
        # --- EoS tokens --------------
        search_target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
        answer_target_sequences = ["</answer>", " </answer>", "</answer>\n", " </answer>\n", "</answer>\n\n", " </answer>\n\n"]
        think_target_sequences = ["</think>", " </think>", "</think>\n", " </think>\n", "</think>\n\n", " </think>\n\n"]
        documents_analysis_target_sequences = ["</documents_analysis>", " </documents_analysis>", "</documents_analysis>\n", " </documents_analysis>\n", "</documents_analysis>\n\n", " </documents_analysis>\n\n"]
        critical_rethinking_target_sequences = ["</critical_rethinking>", " </critical_rethinking>", "</critical_rethinking>\n", " </critical_rethinking>\n", "</critical_rethinking>\n\n", " </critical_rethinking>\n\n"]
        answer_validation_target_sequences = ["</answer_validation>", " </answer_validation>", "</answer_validation>\n", " </answer_validation>\n", "</answer_validation>\n\n", " </answer_validation>\n\n"]
        self.search_stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(search_target_sequences, self.tokenizer)])
        self.answer_stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(answer_target_sequences, self.tokenizer)])
        self.think_stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(think_target_sequences, self.tokenizer)])
        self.documents_analysis_stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(documents_analysis_target_sequences, self.tokenizer)])
        self.critical_rethinking_stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(critical_rethinking_target_sequences, self.tokenizer)])
        self.answer_validation_stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(answer_validation_target_sequences, self.tokenizer)])
    
    # === LLM Generation ==========
    def generate_sequential(self,
        input_text,
        stopping_criteria=None,
        max_new_tokens=1024,
        num_return:int = 1,
        temperature:float = 0.7,
        do_sample:bool = True
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
                    temperature=temperature,
                    do_sample=do_sample,
                )
                generated_tokens = outputs[0][input_ids.shape[1]:]
                output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                generated_texts.append(output_text)
        
        return generated_texts
    
    def generate_batch(self,
        input_text,
        stopping_criteria,
        max_new_tokens = 1024,
        num_return:int = 5,
        temperature:float = 1.0,
        do_sample:bool = True
    ):    
        if self.tokenizer.chat_template:
            input_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": input_text}],
                add_generation_prompt=True,
                tokenize=False
            )
        
        inputs = self.tokenizer(input_prompt, return_tensors='pt').to(self.generation_model.device)
        input_ids = inputs["input_ids"]
        batch_size = input_ids.shape[0]
        
        with torch.no_grad():
            model_output = self.generation_model.generate(
                **inputs,
                return_dict_in_generate=True,
                output_logits=True,
                max_new_tokens=max_new_tokens,
                # stopping_criteria=stopping_criteria,
                num_return_sequences=num_return,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=temperature,
                do_sample=do_sample,
            )
            # model_output.past_key_values = None
            # model_output.sequences = model_output.sequences.cpu()
            
            # if type(eos_token_id) == list:
            #     temp = torch.stack([torch.argmax((model_output.sequences[:, len(input_ids[0]):] == eos).to(dtype=torch.int), dim=-1,) for eos in eos_token_id]).T
            #     for i in range(len(temp)):
            #         if_eos = False
            #         for j in range(len(temp[i])):
            #             if temp[i][j] != 0:
            #                 if_eos = True
            #                 break
            #         if if_eos == False:#if it doesn't contain eos token
            #             temp[i][-1] = model_output.sequences.shape[1] - len(input_ids[0])  - 1
            #     indices = [torch.min(temp[i][temp[i] > 0]).item() for i in range(len(temp))]
            # else:
            #     indices = torch.argmax((model_output.sequences[:, len(input_ids[0]):] == eos_token_id).to(dtype=torch.int), dim=-1,)
            # indices[indices == 0] = model_output.sequences.shape[1] - len(input_ids[0])
            
            # tokens = [seq[len(input_ids[0]): indices[i] + len(input_ids[0])].tolist() for i, seq in enumerate(model_output.sequences)]
            # tokens_text = [[self.tokenizer.decode(token) for token in tokens_] for tokens_ in tokens]
            # generated_texts = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        
            all_generated = model_output.sequences[:, input_ids.shape[1]:]  # strip prompt

            # Group into batches
            grouped = all_generated.view(batch_size, num_return, -1)
            generated_texts = [
                self.tokenizer.batch_decode(group, skip_special_tokens=True)
                for group in grouped
            ][0]
        
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
                        
                        response = self.generate_sequential(
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
    def get_prompt_text(self, curr_node, solution_trace: Dict[int, Dict[str, str]]):
        user_query = solution_trace[0]['user_query'] 
        
        if curr_node == 'retrieve_documents':
            prompt_text = self.retrieve_documents_prompt
        elif curr_node == 'documents_analysis':
            prompt_text = self.documents_analysis_prompt
        elif curr_node == 'documents_rethinking':
            prompt_text = self.documents_rethinking_prompt
        elif curr_node == 'answer_generation':
            prompt_text = self.answer_generation_prompt
        elif curr_node == 'answer_generation_only_answer':
            prompt_text = self.answer_generation_only_answer_prompt
        elif curr_node == 'answer_validation':
            prompt_text = self.answer_validation_prompt
        else:
            raise NotImplementedError("Node type not implemented!")
        prompt_text += f"Question: {user_query.strip()}\n\n\n"
        
        # Path so far
        for item_idx in solution_trace:
            solution_item = solution_trace[item_idx]
            node_keys = list(solution_item.keys())
            node_type = node_keys[0]
        
            if node_type == 'retrieve_documents':
                prompt_text += f"<think> {solution_item[node_type]['think']} </think>\n"
                prompt_text += f"<search> {solution_item[node_type]['search_query']} </search>\n"
                docs = solution_item[node_type]['docs']
                prompt_text += f"<information> {passages2string(docs)}</information>\n" if len(docs) > 0 else ''
        
            elif node_type == "documents_analysis":
                prompt_text += f"<documents_analysis> {solution_item[node_type]['documents_analysis']} </documents_analysis>\n"
                prompt_text += f"<search> {solution_item[node_type]['search_query']} </search>\n"
                docs = solution_item[node_type]['docs']
                prompt_text += f"<information> {passages2string(docs)}</information>\n" if len(docs) > 0 else ''
                
            elif node_type == "documents_rethinking":
                prompt_text += f"<critical_rethinking> {solution_item[node_type]['critical_rethinking']} </critical_rethinking>\n"
                prompt_text += f"<search> {solution_item[node_type]['search_query']} </search>\n"
                docs = solution_item[node_type]['docs']
                prompt_text += f"<information> {passages2string(docs)}</information>\n" if len(docs) > 0 else ''
                
            elif node_type == "answer_generation":
                prompt_text += f"<think> {solution_item[node_type]['think']} </think>\n"
                prompt_text += f'<answer> {solution_item[node_type]["answer"]} </answer>\n'
        
            elif node_type == "answer_validation":
                prompt_text += f'<answer_validation> {solution_item[node_type]["answer_validation"]} </answer_validation>\n'
        
        return prompt_text

    # === Actions =================
    def retrieve_documents(self, solution_trace: Dict[int, Dict[str, str]]):
        input_prompt_text = self.get_prompt_text('retrieve_documents', solution_trace)
        initial_output = self.generate_sequential(input_prompt_text, self.search_stopping_criteria, num_return=1)[0]
        think, search_query = self.retrieve_documents_postprocessing(solution_trace, input_prompt_text, initial_output)
        retrieved_docs = self.retriever.search(search_query) if search_query != '' else []
        return think, search_query, retrieved_docs
    
    def documents_analysis(self, solution_trace: Dict[int, Dict[str, str]]):
        input_prompt_text = self.get_prompt_text('documents_analysis', solution_trace)
        initial_output = self.generate_sequential(input_prompt_text, self.search_stopping_criteria, num_return=1)[0]
        documents_analysis, search_query = self.documents_analysis_postprocessing(solution_trace, input_prompt_text, initial_output)
        retrieved_docs = self.retriever.search(search_query) if search_query != '' else []
        return documents_analysis, search_query, retrieved_docs
    
    def documents_rethinking(self, solution_trace: Dict[int, Dict[str, str]]):
        input_prompt_text = self.get_prompt_text('documents_rethinking', solution_trace)
        initial_output = self.generate_sequential(input_prompt_text, self.search_stopping_criteria, num_return=1)[0]
        
        # print(input_prompt_text)
        # print('-')
        # print(initial_output)
        # print('--')
        
        critical_rethinking, search_query = self.documents_rethinking_postprocessing(solution_trace, input_prompt_text, initial_output)
        retrieved_docs = self.retriever.search(search_query) if search_query != '' else []
        return critical_rethinking, search_query, retrieved_docs
    
    def answer_generation(self, solution_trace: Dict[int, Dict[str, str]]):
        user_query = solution_trace[0]['user_query']
        input_prompt_text = self.get_prompt_text('answer_generation', solution_trace)
        initial_output = self.generate_sequential(input_prompt_text, self.answer_stopping_criteria, num_return=1)[0]
        think, initial_answer = self.answer_generation_postprocessing(solution_trace, input_prompt_text, initial_output)
        
        # Node reward
        input_prompt_text_batch = self.get_prompt_text('answer_generation_only_answer', solution_trace)
        input_prompt_text_ = input_prompt_text_batch + f"<think> {think} </think>\n"
        batch_output = self.generate_batch(input_prompt_text_, self.answer_stopping_criteria, num_return=5)
        batch_output_ = [get_answer(output_text) for output_text in batch_output]
        answer_, node_reward = self._get_most_likely_answer(user_query=user_query, output_list=batch_output_)
        answer = answer_ if answer_ else initial_answer
        
        return think, answer, node_reward
    
    def answer_validation(self, solution_trace: Dict[int, Dict[str, str]]):
        input_prompt_text = self.get_prompt_text('answer_validation', solution_trace)
        initial_output = self.generate_sequential(input_prompt_text, self.answer_validation_stopping_criteria, num_return=1)[0]
        answer_validation = self.answer_validation_postprocessing(solution_trace, input_prompt_text, initial_output)
        return answer_validation
    
    def finish(self, solution_trace: Dict[int, Dict[str, str]]):
        is_finished = True
        last_answer_gen = None
        for step_id in sorted(solution_trace):
            node_dict = solution_trace[step_id]
            if "answer_generation" in node_dict:
                last_answer_gen = (step_id, node_dict["answer_generation"])
        
        if last_answer_gen:
            node_reward = last_answer_gen[1]['node_reward']
        else:
            node_reward = 0.0
        
        return is_finished, node_reward
    
    
    # ===
    def retrieve_documents_postprocessing(self, solution_trace, input_prompt_text, output):
        qid = solution_trace[0]['qid']
        think_pattern = r'<think>(.*?)</think>'
        thinks = ', '.join(([t.strip() for t in re.findall(think_pattern, output, re.DOTALL)]))
        search_query = get_query(output) 
        
        # --- Check if regeneration is needed
        if thinks == '':
            print(f"Think is not provided for query {qid}!!!")
            for i in range(self.args.retry):
                print(f"Think, try {i+1} ...")
                output = self.generate_sequential(
                    input_prompt_text,
                    self.search_stopping_criteria,
                    temperature=1.0,
                )[0]
                thinks = ', '.join(([t.strip() for t in re.findall(think_pattern, output, re.DOTALL)]))
                if thinks != '':
                    search_query = get_query(output) 
                    break
            else:
                print(f"Failed to generate 'think' after all retries for query {qid}")
                
        # --- Check if search query regeneration is needed
        if search_query == None:
            print(f"Retrieve_documents (SQ) is not provided for query {qid}")
            input_prompt_text_ = input_prompt_text + f'<think> {thinks} </think>\n'
            for i in range(self.args.retry):
                print(f"Search Query, try {i+1} ...")
                output = self.generate_sequential(
                    input_prompt_text_,
                    self.search_stopping_criteria,
                    temperature=1.0,
                )[0]
                
                search_query = get_query(output)
                if search_query != None:
                    break
            else:
                print(f"Failed to generate 'retrieve_documents (SQ) after all retries for query {qid}")
        search_query = '' if search_query == None else search_query

        return thinks, search_query

    def documents_analysis_postprocessing(self, solution_trace, input_prompt_text, output):
        qid = solution_trace[0]['qid']
        documents_analysis_pattern = r'<documents_analysis>(.*?)</documents_analysis>'
        documents_analysis_ = ', '.join(([t.strip() for t in re.findall(documents_analysis_pattern, output, re.DOTALL)]))
        search_query = get_query(output)
        
        # --- Check if regeneration is needed
        if documents_analysis_ == '':
            print(f"Documents_analysis is not provided for query {qid}!!!")
            for i in range(self.args.retry):
                print(f"Documents_analysis {qid}, try {i+1} ...")
                output = self.generate_sequential(
                    input_prompt_text,
                    self.documents_analysis_stopping_criteria,
                    temperature=1.0,
                )[0]
                documents_analysis_ = ', '.join(([t.strip() for t in re.findall(documents_analysis_pattern, output, re.DOTALL)]))
                if documents_analysis_ != '':
                    break
            else:
                print(f"Failed to generate 'documents_analysis' after all retries for query {qid}")
        
        # --- Check if search query regeneration is needed
        if search_query == None:
            print(f"Documents analysis (SQ) is not provided for query {qid}")
            input_prompt_text_ = input_prompt_text + f'<documents_analysis> {documents_analysis_} </documents_analysis>\n'
            for i in range(self.args.retry):
                print(f"Search Query, try {i+1} ...")
                output = self.generate_sequential(
                    input_prompt_text_,
                    self.search_stopping_criteria,
                    temperature=1.0,
                )[0]
                
                search_query = get_query(output)
                if search_query != None:
                    break
            else:
                print(f"Failed to generate 'documents analysis (SQ)' after all retries for query {qid}")
        search_query = '' if search_query == None else search_query
        
        return documents_analysis_, search_query
    
    def documents_rethinking_postprocessing(self, solution_trace, input_prompt_text, output):
        qid = solution_trace[0]['qid']
        critical_rethinking_pattern = r'<critical_rethinking>(.*?)</critical_rethinking>'
        critical_rethinking_ = ', '.join(([t.strip() for t in re.findall(critical_rethinking_pattern, output, re.DOTALL)]))
        search_query = get_query(output)
        
        # --- Check if regeneration is needed
        if critical_rethinking_ == '':
            print(f"critical_rethinking is not provided for query {qid}!!!")
            for i in range(self.args.retry):
                print(f"critical_rethinking {qid}, try {i+1} ...")
                output = self.generate_sequential(
                    input_prompt_text,
                    self.critical_rethinking_stopping_criteria,
                    temperature=1.0,
                )[0]
                critical_rethinking_ = ', '.join(([t.strip() for t in re.findall(critical_rethinking_pattern, output, re.DOTALL)]))
                if critical_rethinking_ != '':
                    break
            else:
                print(f"Failed to generate 'critical_rethinking' after all retries for query {qid}")
        
        # --- Check if search query regeneration is needed
        if search_query == None:
            print(f"Documents analysis (SQ) is not provided for query {qid}")
            input_prompt_text_ = input_prompt_text + f'<critical_rethinking> {critical_rethinking_} </critical_rethinking>\n'
            for i in range(self.args.retry):
                print(f"Search Query, try {i+1} ...")
                output = self.generate_sequential(
                    input_prompt_text_,
                    self.search_stopping_criteria,
                    temperature=1.0,
                )[0]
                
                search_query = get_query(output)
                if search_query != None:
                    break
            else:
                print(f"Failed to generate 'critical_rethinking (SQ)' after all retries for query {qid}")
        search_query = '' if search_query == None else search_query
        
        return critical_rethinking_.strip(), search_query
    
    def answer_generation_postprocessing(self, solution_trace, input_prompt_text, output):
        qid = solution_trace[0]['qid']
        think_pattern = r'<think>(.*?)</think>'
        thinks = ', '.join(([t.strip() for t in re.findall(think_pattern, output, re.DOTALL)]))
        most_likely_answer = get_answer(output)
        
        ### = Check if regenerate needed
        if thinks == '':
            print(f"Think is not provided for query {qid}")
            for i in range(self.args.retry):
                print(f"Think, try {i+1} ...")
                output = self.generate_sequential(
                    input_prompt_text,
                    self.answer_stopping_criteria,
                    temperature=1.0,
                )[0]
                
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
                output = self.generate_sequential(
                    input_prompt_text_,
                    self.answer_stopping_criteria,
                    temperature=1.0
                )[0]
                most_likely_answer = get_answer(output)
                if most_likely_answer != None:
                    break
            else:
                print(f"Failed to generate the 'most-likely answer' after all retries for query {qid}")
        most_likely_answer = '' if most_likely_answer == None else most_likely_answer
        
        return thinks, most_likely_answer.strip()

    def answer_validation_postprocessing(self, solution_trace, input_prompt_text, output):
        qid = solution_trace[0]['qid']
        answer_validation_pattern = r'<answer_validation>(.*?)</answer_validation>'
        answer_validation_ = ', '.join(([t.strip() for t in re.findall(answer_validation_pattern, output, re.DOTALL)]))
        if answer_validation_ == '':
            print(f"answer_validation is not provided for query {qid}!!!")
            for i in range(self.args.retry):
                print(f"answer_validation {qid}, try {i+1} ...")
                output = self.generate_sequential(
                    input_prompt_text,
                    self.answer_validation_stopping_criteria,
                    temperature=1.0,
                )[0]
                answer_validation_ = ', '.join(([t.strip() for t in re.findall(answer_validation_pattern, output, re.DOTALL)]))
                if answer_validation_ != '':
                    break
            else:
                print(f"Failed to generate 'answer_validation' after all retries for query {qid}")
        
        return answer_validation_.strip()


    
    