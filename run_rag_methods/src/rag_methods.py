
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import copy
import spacy
import torch
import requests
import numpy as np
from math import exp
from bs4 import BeautifulSoup
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, StoppingCriteriaList

from run_rag_methods.src.generators import LLMGenerator, StopOnSequence
from run_rag_methods.src.retrievers_local import BM25Retriever, ContrieverRetriever, RerankRetriever, DenseRetriever
from run_rag_methods.src.templetes import (
    SYSTEM_PROMPT_DIRECT, SYSTEM_PROMPT_COT, SYSTEM_PROMPT_IRCOT,
    SYSTEM_PROMPT_DRAGIN,
    SELF_ASK_PROMPT_SINGLE_HOP, SELF_ASK_PROMPT_MULTI_HOP,
    REACT_INSTRUCTION,
    get_singleqa_search_o1_instruction, get_multiqa_search_o1_instruction,
    get_task_instruction_openqa
)
from utils.general_utils import passages2string

nlp = spacy.load("en_core_web_sm")

# def passages2string(retrieval_result):
#     format_reference = ''
#     for idx, doc_item in enumerate(retrieval_result):
#         content = doc_item['contents']
#         title = content.split("\n")[0]
#         text = "\n".join(content.split("\n")[1:])
#         # format_reference += f"Wikipedia Title: {title}\n{text}\n\n"
#         format_reference += f"Doc {idx+1} (Title: {title}) {text}\n"
#     return format_reference

# Adaptive RAG style
def get_answer(text):
    parts = text.split("the answer is: ", 1)  # Split at the first occurrence
    if len(parts) <= 1:
        return None
    pred = parts[1].strip() if len(parts) > 1 else ""
    pattern = r"\.?</s>"
    pred = re.sub(pattern, "", pred)
    pred = pred.rstrip(".?!")
    return pred

class BasicRAG:
    def __init__(self, generation_model, generation_tokenizer, device, args):
        self.args = args
        self.generator = LLMGenerator(generation_model, generation_tokenizer, device, args)
        
        # === Retrievers ============================= 
        if args.retriever_name == 'bm25':
            self.retriever = BM25Retriever(args)  
        elif args.retriever_name == 'contriever':
            self.retriever = ContrieverRetriever(args)
        elif args.retriever_name in ['rerank_l6', 'rerank_l12']:
            self.retriever = RerankRetriever(args)
        elif args.retriever_name in ['e5', 'bge']:
            self.retriever = DenseRetriever(args)
                
        # === Few-shot examples =======================
        self.DIRECT_EXAMPLES = [
            {
                "Q": 'Jeremy Theobald and Christopher Nolan share what profession?',
                "A": 'Producer'
            },
            {
                "Q": 'What film directed by Brian Patrick Butler was inspired by a film directed by F.W. Murnau?',
                "A": 'The Phantom Hour'
            },
            {
                "Q": 'How many episodes were in the South Korean television series in which Ryu Hye−young played Bo−ra?',
                "A": '20'
            },
            {
                "Q": 'Vertical Limit stars which actor who also played astronaut Alan Shepard in "The Right Stuff"?',
                "A": 'Scott Glenn'
            },
            {
                "Q": 'What was the 2014 population of the city where Lake Wales Medical Center is located?',
                "A": '15,140'
            },
            {
                "Q": 'Who was born first? Jan de Bont or Raoul Walsh?',
                "A": 'Raoul Walsh'
            },
            {
                "Q": 'In what country was Lost Gravity manufactured?',
                "A": 'Germany'
            },
            {
                "Q": 'Which of the following had a debut album entitled "We Have an Emergency": Hot Hot Heat or The Operation M.D.?',
                "A": 'The Operation M.D.'
            }
        ]
        self.COT_EXAMPLES = [
            {
                "Q": 'Jeremy Theobald and Christopher Nolan share what profession?',
                "A": 'Jeremy Theobald is an actor and producer. Christopher Nolan is a director, producer, and screenwriter. Therefore, they both share the profession of being a producer. So the answer is: producer'
            },
            {
                "Q": 'What film directed by Brian Patrick Butler was inspired by a film directed by F.W. Murnau?',
                "A": 'Brian Patrick Butler directed the film The Phantom Hour. The Phantom Hour was inspired by the films such as Nosferatu and The Cabinet of Dr. Caligari. Of these Nosferatu was directed by F.W. Murnau. So the answer is: The Phantom Hour'
            },
            {
                "Q": 'How many episodes were in the South Korean television series in which Ryu Hye−young played Bo−ra?',
                "A": 'The South Korean television series in which Ryu Hye−young played Bo−ra is Reply 1988. The number of episodes Reply 1988 has is 20. So the answer is: 20'
            },
            {
                "Q": 'Vertical Limit stars which actor who also played astronaut Alan Shepard in "The Right Stuff"?',
                "A": 'The actor who played astronaut Alan Shepard in "The Right Stuff" is Scott Glenn. The movie Vertical Limit also starred Scott Glenn. So the answer is: Scott Glenn'
            },
            {
                "Q": 'What was the 2014 population of the city where Lake Wales Medical Center is located?',
                "A": 'Lake Wales Medical Center is located in the city of Polk County, Florida. The population of Polk County in 2014 was 15,140. So the answer is: 15,140'
            },
            {
                "Q": 'Who was born first? Jan de Bont or Raoul Walsh?',
                "A": 'Jan de Bont was born on 22 October 1943. Raoul Walsh was born on March 11, 1887. Thus, Raoul Walsh was born the first. So the answer is: Raoul Walsh'
            },
            {
                "Q": 'In what country was Lost Gravity manufactured?',
                "A": 'The Lost Gravity (roller coaster) was manufactured by Mack Rides. Mack Rides is a German company. So the answer is: Germany'
            },
            {
                "Q": 'Which of the following had a debut album entitled "We Have an Emergency": Hot Hot Heat or The Operation M.D.?',
                "A": 'The debut album of the band "Hot Hot Heat" was "Make Up the Breakdown". The debut album of the band "The Operation M.D." was "We Have an Emergency". So the answer is: The Operation M.D.'
            },
            # {
            #     "Q": '',
            #     "A": ''
            # }
        ]

        self.direct_examples_text = '\n\n'.join([f'Q: {exp["Q"]}\nA: {exp["A"]}' for exp in self.DIRECT_EXAMPLES])
        self.cot_examples_text = '\n\n'.join([f'Q: {exp["Q"]}\nA: {exp["A"]}' for exp in self.COT_EXAMPLES])
        
        # === Prompt ================================
        self.user_prompt_with_context = "{examples}\n\n{documents}\nQ: {question}\nA:"
        self.user_prompt_wo_context = "{examples}\n\nQ: {question}\nA:"
        self.user_prompt_self_ask = "{documents}Quesiton: {question}\nAre follow up questions needed here: Yes.\n"

    def get_top_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[0] if len(sentences) > 0 else ""

    def get_last_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[-1] if len(sentences) > 0 else "" 
       
    def get_unique_docs(self, docs_lst:list):
        return list({doc['id']: doc for doc in docs_lst}.values()) 
    
    def regenerate(self, system_prompt, question, docs, path, with_context=True):
        user_prompt = self.user_prompt_with_context.format(examples=self.cot_examples_text, documents=passages2string(docs), question=question) \
            if with_context else self.user_prompt_wo_context.format(examples=self.cot_examples_text, question=question)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": ' '.join([step['think'] for step in path]) + " So the answer is:"}
        ]               
        _, output_text = self.generator.generate(messages)
        return output_text

class DirectInference(BasicRAG):
    def __init__(self, generation_model, generation_tokenizer, device, args):
        super().__init__(generation_model, generation_tokenizer, device, args)
        self.system_prompt = SYSTEM_PROMPT_DIRECT
        self.answer_template = '{answer}'
    
    def inference(self, question):
        path = []
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt_wo_context.format(examples=self.direct_examples_text, question=question)}
        ]
        _, output_text = self.generator.generate(messages)
        path.append({'think': '', 'answer': output_text})
        pred_answer = output_text
        
        return pred_answer, path

    # --
    def get_input_prompt_self_consistency(self, question, trace):
        prompt_text = self.user_prompt_wo_context.format(
            examples=self.direct_examples_text,
            question=question
        )
        return prompt_text
        
    def partial_inference_self_consistency(self, question, trace):
        prompt_text = self.get_input_prompt_self_consistency(question, trace)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt_text}
        ]
        answer_list = self.generator.generate_batch(
            messages,
            num_return=self.args.n_generations,
            temperature=self.args.consistency_temperature
        )
        return answer_list

class CoTInference(BasicRAG):
    def __init__(self, generation_model, generation_tokenizer, device, args):
        super().__init__(generation_model, generation_tokenizer, device, args)
        self.system_prompt = SYSTEM_PROMPT_COT
        self.answer_template = '{answer}'
    
    def get_think_and_answer(self, text):
        match = re.search(r'\s*So the answer is:\s*(.*)', text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            thinking = text[:match.start()].strip()
            return thinking, answer
        else:
            return text.strip(), None
    
    def inference(self, question):
        path = []
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt_wo_context.format(
                examples=self.cot_examples_text,
                question=question
            )}
        ]
        _, output_text = self.generator.generate(messages)
        think, pred_answer = self.get_think_and_answer(output_text)
        path.append({'think': think, 'answer': pred_answer})
        
        # Regenerate the last sentence if it is needed
        if "So the answer is:" not in output_text:
            output_text = self.regenerate(self.system_prompt, question, [], path, with_context=False)
            pred_answer = get_answer(output_text) if get_answer(output_text) else output_text
            path[-1]['answer'] = pred_answer
        
        return pred_answer, path
  
    # --
    def get_input_prompt_self_consistency(self, question, trace):
        prompt_text = self.user_prompt_wo_context.format(
            examples=self.cot_examples_text,
            question=question
        )
        prompt_text += f"{trace[-1].get('think', '')}" + " So the answer is:"
        return prompt_text
        
    def partial_inference_self_consistency(self, question, trace):
        prompt_text = self.get_input_prompt_self_consistency(question, trace)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt_text}
        ] 
        answer_list = self.generator.generate_batch(
            messages,
            num_return=self.args.n_generations,
            temperature=self.args.consistency_temperature
        )
        return answer_list
    
    def get_input_prompt_reasoning_consistency(self, question, trace):
        prompt_text = self.user_prompt_wo_context.format(
            examples=self.cot_examples_text,
            question=question
        )
        # prompt_text += f"{ trace[-1]['think']}"
        return prompt_text 
        
    def partial_inference_reasoning_consistency(self, input_prompt_text):
        messages = [{"role": "user", "content": input_prompt_text}]
        _, output_text = self.generator.generate(messages, temperature=self.args.consistency_temperature)
        think, pred_answer = self.get_think_and_answer(output_text)
        
        if "So the answer is:" not in output_text:
            input_prompt_text_ = input_prompt_text + f" {think} So the answer is:"
            messages = [{"role": "user", "content": input_prompt_text_}]
            _, output_text = self.generator.generate(messages, temperature=self.args.consistency_temperature)
            pred_answer = get_answer(output_text) if get_answer(output_text) else output_text
            
        return think, pred_answer

class CoTSingleRAG(BasicRAG):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.cot_examples_text = '\n\n'.join([f'Q: {exp["Q"]}\nA: {exp["A"]}' for exp in self.COT_EXAMPLES])
        self.system_prompt = SYSTEM_PROMPT_COT
    
    def inference(self, question):
        path = []
        search_docs = self.retriever.search(question)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt_with_context.format(
                examples=self.cot_examples_text,
                documents=search_docs,
                question=question
            )}
        ]
        _, output_text = self.generator.generate(messages)
        path.append({'search_query': question, 'docs': search_docs, 'think': output_text})
        
        # Regenerate the last sentence if it is needed
        if "So the answer is:" not in output_text:
            docs_tmp = [doc for step in path for doc in step['docs']]
            output_text = self.regenerate(self.system_prompt, question, docs_tmp, path)
            path.append({'think': f'So the answer is: {output_text}'})
            pred_answer = get_answer(output_text) if get_answer(output_text) else output_text
        else:
            pred_answer = get_answer(output_text)
        
        return pred_answer, path

class FixLengthRAG(BasicRAG):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.system_prompt = SYSTEM_PROMPT_IRCOT
    
    def inference(self, question):
        path, text = [], ""
        search_query = question
        while True:
            old_len = len(text)
            retrieved_docs = self.retriever.search(search_query)
        
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt_with_context.format(
                    examples=self.cot_examples_text,
                    documents=passages2string(retrieved_docs),
                    question=question
                )},
                {"role": "assistant", "content": text}
            ]
            _, new_text = self.generator.generate(
                messages,
                max_new_tokens=self.args.generate_fix_length
            )
            path.append({'search_query': search_query, 'docs': retrieved_docs, 'think': new_text})
            
            text = text.strip() + " " + new_text.strip()
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.args.max_new_tokens or len(text) <= old_len or "the answer is" in text:
                break
                
            search_query = new_text.strip()
        
        # Regenerate the last sentence if it is needed
        if "So the answer is:" not in text:
            docs_tmp = path[-1]['docs']
            output_text = self.regenerate(self.system_prompt, question, docs_tmp, path)
            path.append({'think': f'So the answer is: {output_text}'})
            pred_answer = get_answer(output_text) if get_answer(output_text) else output_text       
        else:
            pred_answer = get_answer(text)
        
        return pred_answer, path

class FixSentenceRAG(BasicRAG):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.system_prompt = SYSTEM_PROMPT_IRCOT

    def inference(self, question):
        path, text = [], ""
        search_query = question
        while True:
            old_len = len(text)
            retrieved_docs = self.retriever.search(search_query)
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt_with_context.format(
                    examples=self.cot_examples_text,
                    documents=passages2string(retrieved_docs),
                    question=question
                )},
                {"role": "assistant", "content": text}
            ]
            _, new_text = self.generator.generate(
                messages,
                stopping_criteria=self.generator.ircot_stopping_criteria
            )
            path.append({'search_query': search_query, 'docs': retrieved_docs, 'think': new_text})
            
            text = text.strip() + " " + new_text.strip()
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.args.max_new_tokens or len(text) <= old_len or "the answer is" in text:
                break
                
            search_query = new_text.strip()
        
        # Regenerate the last sentence if it is needed
        if "So the answer is:" not in text:
            docs_tmp = path[-1]['docs']
            output_text = self.regenerate(self.system_prompt, question, docs_tmp, path)
            path.append({'think': f'So the answer is: {output_text}'})
            pred_answer = get_answer(output_text) if get_answer(output_text) else output_text               
        else:
            pred_answer = get_answer(text)
        
        return pred_answer, path

class IRCOT_RAG(BasicRAG):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.system_prompt = SYSTEM_PROMPT_IRCOT

    def inference(self, question):
        # Initial retrieval
        search_query = question
        cur_search_docs = self.retriever.search(search_query)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt_with_context.format(
                examples=self.cot_examples_text,
                documents=passages2string(cur_search_docs),
                question=question
            )}
        ]
        
        path, iter_num = [], 0
        while iter_num < self.args.max_iter:
            output, output_text = self.generator.generate(messages, self.generator.ircot_stopping_criteria)
            path.append({'search_query': search_query, 'docs': cur_search_docs, 'think': output_text})
            pred_answer = get_answer(output_text)
            
            if "So the answer is:" in output_text:
                path.append({'think': output_text})
                break
            iter_num += 1
            if (output[-1].item() in self.generator.curr_eos) or (iter_num == self.args.max_iter):
                break  # Don't perform another retrieval or prompt construction
            
            search_query = output_text
            cur_search_docs = self.retriever.search(search_query) if search_query else []
            tmp = [doc for step in path for doc in step['docs']] + cur_search_docs
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt_with_context.format(
                    examples=self.cot_examples_text,
                    documents=passages2string(tmp),
                    question=question
                )},
                {"role": "assistant", "content": ' '.join([step['think'] for step in path])}
            ]
        
        # Regenerate the last sentence if it is needed
        if "So the answer is:" not in output_text:
            docs_tmp = [doc for step in path for doc in step['docs']]
            output_text = self.regenerate(self.system_prompt, question, docs_tmp, path)
            path.append({'think': f'So the answer is: {output_text}'})
            pred_answer = get_answer(output_text) if get_answer(output_text) else output_text       
        else:
            pred_answer = get_answer(output_text)
        
        return pred_answer, path

class FLARE_RAG_V1(BasicRAG):
    # Base on DRAGIN git code
    def __init__(self, args, device):
        super().__init__(args, device)
        self.modifier = self.modifier_token if args.modifier_method=='token' else self.modifier_entity
        self.system_prompt = SYSTEM_PROMPT_IRCOT
        
    def modifier_token(self, text, tokens, logprobs):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        tid = 0
        for sid, sent in enumerate(sentences):
            pos = 0
            tr = tid
            while tr < len(tokens):
                apr = sent[pos:].find(tokens[tr])
                if apr == -1:
                    break
                pos = apr + len(tokens[tr])
                tr += 1
            probs = [1 - exp(v) for v in logprobs[tid:tr+1]]
            probs = np.array(probs)
            p = {
                "avg": np.mean,
                "max": np.max,
                "min": np.min,
            }.get(self.args.sentence_solver, lambda x: 0)(probs)
            
            if p > self.args.hallucination_threshold: # hallucination
                # keep sentences before hallucination 
                prev = "" if sid == 0 else " ".join(sentences[:sid])
                # replace all hallucinated tokens in current sentence with [xxx]
                curr = sentences[sid]
                pos = 0
                # # 这里改成了替换掉最大的那个，而不是所有的
                # max_prob = 0
                # for prob, tok in zip(probs, tokens[tid:tr+1]):
                #     max_prob = max(prob, max_prob)
                for prob, tok in zip(probs, tokens[tid:tr+1]):
                    apr = curr[pos:].find(tok) + pos
                    if prob > self.args.hallucination_threshold:
                    # if prob == max_prob:
                        curr = curr[:apr] + "[xxx]" + curr[apr+len(tok):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(tok)
                return prev, curr, True
            tid = tr + 1
        
        # No hallucination
        return text, None, False
    
    # TODO: The implementation has problem ....
    def modifier_entity(self, text, tokens, logprobs):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        entity = []
        for sent in sentences:
            doc = nlp(sent)
            li = [ent.text for ent in doc.ents]
            entity.append(li)
        
        belonging = [-1] * len(text)
        pos = 0
        for tid, tok in enumerate(tokens):
            apr = text[pos:].find(tok) + pos
            assert apr != -1
            for j in range(pos, apr+len(tok)):
                belonging[j] = tid
            pos = apr + len(tok)
        
        entity_intv = []
        for sid, sent in enumerate(sentences):
            tmp = []
            pos = text.find(sent)
            for ent in entity[sid]:
                apr = text[pos:].find(ent) + pos
                el = belonging[apr]
                er = belonging[apr + len(ent) - 1]
                tmp.append((el, er))
                pos = apr + len(ent)
            entity_intv.append(tmp)

        entity_prob = []
        for ent_itv_per_sent in entity_intv:
            tmp = []
            for itv in ent_itv_per_sent:
                probs = np.array(logprobs[itv[0]:itv[1]+1])
                p = {
                    "avg": np.mean,
                    "max": np.max,
                    "min": np.min,
                    "first": lambda x: x[0] if len(x) > 0 else 0
                }.get(self.entity_solver, lambda x: 0)(probs)
                tmp.append(p)
            entity_prob.append(tmp)

        for sid in range(len(sentences)):
            if len(entity_prob[sid]) == 0:
                continue
            probs = [1 - exp(v) for v in entity_prob[sid]]
            probs = np.array(probs)
            p = {
                "avg": np.mean,
                "max": np.max,
                "min": np.min,
            }.get(self.args.sentence_solver, lambda x: 0)(probs)
            if p > self.args.hallucination_threshold: # hallucination
                # keep sentences before hallucination 
                prev = "" if sid == 0 else " ".join(sentences[:sid])
                # replace all hallucinated entities in current sentence with [xxx]
                curr = sentences[sid]
                pos = 0
                for prob, ent in zip(probs, entity[sid]):
                    apr = curr[pos:].find(ent) + pos
                    if prob > self.args.hallucination_threshold:
                        curr = curr[:apr] + "[xxx]" + curr[apr+len(ent):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(ent)
                return prev, curr, True
        # No hallucination
        return text, None, False
    
    def inference(self, question):
        
        path, text = [], ''
        while True:
            old_len = len(text)
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt_wo_context.format(
                    examples=self.cot_examples_text,
                    question=question
                )},
                {"role": "assistant", "content": text}
            ]
            # print(messages)
            # print('----')
            new_text, tokens_text, logprobs = self.generator.generate_with_scores(
                messages,
                max_new_tokens=self.args.max_new_tokens,
            )
            ptext, curr, hallucination = self.modifier(new_text, tokens_text, logprobs)
            
            # print(new_text)
            # print(ptext)
            # print('-')
            # print(curr)
            # print(hallucination)
            # print('----')
            
            if not hallucination:
                text = text.strip() + " " + new_text.strip()
                path.append({'search_query': '', 'docs': [], 'think': new_text.strip()})
            else:
                if self.args.query_formulation == "direct":
                    retrieve_question = curr.replace("[xxx]", "")
                elif self.args.query_formulation == "forward_all":
                    tmp_all = [question, text, ptext]
                    retrieve_question = " ".join(s for s in tmp_all if len(s) > 0)
                else:
                    raise NotImplemented

                search_docs = self.retriever.search(retrieve_question)
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.user_prompt_with_context.format(
                        examples=self.cot_examples_text,
                        documents=passages2string(search_docs),
                        question=question
                    )},
                    {"role": "assistant", "content": text + " " + ptext.strip()}
                ]
                _, new_text = self.generator.generate(messages, self.generator.ircot_stopping_criteria)
                text = text.strip() + " " + ptext.strip() + " " + new_text.strip()
                path.append({'search_query': retrieve_question, 'docs': search_docs, 'think': f"{ptext.strip()} {new_text.strip()}"})

            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.args.max_new_tokens or len(text) <= old_len or "the answer is" in text:
                break

        # Regenerate the last sentence if it is needed
        if "So the answer is:" not in text:
            docs_tmp = path[-1]['docs']
            output_text = self.regenerate(self.system_prompt, question, docs_tmp, path)
            path.append({'think': f'So the answer is: {output_text}'})
            pred_answer = get_answer(output_text) if get_answer(output_text) else output_text       
        else:
            pred_answer = get_answer(text)
        
        return pred_answer, path

class DRAGIN_RAG(BasicRAG):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.system_prompt = SYSTEM_PROMPT_DRAGIN
    
    def get_sentence_token_indices(self, text, tokens):
        doc = nlp(text)
        sentence_indices = []
        token_pointer = 0  # Keeps track of token position

        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_tokens = sent_text.split()  # Basic tokenization (ensure consistency with tokens)
            
            # Skip empty sentences
            if not sent_tokens:
                continue
            
            # Find the start index of the sentence in the token list
            while token_pointer < len(tokens) and tokens[token_pointer] != sent_tokens[0]:
                token_pointer += 1
            start_index = token_pointer
            
            # Find the end index of the sentence in the token list
            for _ in sent_tokens:
                if token_pointer < len(tokens):
                    token_pointer += 1
            end_index = token_pointer
            sentence_indices.append((sent_text, start_index, end_index))
        
        return sentence_indices
    
    def modifier(self, text, tokens, attentions, weight):
        sentences = self.get_sentence_token_indices(text, tokens) # [(sent, tl, tr), ...]
        for sid, (sent, tl, tr) in enumerate(sentences):
            attns = attentions[tl:tr]
            attns = np.array(attns) / sum(attns)
            value = [attns[i-tl] * weight[i] * (tr-tl) for i in range(tl, tr)] 
            thres = [1 if v > self.args.hallucination_threshold else 0 for v in value]
            
            if 1 in thres:
                # hallucinated
                if "check_real_words" in self.args and self.args.check_real_words:
                    doc = nlp(sent)
                    real_words = set(token.text for token in doc if token.pos_ in 
                        ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
                    def match(tok):
                        for word in real_words:
                            if word in tok:
                                return True
                        return False
                    for i in range(len(thres)):
                        if not match(tokens[tl+i]):
                            thres[i] = 0                
                
                prev = "" if sid == 0 else " ".join([sentences[i][0] for i in range(sid)])
                
                return True, prev, tokens[tl:tr], thres
            
        return False, text, None, None

    def keep_real_words(self, prev_text, curr_tokens, curr_hit):
        
        curr_text = " ".join(curr_tokens)
        all_text = prev_text + " " + curr_text
        input_ids = self.generator.tokenizer.encode(all_text, return_tensors="pt")
        input_length = input_ids.shape[1]
        tokens_tmp = self.generator.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        atten_tmp = self.generator.generator(
            input_ids.to(self.generator.generator.device),
            return_dict_in_generate=True,
            output_attentions=True
        ).attentions[-1][0]

        # merge tokens
        range_ = []
        for i, t in enumerate(tokens_tmp):
            if i == 0 or t.startswith(self.generator.space_token) or input_ids[0][i] == 13:
                range_.append([i, i])
            else:
                range_[-1][-1] += 1
        tokens = []
        for r in range_:
            tokenseq = "".join(tokens_tmp[r[0]: r[1]+1]).replace(self.generator.space_token, "")
            tokens.append(tokenseq)

        # 获取幻觉词对应的 attention
        # Get the attention corresponding to the hallucination word
        curr_st = len(tokens) - len(curr_tokens)
        atten_tmp = torch.mean(atten_tmp, dim=0)
        attns = []
        for r in range_:
            # att = torch.zeros(atten_tmp.shape[0], input_length)
            att = torch.zeros(input_length)
            for i in range(r[0], r[1] + 1):
                if i == 0:
                    continue
                v = atten_tmp[i-1][:r[0]] # 上一位的
                v = v / v.sum()
                t = torch.zeros(input_length)
                t[:r[0]] = v
                att += t
            att /= (r[1] - r[0] + 1)
            # merge token for att
            att = torch.tensor([att[rr[0]:rr[1]+1].sum() for rr in range_])
            attns.append(att)
            
        # 计算每个超过阈值的 token 在前文的 attentions
        # Calculate the attentions of each token exceeding the threshold in the previous text
        forward_attns = torch.zeros(len(tokens))
        hit_cnt = 0
        for i in range(len(curr_hit)):
            if curr_hit[i] == 1:
                forward_attns += attns[curr_st + i]
                hit_cnt += 1
        forward_attns /= hit_cnt
        forward_attns = forward_attns.tolist()

        # 分析词性，保留实词对应的 attns
        # Analyze the part of speech and keep the attns corresponding to the content words
        doc = nlp(all_text)
        real_words = set(token.text for token in doc if token.pos_ in 
                      ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
        
        def match(token):
            for word in real_words:
                if word in token:
                    return True
            return False
        
        real_pairs = []
        for i in range(len(tokens)):
            tok, att = tokens[i], forward_attns[i]
            if i >= curr_st and curr_hit[i - curr_st]:
                continue
            if match(tok):
                real_pairs.append((att, tok, i))
        

        if "retrieve_keep_top_k" in self.args:
            top_k = min(self.args.retrieve_keep_top_k, len(real_pairs))
        elif "retrieve_keep_ratio" in self.args:
            top_k = int(len(real_pairs) * self.args.retrieve_keep_ratio)
        
        real_pairs = sorted(real_pairs, key = lambda x:x[0], reverse=True)
        real_pairs = real_pairs[:top_k]
        real_pairs = sorted(real_pairs, key = lambda x:x[2])
        
        return " ".join([x[1] for x in real_pairs])
    
    def fetch_last_n_tokens(self, text, num, tokenizer):
        tokens = tokenizer.tokenize(text)
        if num >= len(tokens):
            return text
        last_n_tokens = tokens[-num:]
        last_n_sentence = ' '.join(last_n_tokens)
        return last_n_sentence
    
    def inference(self, question):
        path, text = [], ""
        while True:
            old_len = len(text)
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt_wo_context.format(
                    examples=self.cot_examples_text,
                    question=question
                )},
                {"role": "assistant", "content": text}
            ]
            new_text, tokens, attns, logprobs, entropies = self.generator.generate_attn(
                messages, self.args.max_new_tokens, use_entropy = True
            )
            weight = entropies if self.args.rag_method == "dragin" else [-v for v in logprobs]
            hallucination, ptext, curr_tokens, curr_hit =  self.modifier(new_text, tokens, attns, weight)
            
            # print(messages)
            # print('-')
            # print(new_text)
            # print(ptext)
            # print(curr_tokens)
            # print(curr_hit)
            # print('-')
            
            if not hallucination:
                text = text.strip() + " " + new_text.strip()
                path.append({'search_query': '', 'docs': [], 'think': new_text.strip()})
            else:
                forward_all = [question, text, ptext]
                forward_all = " ".join(s for s in forward_all if len(s) > 0)
            
                if self.args.query_formulation == "current":
                    retrieve_question = " ".join(curr_tokens)
                elif self.args.query_formulation == "current_wo_wrong":
                    retrieve_question = " ".join(list(curr_tokens[i] if curr_hit[i] == 0 else "" for i in range(len(curr_tokens))))
                elif self.args.query_formulation == "forward_all":
                    retrieve_question = forward_all
                elif self.args.query_formulation == "last_sentence":
                    retrieve_question = self.get_last_sentence(forward_all)
                elif self.args.query_formulation == "last_n_tokens":
                    assert "retrieve_keep_top_k" in self.args
                    retrieve_question = self.fetch_last_n_tokens(forward_all, self.args.retrieve_keep_top_k, self.generator.tokenizer)
                elif self.args.query_formulation == "real_words": 
                    retrieve_question = self.keep_real_words(
                        prev_text = question + " " + text + " " + ptext, 
                        curr_tokens = curr_tokens, 
                        curr_hit = curr_hit,
                    ) 
                else:
                    raise NotImplemented
                
                search_docs = self.retriever.search(retrieve_question)
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.user_prompt_with_context.format(
                        examples=self.cot_examples_text,
                        documents=passages2string(search_docs),
                        question=question
                    )},
                    {"role": "assistant", "content": text + " " + ptext.strip()}
                ]
                _, new_text = self.generator.generate(
                    messages,
                    max_new_tokens=self.args.max_new_tokens
                )
                text = text.strip() + " " + ptext.strip() + " " + new_text.strip()
                
                path.append({'search_query': retrieve_question, 'docs': search_docs, 'think': f"{ptext.strip()} {new_text.strip()}"})
            
                # print(messages)
                # print('-')
                # print(retrieve_question)
                # print(new_text)
            
            # print('----') 
               
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.args.max_new_tokens or len(text) <= old_len or "the answer is" in text:
                break
            
        
        # Regenerate the last sentence if it is needed
        if "So the answer is:" not in text:
            docs_tmp = path[-1]['docs']
            output_text = self.regenerate(self.system_prompt, question, docs_tmp, path)
            path.append({'think': f'So the answer is: {output_text}'})
            pred_answer = get_answer(output_text) if get_answer(output_text) else output_text       
        else:
            pred_answer = get_answer(text)
        
        return pred_answer, path

class SelfAsk_RAG(BasicRAG):
    def __init__(self, generation_model, generation_tokenizer, device, args):
        super().__init__(generation_model, generation_tokenizer, device, args)
        self.single_hop = False
        self.system_prompt = SELF_ASK_PROMPT_MULTI_HOP
        self.FOLLOW_UP_PATTERN = r"Follow up:.*\n"
        self.answer_template = '{answer}'
    
    def documents2string(self, retrieval_result):
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Context{idx+1}: {text}\n"

        return format_reference
    
    def extract_follow_up(self, text: str) -> str:
        match = re.search(r'Follow up:\s*(.*?)\nIntermediate answer:', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
        
    def extract_intermediate(self, text: str) -> str:
        match = re.search(r'(.*?)(?:Follow up:|So the final answer is:)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def extract_final_answer(self, text: str) -> str:
        parts = text.split("So the final answer is: ", 1)  # Split at the first occurrence
        if len(parts) <= 1:
            return None
        pred = parts[1].strip() if len(parts) > 1 else ""
        pattern = r"\.?</s>"
        pred = re.sub(pattern, "", pred)
        pred = pred.rstrip(".?!")
        return pred

    def inference(self, question):
        path, text = [], ""
        
        # Initial retrieval
        search_query = question
        cur_search_docs = self.retriever.search(search_query)
        user_input_prompt = self.user_prompt_self_ask.format(
            documents = self.documents2string(cur_search_docs),
            question=question
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input_prompt}
        ]
        path.append({'think': '', 'search_query': search_query, 'docs': cur_search_docs})

        for idx in range(self.args.max_iter):
            output, output_text = self.generator.generate(messages, self.generator.selfask_stopping_criteria)
            
            if ("So the final answer is:" in output_text):
                text += output_text
                break
            if (output[-1].item() in self.generator.curr_eos) or (idx+1 == self.args.max_iter):
                break # Don't perform another retrieval or prompt construction
        
            intermediate_ans = self.extract_intermediate(output_text)
            search_query = self.extract_follow_up(output_text)
            cur_search_docs = self.retriever.search(search_query) if search_query else []
            tmp_docs = [doc for step in path for doc in step['docs']] + cur_search_docs
            unq_tmp_doc = self.get_unique_docs(tmp_docs)
            
            path.append({
                'think': intermediate_ans,
                'search_query': search_query,
                'docs': cur_search_docs
            })
            
            if idx == 0:
                text += f"Follow up: {search_query}\nIntermediate answer: "
            else:
                text += f"{intermediate_ans}\nFollow up: {search_query}\nIntermediate answer: "
            
            user_input_prompt = self.user_prompt_self_ask.format(
                documents = self.documents2string(unq_tmp_doc),
                question=question
            ) + text
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input_prompt}
            ]
        
        # Regenerate the last sentence if it is needed
        if "So the final answer is:" not in output_text:
            text += f"{output_text}.\nSo the final answer is: "
            
            tmp_docs = [doc for step in path for doc in step['docs']]
            unq_tmp_doc = self.get_unique_docs(tmp_docs)
            user_input_prompt = self.user_prompt_self_ask.format(
                documents = self.documents2string(unq_tmp_doc),
                question=question
            ) + text
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input_prompt}
            ]
            _, output_text = self.generator.generate(messages)
            pred_answer = self.extract_final_answer(output_text) if self.extract_final_answer(output_text) else output_text
            path.append({'think': output_text, 'answer': pred_answer})
            
        else:
            intermediate_ans = self.extract_intermediate(output_text)
            pred_answer = self.extract_final_answer(output_text)
            path.append({'think': intermediate_ans, 'answer': pred_answer})

        return pred_answer, path

    # --
    def get_input_prompt_self_consistency(self, question, trace):
        all_docs = [doc for step in trace[:-1] for doc in step['docs']]
        unq_docs = self.get_unique_docs(all_docs)
        prompt_text = self.user_prompt_self_ask.format(
            documents = self.documents2string(unq_docs),
            question = question
        )
        prompt_text += f"Follow up: {trace[1].get('search_query', '')}\n"
        for step in trace[2:-1]:
            prompt_text += f"Intermediate answer: {step.get('think', '')}\n"
            prompt_text += f"Follow up: {step.get('search_query', '')}\n"
        prompt_text += f"Intermediate answer: {trace[-1].get('think', '')}\n"
        prompt_text += f"So the final answer is: "
        
        return prompt_text

    def partial_inference_self_consistency(self, question, trace):
        prompt_text = self.get_input_prompt_self_consistency(question, trace)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt_text}
        ]
        answer_list = self.generator.generate_batch(
            messages,
            num_return=self.args.n_generations,
            temperature=self.args.consistency_temperature
        )
        return answer_list
    
    def get_input_prompt_reasoning_consistency(self, question, trace):
        all_docs = [doc for step in trace[:-1] for doc in step['docs']]
        unq_docs = self.get_unique_docs(all_docs)
        prompt_text = self.user_prompt_self_ask.format(
            documents = self.documents2string(unq_docs),
            question = question
        )
        prompt_text += f"Follow up: {trace[1].get('search_query', '')}\n"
        for step in trace[2:-1]:
            prompt_text += f"Intermediate answer: {step.get('think', '')}\n"
            prompt_text += f"Follow up: {step.get('search_query', '')}\n"
        prompt_text += f"Intermediate answer: "
        return prompt_text
    
    def partial_inference_reasoning_consistency(self, input_prompt_text):
        messages = [{"role": "user", "content": input_prompt_text}]
        _, output_text = self.generator.generate(messages, temperature=self.args.consistency_temperature)
        intermediate_ans = self.extract_intermediate(output_text) if self.extract_intermediate(output_text) else output_text
        pred_answer = self.extract_final_answer(output_text) if self.extract_final_answer(output_text) else output_text

        if "So the final answer is:" not in output_text:
            prompt_text = input_prompt_text + f"{output_text}. So the final answer is: "
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt_text}
            ]
            _, output_text = self.generator.generate(messages)
            pred_answer = self.extract_final_answer(output_text) if self.extract_final_answer(output_text) else output_text
        
        return intermediate_ans, pred_answer

    def partial_inference_rag_consistency(self, question, generated_trace):
        
        # -- Generated path so far ---
        all_docs = [doc for step in generated_trace for doc in step['docs']]
        unq_docs = self.get_unique_docs(all_docs)
        
        text = f"Follow up: {generated_trace[1].get('search_query', '')}\n"
        for step in generated_trace[2:]:
            text += f"Intermediate answer: {step.get('think', '')}\n"
            text += f"Follow up: {step.get('search_query', '')}\n"
        text += f"Intermediate answer: "
        
        
        user_input_prompt = self.user_prompt_self_ask.format(
            documents = self.documents2string(unq_docs),
            question = question
        ) + text
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input_prompt}
        ]
        
        # print(user_input_prompt)
        # print('-')
        
        # -- Generate the rest -------
        path = []
        for idx in range(self.args.max_iter):
            output, output_text = self.generator.generate(messages, self.generator.selfask_stopping_criteria)
            
            if ("So the final answer is:" in output_text):
                text += output_text
                break
            if (output[-1].item() in self.generator.curr_eos) or (idx+1 == self.args.max_iter):
                break # Don't perform another retrieval or prompt construction
        
            intermediate_ans = self.extract_intermediate(output_text)
            search_query = self.extract_follow_up(output_text)
            cur_search_docs = self.retriever.search(search_query) if search_query else []
            tmp_docs = [doc for step in path for doc in step['docs']] + all_docs + cur_search_docs
            unq_tmp_doc = self.get_unique_docs(tmp_docs)
            
            path.append({
                'think': intermediate_ans,
                'search_query': search_query,
                'docs': cur_search_docs
            })
            
            text += f"{intermediate_ans}\nFollow up: {search_query}\nIntermediate answer: "
            user_input_prompt = self.user_prompt_self_ask.format(
                documents = self.documents2string(unq_tmp_doc),
                question=question
            ) + text
            # print(user_input_prompt)
            # print('-')
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input_prompt}
            ]
        
        if "So the final answer is:" not in output_text:
            text += f"{output_text}\nSo the final answer is: "
            tmp_docs = [doc for step in path for doc in step['docs']] + all_docs
            unq_tmp_doc = self.get_unique_docs(tmp_docs)
            user_input_prompt = self.user_prompt_self_ask.format(
                documents = self.documents2string(unq_tmp_doc),
                question=question
            ) + text
            # print(user_input_prompt)
            # print('-')
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input_prompt}
            ]
            _, output_text = self.generator.generate(messages)
            pred_answer = self.extract_final_answer(output_text) if self.extract_final_answer(output_text) else output_text
            path.append({'think': output_text, 'answer': pred_answer})
        
        else:
            intermediate_ans = self.extract_intermediate(output_text)
            pred_answer = self.extract_final_answer(output_text)
            path.append({'think': intermediate_ans, 'answer': pred_answer})
        
        # print('----')
        
        return pred_answer, path
        

class ReAct_RAG(BasicRAG):
    # Ref: https://github.com/ysymyth/ReAct/blob/master/hotpotqa.ipynb
    def __init__(self, args, device):
        super().__init__(args, device)
        self.instruction = REACT_INSTRUCTION
        
        self.examples = {
            "webthink_simple6": [
                {
                    "question": 'What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?',
                    "steps": [
                        {
                            "Thought": 'I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.',
                            "Action": 'Search[Colorado orogeny]',
                            "Observation": 'The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.'
                        },
                        {
                            "Thought": 'It does not mention the eastern sector. So I need to look up eastern sector.',
                            "Action": 'Lookup[eastern sector]',
                            "Observation": '(Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.',
                        },
                        {
                            "Thought": 'The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.',
                            "Action": 'Search[High Plains]',
                            "Observation": 'High Plains refers to one of two distinct land regions:',
                        },
                        {
                            "Thought": 'I need to instead search High Plains (United States).',
                            "Action": 'Search[High Plains (United States)]',
                            "Observation": 'The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).',
                        },
                        {
                            "Thought": 'High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.',
                            "Action": 'Finish[1,800 to 7,000 ft]',
                        }
                    ]
                },
                {
                    "question": 'Musician and satirist Allie Goertz wrote a song about the \"The Simpsons\" character Milhouse, who Matt Groening named after who?',
                    "steps": [
                        {
                            "Thought": 'The question simplifies to \"The Simpsons\" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.',
                            "Action": 'Search[Milhouse]',
                            "Observation": 'Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.',
                        },
                        {
                            "Thought": 'The paragraph does not tell who Milhouse is named after, maybe I can look up \"named after\".',
                            "Action": 'Lookup[named after]',
                            "Observation": '(Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous.',
                        },
                        {
                            "Thought": 'Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.',
                            "Action": 'Finish[Richard Nixon]',
                        }
                    ]
                },
                {
                    "question": 'Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?',
                    "steps": [
                        {
                            "Thought": 'I need to search Adam Clayton Powell and The Saimaa Gesture, and find which documentary is about Finnish rock groups.',
                            "Action": 'Search[Adam Clayton Powell]',
                            "Observation": "Could not find [Adam Clayton Powell]. Similar: ['Adam Clayton Powell III', 'Seventh Avenue (Manhattan)', 'Adam Clayton Powell Jr. State Office Building', 'Isabel Washington Powell', 'Adam Powell', 'Adam Clayton Powell (film)', 'Giancarlo Esposito'].",
                        },
                        {
                            "Thought": 'To find the documentary, I can search Adam Clayton Powell (film).',
                            "Action": 'Search[Adam Clayton Powell (film)]',
                            "Observation": 'Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg.\nThe film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience.',
                        },
                        {
                            "Thought": 'Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.',
                            "Action": 'Finish[The Saimaa Gesture]',
                        },
                    ]
                    
                },
                {
                    "question": 'What profession does Nicholas Ray and Elia Kazan have in common?',
                    "steps": [
                        {
                            "Thought": 'I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.',
                            "Action": 'Search[Nicholas Ray]',
                            "Observation": 'Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 \u2013 June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.',
                        },
                        {
                            "Thought": 'Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.',
                            "Action": 'Search[Elia Kazan]',
                            "Observation": 'Elia Kazan was an American film and theatre director, producer, screenwriter and actor.',
                        },
                        {
                            "Thought": 'Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.',
                            "Action": 'Finish[director, screenwriter, actor]',
                        },
                    ]
                },
                {
                    "question": "Which magazine was started first Arthur's Magazine or First for Women?",
                    "steps": [
                        {
                            "Thought": "I need to search Arthur's Magazine and First for Women, and find which was started first.",
                            "Action": "Search[Arthur's Magazine]",
                            "Observation": "Arthur's Magazine (1844-\u0080\u00931846) was an American literary periodical published in Philadelphia in the 19th century.",
                        },
                        {
                            "Thought": "Arthur's Magazine was started in 1844. I need to search First for Women next.",
                            "Action": 'Search[First for Women]',
                            "Observation": "First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989.",
                        },
                        {
                            "Thought": "First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.",
                            "Action": "Finish[Arthur's Magazine]",
                        },
                    ]
                },
                {
                    "question": 'Were Pavel Urysohn and Leonid Levin known for the same type of work?',
                    "steps": [
                        {
                            "Thought": 'I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.',
                            "Action": 'Search[Pavel Urysohn]',
                            "Observation": 'Pavel Samuilovich Urysohn (February 3, 1898 \u00e2\u0080\u0093 August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.',
                        },
                        {
                            "Thought": 'Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.',
                            "Action": 'Search[Leonid Levin]',
                            "Observation": 'Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist.',
                        },
                        {
                            "Thought": 'Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work.',
                            "Action": 'Finish[yes]',
                        },
                    ]
                },
            ]
        }
        self.examples_text = ''
        for example in self.examples['webthink_simple6']:
            self.examples_text += f"Question: {example['question']}\n"
            for step_i, think_step in enumerate(example['steps']):
                for step_key, step_val in think_step.items():
                    self.examples_text += f"{step_key} {step_i+1}: {step_val}\n"
            self.examples_text += "\n"
    
    def generate_stopping_criteria(self, sequences):
        return StoppingCriteriaList([StopOnSequence(sequences, self.generator.tokenizer)])

    @staticmethod
    def clean_str(p):
        return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")

    @staticmethod
    def get_page_obs(page):
        # find all paragraphs
        paragraphs = page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # find all sentence
        sentences = []
        for p in paragraphs:
            sentences += p.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        return ' '.join(sentences[:5])

    @staticmethod
    def clean_action(text):
        cleaned_text = re.sub(r'Observation \d+:\s*', '', text)
        return cleaned_text

    # For search action
    def wikipedia_search(self, entity):
        entity_ = entity.replace(" ", "+")
        search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
        response_text = requests.get(search_url).text
        soup = BeautifulSoup(response_text, features="html.parser")
        result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
        if result_divs:  # mismatch
            result_titles = [self.clean_str(div.get_text().strip()) for div in result_divs]
            obs = f"Could not find {entity}. Similar: {result_titles[:5]}."
        else:
            page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
            if any("may refer to:" in p for p in page):
                self.search_step("[" + entity + "]")
            else:
                self.page = ""
                for p in page:
                    if len(p.split(" ")) > 2:
                        self.page += self.clean_str(p)
                        if not p.endswith("\n"):
                            self.page += "\n"
                obs = self.get_page_obs(self.page)
                self.lookup_keyword = self.lookup_list = self.lookup_cnt = None
        return obs

    def retriever_search(self, search_query):
        search_docs = self.retriever.search(search_query)
        
        self.page = ""
        for doc in search_docs:
            content = doc['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            self.page += text
        
        self.lookup_keyword = self.lookup_list = self.lookup_cnt = None
        
        return search_docs

    # For lookup action
    def construct_lookup_list(self, keyword):
        # find all paragraphs
        if self.page is None:
            return []
        paragraphs = self.page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # find all sentence
        sentences = []
        for p in paragraphs:
            sentences += p.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]

        parts = sentences
        parts = [p for p in parts if keyword.lower() in p.lower()]
        return parts

    def get_observation(self, action):
        done = False
        answer = None
        action = action.strip().lower()
        if action.startswith("search[") and action.endswith("]"):
            entity = action[len("search["):-1]
            # obs = self.wikipedia_search(entity) # Search in wikipedia
            obs = passages2string(self.retriever_search(entity)) # Search with retriever
        
        elif action.startswith("lookup[") and action.endswith("]"):
            keyword = action[len("lookup["):-1]
            if self.lookup_keyword != keyword:  # reset lookup
                self.lookup_keyword = keyword
                self.lookup_list = self.construct_lookup_list(keyword)
                self.lookup_cnt = 0
            if self.lookup_cnt >= len(self.lookup_list):
                obs = "No more results.\n"
            else:
                obs = f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) " + self.lookup_list[self.lookup_cnt]
                self.lookup_cnt += 1

        elif action.startswith("finish[") and action.endswith("]"):
            answer = action[len("finish["):-1]
            obs = f"Episode finished\n"
            done = True

        elif action.startswith("think[") and action.endswith("]"):
            obs = "Nice thought."
        else:
            obs = "Invalid action: {}".format(action)

        return obs, done, answer

    def inference(self, question):
        self.page = None  # current Wikipedia page
        self.lookup_keyword = None  # current lookup keyword
        self.lookup_list = None  # list of paragraphs containing current lookup keyword
        self.lookup_cnt = None  # current lookup index
        
        search_query = question
        instruct_exps = self.instruction + self.examples_text 
        input_prompt = instruct_exps + f"Question: {search_query}\n"
        path = []
        for iter_num in range(1, self.args.max_iter+1):
            # Get Thought & Action
            messages = [
                {"role": "system", "content": ''},
                {"role": "user", "content": input_prompt + f"Thought {iter_num}:"}
            ]
            _, thought_action = self.generator.generate(
                messages,
                self.generate_stopping_criteria([f"\n\nObservation {iter_num}:", f"\nObservation {iter_num}:", f"Observation {iter_num}:", f"Observation {iter_num}: ", f"\nObservation {iter_num}: "])
            )
            try:
                thought, action = thought_action.strip().split(f"\nAction {iter_num}: ")
            except:
                # print('ohh...', thought_action)
                thought = thought_action.strip().split('\n')[0]
                messages = [
                    {"role": "system", "content": ''},
                    {"role": "user", "content": input_prompt + f"Thought {iter_num}: {thought}\nAction {iter_num}:"}
                ]
                _, action = self.generator.generate(messages, self.generate_stopping_criteria(["\n", "\n\n", " \n", " \n\n"]))
            
            thought = thought.replace('\n', '')
            action = self.clean_action(action).replace('\n', '')
            
            # Get Observation
            obs, done, pred_answer = self.get_observation(action)
            obs = obs.replace('\\n', '')
            
            path.append({'Thought': thought, 'Action': action, 'Observation': obs})
            step_str = f"Thought {iter_num}: {thought}\nAction {iter_num}: {action}\nObservation {iter_num}: {obs}\n"
            input_prompt += step_str
            
            if done:
                break
        
        # Regenerate the last sentence if it is needed
        action = action.strip().lower()
        if not (action.startswith("finish[") and action.endswith("]")):
            messages = [
                {"role": "system", "content": ''},
                {"role": "user", "content": input_prompt + f"Action {iter_num+1}: Finish["}
            ]
            _, output_text = self.generator.generate(
                messages,
                self.generate_stopping_criteria(["]", "] ", ']\n', ']\n ', ']\n\n', ']\n\n ']),
                max_new_tokens=32
            )
            pred_answer = output_text[:-1]
            path.append({'Action': pred_answer})
        
        return pred_answer, path

class SearchO1_RAG(BasicRAG):
    def __init__(self, args, device):
        super().__init__(args, device)
        
        # Define special tokens
        self.BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
        self.END_SEARCH_QUERY = "<|end_search_query|>"
        self.BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
        self.END_SEARCH_RESULT = "<|end_search_result|>"

        self.MAX_SEARCH_LIMIT = 5
        if args.dataset in ['hotpotqa', 'musique', '2wikimultihopqa', 'bamboogle']:
            self.MAX_SEARCH_LIMIT = 10
            self.MAX_TURN = 15

        if args.dataset in ['nq', 'triviaqa']:
            self.instruction = get_singleqa_search_o1_instruction(self.MAX_SEARCH_LIMIT)
        elif args.dataset in ['hotpotqa', 'musique', '2wikimultihopqa', 'bamboogle']:
            self.instruction = get_multiqa_search_o1_instruction(self.MAX_SEARCH_LIMIT)

    def inference(self, question):
        user_prompt = get_task_instruction_openqa(question)
        prompt = [{"role": "user", "content": self.instruction + user_prompt}]
        # TODO
    
class SearchR1_RAG(BasicRAG):
    def __init__(self, generation_model, generation_tokenizer, device, args):
        super().__init__(generation_model, generation_tokenizer, device, args)
        self.curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'
        self.answer_template = '<answer>{answer}</answer>'
        self.prompt = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    
    def get_think(self, text):
        pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            # return matches[-1]
            return matches[0]
        else:
            return None

    def get_partial_think(self, text):
        pattern = re.compile(r"(.*?)</think>", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            # return matches[-1]
            return matches[0]
        else:
            return None

    def get_query(self, text):
        pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            # return matches[-1]
            return matches[0]
        else:
            return None

    def get_answer(self, text):
        pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            # return matches[-1]
            return matches[0]
        else:
            return None
    
    def get_partial_answer(self, text):
        pattern = re.compile(r"(.*?)</answer>", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            # return matches[-1]
            return matches[0]
        else:
            return None

    def inference(self, question):
        input_prompt = self.prompt.format(question=question)
        messages = [{"role": "user", "content": input_prompt}]
        
        path, cnt = [], 0
        while True:
            output_, output_text = self.generator.generate(messages, self.generator.searchr1_stopping_criteria)
    
            if output_[-1].item() in self.generator.curr_eos:
                break
        
            tmp_query = self.get_query(output_text)
            if tmp_query:
                search_docs = self.retriever.search(tmp_query)
                search_results = passages2string(search_docs)
            else:
                search_docs, search_results = [], ''

            path.append({
                'think': self.get_think(output_text),
                'search_query': tmp_query,
                'docs': search_docs
            })
            search_text = self.curr_search_template.format(output_text=output_text, search_results=search_results)
            input_prompt += search_text
            messages = [{"role": "user", "content": input_prompt}]
            cnt += 1

        one_step_think = self.get_think(output_text)
        pred_answer = self.get_answer(output_text)
        path.append({'think': one_step_think, 'answer': pred_answer})
            
        return pred_answer, path
    
    # --
    def get_input_prompt_self_consistency(self, question, trace):
        prompt_text = self.prompt.format(question=question)
        for step in trace[:-1]:
            prompt_text += f"<think> {step['think']} </think>\n"
            prompt_text += f"<search> {step['search_query']} </search>\n"
            prompt_text += f"<information> {passages2string(step['docs'])} </information>\n\n"
        prompt_text += f"<think> {trace[-1]['think']} </think>\n"
        # prompt_text += f"<answer> "
        
        return prompt_text
    
    def partial_inference_self_consistency(self, question, trace):
        answer_list = []
        input_prompt_text = self.get_input_prompt_self_consistency(question, trace)
        messages = [{"role": "user", "content": input_prompt_text}]
        for i in range(self.args.n_generations):
            output_, output_text = self.generator.generate(
                messages,
                self.generator.searchr1_answer_stopping_criteria,
                temperature=self.args.consistency_temperature
            )
            answer_ = self.get_answer(output_text)
            answer = answer_.strip() if answer_ else ''
            answer_list.append(answer)
        
        return answer_list
    
    def get_input_prompt_reasoning_consistency(self, question, trace):
        prompt_text = self.prompt.format(question=question)
        for step in trace[:-1]:
            prompt_text += f"<think> {step['think']} </think>\n"
            prompt_text += f"<search> {step['search_query']} </search>\n"
            prompt_text += f"<information> {passages2string(step['docs'])} </information>\n\n"
        # prompt_text += f"<think> {trace[-1]['think']} "
        return prompt_text 
        
    def partial_inference_reasoning_consistency(self, input_prompt_text):
        messages = [{"role": "user", "content": input_prompt_text}]
        _, output_text = self.generator.generate(
            messages,
            self.generator.searchr1_answer_stopping_criteria,
            temperature=self.args.consistency_temperature
        )
        answer_ = self.get_answer(output_text)
        answer = answer_.strip() if answer_ else ''
        think_ = self.get_think(output_text)
        think = think_.strip() if think_ else ''
        return think, answer
        
    def partial_inference_rag_consistency(self, question, generated_trace):
        input_prompt = self.prompt.format(question=question)
        generated_trace_text = ''.join(
            self.curr_search_template.format(
                output_text=f"<think> {step['think']} </think>\n<search> {step['search_query']} </search>\n",
                search_results=passages2string(step['docs'])
            ) for step in generated_trace
        )
        input_prompt += generated_trace_text
        messages = [{"role": "user", "content": input_prompt}]
        
        path, cnt = [], 0
        while True:
            output_, output_text = self.generator.generate(messages, self.generator.searchr1_stopping_criteria)
            if output_[-1].item() in self.generator.curr_eos:
                break

            tmp_query = self.get_query(output_text)
            if tmp_query:
                search_docs = self.retriever.search(tmp_query)
                search_results = passages2string(search_docs)
            else:
                search_docs, search_results = [], ''

            path.append({
                'think': self.get_think(output_text),
                'search_query': tmp_query,
                'docs': search_docs
            })
            search_text = self.curr_search_template.format(output_text=output_text, search_results=search_results)
            input_prompt += search_text
            messages = [{"role": "user", "content": input_prompt}]
            cnt += 1

        one_step_think = self.get_think(output_text)
        pred_answer = self.get_answer(output_text)
        path.append({'think': one_step_think, 'answer': pred_answer})
            
        return pred_answer, path
    
        






# intermediate_ans = self.extract_intermediate(output_text)
# path.append({'search_query': search_query, 'docs': cur_search_docs, 'think': intermediate_ans})
# if intermediate_ans:
#     text += f"{intermediate_ans}\n"        
    
   
# followup_query = output_text.split("Intermediate answer:")[0]
# text += followup_query
# stop_condition = "Follow up:"
# if stop_condition == "Intermediate answer:":
# elif stop_condition == "Follow up:":
#     followup_split = re.split(self.FOLLOW_UP_PATTERN, output_text)
#     intermediate_answer = followup_split[0]
#     text += intermediate_answer

#     if len(followup_split) > 1:
#         text += re.findall(self.FOLLOW_UP_PATTERN, output_text)[0]
#     stop_condition = "Intermediate answer:"
# if text[-1] == "\n":
#     text = text[:-1]

# if "Follow up: " in output_text:
#     # get the first follow up
#     search_query = [l for l in output_text.split("\n") if "Follow up: " in l][0].split("Follow up: ")[-1]
#     search_docs = self.retriever.search(search_query)
            
# Regenerate the last sentence if it is needed
# if "So the answer is:" not in text:
#     docs_tmp = path[-1]['docs']
#     output_text = self.regenerate(self.system_prompt, question, docs_tmp, path)
#     path.append({'think': f'So the answer is: {output_text}'})
#     pred_answer = get_answer(output_text) if get_answer(output_text) else output_text       
# else:
#     pred_answer = get_answer(text)

# return pred_answer, path







class FLARE_RAG_V2(BasicRAG):
    # Base on FlashRAG git code
    def __init__(self, args, device, look_ahead_steps=64):
        super().__init__(args, device)
        self.system_prompt = SYSTEM_PROMPT_COT
        self.max_generation_length = args
        self.look_ahead_steps = look_ahead_steps

    def get_next_sentence(self, output, scores):
        tokenizer = self.generator.tokenizer
        text_sentences = re.split(r"(?<=[^A-Z].[.?]) +", output)
        if isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
            token_id_sentences = [tokenizer.encode(s, add_special_tokens=False) for s in text_sentences]
        else:
            token_id_sentences = [tokenizer.encode(s, allowed_special="all") for s in text_sentences]

        output_ids = tokenizer.encode(output, add_special_tokens=False)
        first_sent_ids = token_id_sentences[0]
        first_sent_score = scores[: len(first_sent_ids)]

        return text_sentences[0], first_sent_score

    def judge_sent_confidence(self, sent, sent_score):
        judge_result = all([score > self.args.hallucination_threshold for score in sent_score])
        new_query = None
        if not judge_result:
            tokenizer = self.generator.tokenizer
            if isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
                sent_ids = tokenizer.encode(sent, add_special_tokens=False)
            else:
                sent_ids = tokenizer.encode(sent, allowed_special="all")
            # assert len(sent_ids) == len(sent_score)
            new_query_ids = [i for i, score in zip(sent_ids, sent_score) if score > self.args.hallucination_threshold]
            new_query = tokenizer.decode(new_query_ids)
            if len(new_query) == 0:
                judge_result = True
        return judge_result, new_query

    def inference(self, question):
        path, text = [], ""
        gen_length, iter_round = 0, 0
        while gen_length < self.args.max_new_tokens and iter_round < self.args.max_iter:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt_wo_context.format(
                    examples=self.cot_examples_text,
                    question=question
                )},
                {"role": "assistant", "content": text}
            ]
            output_text, scores = self.generator.generate_with_scores(
                messages,
                stopping_criteria=self.generator.flare_stopping_criteria,
                max_new_tokens=self.look_ahead_steps
            )
            # scores: token logits of the whole generation seq
            next_sent, next_sent_score = self.get_next_sentence(output_text, scores)
            judge_result, query = self.judge_sent_confidence(next_sent, next_sent_score)
            
            # print(output_text)
            # print(next_sent)
            # print(next_sent_score)
            # print(query)
            # print('----')
            
            if not judge_result:
                # do retrieval-augmented generation
                search_docs = self.retriever.search(query)
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.user_prompt_with_context.format(
                        examples=self.cot_examples_text,
                        documents=passages2string(search_docs),
                        question=question
                    )},
                    {"role": "assistant", "content": text}
                ]
                output, scores = self.generator.generate_with_scores(
                    messages,
                    stopping_criteria=self.generator.flare_stopping_criteria,
                    max_new_tokens=self.look_ahead_steps
                )
                next_sent, _ = self.get_next_sentence(output, scores)
                path.append({'search_query': query, 'docs': search_docs, 'think': next_sent})
                
            else:
                path.append({'search_query': '', 'docs': [], 'think': next_sent})

            text += f" {next_sent}"
            gen_length += len(next_sent_score)
            iter_round += 1
        
        # Regenerate the last sentence if it is needed
        if "So the answer is:" not in text:
            docs_tmp = [doc for step in path for doc in step['docs']]
            output_text = self.regenerate(self.system_prompt, question, docs_tmp, path)
            path.append({'think': f'So the answer is: {output_text}'})
            pred_answer = get_answer(output_text) if get_answer(output_text) else output_text       
        else:
            pred_answer = get_answer(text)
        
        return pred_answer, path, None

class FLARE_RAG_OLD(BasicRAG):
    def inference(self, question):
        # ----------------------------
        gen_count, ret_count, sent_count, token_count = 0,0,0,0
        
        num_hallucination = 0
        text = ""
        generation_path = []
        
        
        while True:
            old_len = len(text)
            prompt = self.generator.format_longform(question, self.fewshot_examplers, [])
            new_text, tokens_text, logprobs = self.generator.generate(
                prompt,
                self.args.max_new_tokens,
                system_prompt=SYSTEM_PROMPT_LONGFORM,
                return_logprobs=True
            )
            if self.args.use_counter:
                sent_count_cur, token_count_cur = self.counter.add_generate(new_text, self.generator.tokenizer)
                gen_count += 1
                sent_count += sent_count_cur
                token_count += token_count_cur
            
            ptext, curr, hallucination = self.modifier(new_text, tokens_text, logprobs)
            generation_path.append({
                'perv_text': ptext,
                'curr_sent': curr,
                'hallucination': hallucination
            })
            
            if not hallucination:
                text = text.strip() + " " + new_text.strip()
            else:
                num_hallucination += 1
                if self.args.query_formulation == "direct":
                    retrieve_question = curr.replace("[xxx]", "")
                elif self.args.query_formulation == "forward_all":
                    tmp_all = [question, text, ptext]
                    retrieve_question = " ".join(s for s in tmp_all if len(s) > 0)
                else:
                    raise NotImplemented

                # docs, _, _ = self.retrieve([retrieve_question], [qid], [pos_contexts], [neg_contexts], topk=self.args.retrieve_topk)
                retrieved_docs = self.retriever.search(retrieve_question)
                if self.args.use_counter:
                    self.counter.retrieve += 1
                    ret_count += 1
                
                prompt = self.generator.format_longform(question, self.fewshot_examplers, retrieved_docs)
                prompt += " " + text + " " + ptext.strip()
                new_text, _, _ = self.generator.generate(
                    prompt,
                    self.args.max_new_token,
                    system_prompt=SYSTEM_PROMPT_LONGFORM
                )
                if self.args.use_counter:
                    sent_count_cur, token_count_cur = self.counter.add_generate(new_text, self.generator.tokenizer)
                    gen_count += 1
                    sent_count += sent_count_cur
                    token_count += token_count_cur
                    self.counter.hallucinated += 1
                text = text.strip() + " " + ptext.strip() + " " + new_text.strip()
        
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.args.max_new_token or len(text) <= old_len or "the answer is" in text:
                break
        
        return text, num_hallucination, generation_path, (gen_count, ret_count, sent_count, token_count)

class DRAGIN_RAG_OLD(BasicRAG):
    def inference(self, question):
        # --------------------
        gen_count, ret_count, sent_count, token_count = 0,0,0,0
        num_hallucination = 0
        generation_path = []
        case = f"Question: {question}\nAnswer:"
        text = ""

        path, text = [], ""
        while True:
            old_len = len(text)
            prompt = self.generator.format_longform(question, self.fewshot_examplers, [], add_case=False)
            tmp_li = [case, text]
            prompt += " ".join(s for s in tmp_li if len(s) > 0)
            
            new_text, tokens, attns, logprobs, entropies = self.generator.generate_attn(
                prompt, self.args.max_new_token,
                system_prompt=SYSTEM_PROMPT_LONGFORM,
                use_entropy = self.args.rag_method == "dragin",
            )
            if self.args.use_counter:
                sent_count_cur, token_count_cur = self.counter.add_generate(new_text, self.generator.tokenizer)
                gen_count += 1
                sent_count += sent_count_cur
                token_count += token_count_cur
                
            weight = entropies if self.args.rag_method == "dragin" else [-v for v in logprobs]
            hallucination, ptext, curr_tokens, curr_hit =  self.modifier(new_text, tokens, attns, weight)
            generation_path.append({
                'perv_text': ptext,
                'curr_tokens': curr_tokens,
                'curr_thre': curr_hit,
                'hallucination': hallucination
            })
            
            if not hallucination:
                text = text.strip() + " " + new_text.strip()
            else:
                num_hallucination += 1
                forward_all = [question, text, ptext]
                forward_all = " ".join(s for s in forward_all if len(s) > 0)
            
                if self.args.query_formulation == "current":
                    retrieve_question = " ".join(curr_tokens)
                elif self.args.query_formulation == "current_wo_wrong":
                    retrieve_question = " ".join(list(curr_tokens[i] if curr_hit[i] == 0 else "" for i in range(len(curr_tokens))))
                elif self.args.query_formulation == "forward_all":
                    retrieve_question = forward_all
                elif self.args.query_formulation == "last_sentence":
                    retrieve_question = self.get_last_sentence(forward_all)
                elif self.args.query_formulation == "last_n_tokens":
                    assert "retrieve_keep_top_k" in self.args
                    retrieve_question = self.fetch_last_n_tokens(forward_all, self.args.retrieve_keep_top_k, self.generator.tokenizer)
                elif self.args.query_formulation == "real_words": 
                    retrieve_question = self.keep_real_words(
                        prev_text = question + " " + text + " " + ptext, 
                        curr_tokens = curr_tokens, 
                        curr_hit = curr_hit,
                    ) 
                else:
                    raise NotImplemented
                
                generation_path[-1]['retrieve_question'] = retrieve_question
                
                retrieved_docs = self.retriever.search(retrieve_question)
                if self.args.use_counter == True:
                    self.counter.retrieve += 1
                    ret_count += 1
                # docs, _, _ = self.retrieve([retrieve_question], [qid], [pos_contexts], [neg_contexts], topk=self.args.retrieve_topk)
                prompt = self.generator.format_longform(question, self.fewshot_examplers, retrieved_docs, add_case=False)
                tmp_li = [case, text, ptext.strip()]
                prompt += " ".join(s for s in tmp_li if len(s) > 0)
                new_text, _, _ = self.generator.generate(
                    prompt,
                    self.args.max_new_token,
                    system_prompt=SYSTEM_PROMPT_LONGFORM
                )
                if self.args.use_counter:
                    sent_count_cur, token_count_cur = self.counter.add_generate(new_text, self.generator.tokenizer)
                    gen_count += 1
                    sent_count += sent_count_cur
                    token_count += token_count_cur
                    self.counter.hallucinated += 1
                text = text.strip() + " " + ptext.strip() + " " + new_text.strip()
            
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.args.max_new_token or len(text) <= old_len or "the answer is" in text:
                break

        return text, num_hallucination, generation_path, (gen_count, ret_count, sent_count, token_count)
    





























# gen_count, ret_count, sent_count, token_count = 0,0,0,0
# text = ""
# retrieve_question = question

# while True:
#     old_len = len(text)
#     # docs, _, _ = self.retrieve([retrieve_question], [qid], [pos_contexts], [neg_contexts], topk=self.args.retrieve_topk)
#     retrieved_docs = self.retriever.search(retrieve_question)
#     if self.args.use_counter:
#         self.counter.retrieve += 1
#         ret_count += 1
#     prompt = self.generator.format_longform(question, self.fewshot_examplers, retrieved_docs)
#     prompt += text
    
#     if self.args.rag_method == "fix_length_retrieval":
#         new_text, _, _ = self.generator.generate(prompt, self.args.generate_fix_length, system_prompt=SYSTEM_PROMPT_LONGFORM)
#         if self.args.use_counter:
#             sent_count_cur, token_count_cur = self.counter.add_generate(new_text, self.generator.tokenizer)
#             gen_count += 1
#             sent_count += sent_count_cur
#             token_count += token_count_cur
            
#         text = text.strip() + " " + new_text.strip()
#         retrieve_question = new_text.strip()
        
#     elif self.args.rag_method == "fix_sentence_retrieval":
#         new_text, _, _ = self.generator.generate(prompt, self.args.max_new_token, system_prompt=SYSTEM_PROMPT_LONGFORM)
#         if self.args.use_counter:
#             sent_count_cur, token_count_cur = self.counter.add_generate(new_text, self.generator.tokenizer)
#             gen_count += 1
#             sent_count += sent_count_cur
#             token_count += token_count_cur
            
#         new_text = new_text.strip()
#         sentences = list(nlp(new_text).sents)
#         sentences = [str(sent).strip() for sent in sentences]
#         if len(sentences) == 0:
#             break
#         text = text.strip() + " " + str(sentences[0])
#         retrieve_question = sentences[0]

#     else:
#         raise NotImplementedError

#     tokens_count = len(self.generator.tokenizer.encode(text))
#     if tokens_count > self.args.max_new_token or len(text) <= old_len or "the answer is" in text:
#         break

# return text, None, None, (gen_count, ret_count, sent_count, token_count)



# # === Fewshot examples ========================
# examplers_ = getattr(examplers, f'hotpotqa_exps')
# self.fewshot_examplers = (
#     random.sample(examplers_, args.fewshot)
#     if len(examplers_) >= args.fewshot
#     else examplers_
# )
# retrieved_docs = self.retriever.search(question)
# if self.args.use_counter == True:
#     self.counter.retrieve += 1

# prompt = self.generator.format_longform(question, self.fewshot_examplers, retrieved_docs)
# text, _, _ = self.generator.generate(
#     prompt, self.args.max_new_token,
#     system_prompt=SYSTEM_PROMPT_LONGFORM
# )
# if self.args.use_counter:
#     sent_count, token_count = self.counter.add_generate(text, self.generator.tokenizer)
# else:
#     sent_count, token_count = 0, 0
    
# return text, None, None, (1, 1, sent_count, token_count)


# tmp = [doc for step in path for doc in step['docs']]
# messages = [
#     {"role": "system", "content": self.system_prompt},
#     {"role": "user", "content": self.user_prompt_with_context.format(documents=passages2string(tmp), question=question)},
#     {"role": "assistant", "content": ' '.join([step['think'] for step in path]) + " So the answer is:"}
# ]               
# output, output_text = self.generator.generate(messages, self.generator.ircot_stopping_criteria)
# pred_answer = output_text
# path.append({'think': f'So the answer is: {output_text}'})

# try:
#     examplers_ = getattr(examplers, f'{args.dataset}_exps')
# except AttributeError:
#     raise ValueError(f"The dataset '{args.dataset}' does not exist in the 'examplers' module.")


# def reinference(self, question, pred, max_new_tokens=20):
#     user_prompt = self.generator.format_longform(question, self.fewshot_examplers, [])
#     user_prompt += f" {pred}"
#     text, _, _ = self.generator.generate(
#         user_prompt, max_new_tokens,
#         system_prompt=SYSTEM_PROMPT_REGENERATE
#     )
#     return text

# prompt = self.generator.format_longform(question, self.fewshot_examplers, [])
# text, _, _ = self.generator.generate(
#     prompt, self.args.max_new_token,
#     system_prompt=SYSTEM_PROMPT_LONGFORM
# )
# if self.args.use_counter:
#     sent_count, token_count = self.counter.add_generate(text, self.generator.tokenizer)
# else:
#     sent_count, token_count = 0, 0

# (ret, get, sent, token)
# return text, None, None, (0, 1, sent_count, token_count)


# # Initial retrieval
# search_query = question
# cur_search_docs = self.retriever.search(search_query)
# messages = [
#     {"role": "system", "content": self.system_prompt},
#     {"role": "user", "content": self.user_prompt_with_context.format(
#         examples=self.cot_examples_text,
#         documents=passages2string(cur_search_docs),
#         question=question
#     )}
# ]

# path, iter_num = [], 0
# while True:
#     output, output_text = self.generator.generate(messages, self.generator.ircot_stopping_criteria)
#     path.append({'search_query': search_query, 'docs': cur_search_docs, 'think': output_text})
#     pred_answer = get_answer(output_text)
    
#     if "So the answer is:" in output_text:
#         path.append({'think': output_text})
#         break
#     iter_num += 1
#     if (output[-1].item() in self.generator.ircot_curr_eos) or (iter_num == self.args.max_iter):
#         break  # Don't perform another retrieval or prompt construction
    
#     search_query = output_text
#     cur_search_docs = self.retriever.search(search_query) if search_query else []
#     messages = [
#         {"role": "system", "content": self.system_prompt},
#         {"role": "user", "content": self.user_prompt_with_context.format(
#             examples=self.cot_examples_text,
#             documents=passages2string(cur_search_docs),
#             question=question
#         )},
#         {"role": "assistant", "content": ' '.join([step['think'] for step in path])}
#     ]