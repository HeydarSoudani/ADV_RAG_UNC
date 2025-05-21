
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import spacy
import torch
import random
import numpy as np
from math import exp
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from run_rag_methods.src.generators import LLMGenerator
from run_rag_methods.src.retrievers_local import BM25Retriever, ContrieverRetriever, RerankRetriever, DenseRetriever
from run_rag_methods.src.templetes import (
    SYSTEM_PROMPT_DIRECT, SYSTEM_PROMPT_COT, SYSTEM_PROMPT_IRCOT,
    SYSTEM_PROMPT_DRAGIN,
    SELF_ASK_PROMPT_SINGLE_HOP, SELF_ASK_PROMPT_MULTI_HOP
)

nlp = spacy.load("en_core_web_sm")


def passages2string(retrieval_result):
    format_reference = ''
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item['contents']
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        # format_reference += f"Wikipedia Title: {title}\n{text}\n\n"
        format_reference += f"Doc {idx+1} (Title: {title}) {text}\n"
    return format_reference

def get_answer(text):
    parts = text.split("the answer is: ", 1)  # Split at the first occurrence
    if len(parts) <= 1:
        return None
    pred = parts[1].strip() if len(parts) > 1 else ""
    pattern = r"\.?</s>"
    pred = re.sub(pattern, "", pred)
    pred = pred.rstrip(".?!")
    return pred

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
        
        return len(sentences), len(ids)

    def calc(self, other_counter):
        return {
            "retrieve_count": self.retrieve - other_counter.retrieve, 
            "generate_count": self.generate - other_counter.generate,
            "hallucinated_count": self.hallucinated - other_counter.hallucinated, 
            "token_count": self.token - other_counter.token, 
            "sentence_count": self.sentence - other_counter.sentence 
        }

class BasicRAG:
    def __init__(self, args, device):
        self.args = args
        self.counter = Counter()
        self.generator = LLMGenerator(args, device)
        
        # === Retrievers ============================= 
        if args.retriever_name == 'bm25':
            self.retriever = BM25Retriever(args)  
        elif args.retriever_name == 'contriever':
            self.retriever = ContrieverRetriever(args)
        elif args.retriever_name in ['rerank_l6', 'rerank_l12']:
            self.retriever = RerankRetriever(args)
        elif args.retriever_name in ['e5', 'bge']:
            self.retriever = DenseRetriever(args)
                
        # === few-shot examples =======================
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
        self.user_prompt_self_ask = "{documents}Quesiton: {question}\nAre follow up questions needed here: "

    def get_top_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[0] if len(sentences) > 0 else ""

    def get_last_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[-1] if len(sentences) > 0 else "" 
       
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
    def __init__(self, args, device):
        super().__init__(args, device)
        self.system_prompt = SYSTEM_PROMPT_DIRECT
    
    def inference(self, question):
        path = []
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt_wo_context.format(examples=self.direct_examples_text, question=question)}
        ]
        _, output_text = self.generator.generate(messages)
        path.append({'search_query': question, 'docs': [], 'think': output_text})
        pred_answer = output_text
        
        return pred_answer, path

class CoTInference(BasicRAG):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.system_prompt = SYSTEM_PROMPT_COT
    
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
        path.append({'search_query': question, 'docs': [], 'think': output_text})
        
        # Regenerate the last sentence if it is needed
        if "So the answer is:" not in output_text:
            output_text = self.regenerate(self.system_prompt, question, [], path, with_context=False)
            path.append({'think': f'So the answer is: {output_text}'})
            pred_answer = get_answer(output_text) if get_answer(output_text) else output_text
        else:
            pred_answer = get_answer(output_text)
        
        return pred_answer, path
  
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

# Base on DRAGIN git code
class FLARE_RAG_V1(BasicRAG):
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
    def __init__(self, args, device):
        super().__init__(args, device)
        self.single_hop = False
        self.system_prompt = SELF_ASK_PROMPT_MULTI_HOP
        self.FOLLOW_UP_PATTERN = r"Follow up:.*\n"
    
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
        match = re.search(r'(.*?)\nFollow up:', text, re.DOTALL)
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
        
        follow_ups = "No." if self.single_hop else "Yes."
        path, text = [], ""
        
        # Initial retrieval
        search_query = question
        search_docs = self.retriever.search(search_query)
        path.append({'search_query': search_query, 'docs': search_docs, 'think': None})
        
        for idx in range(self.args.max_iter):
            input_prompt = self.user_prompt_self_ask.format(
                documents = self.documents2string(search_docs),
                question=question
            )
            input_prompt += f"{follow_ups}\n{text}"
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": input_prompt}
            ]
            # print(self.system_prompt)
            # print(input_prompt)
            # print('-')
            _, output_text = self.generator.generate(messages, self.generator.selfask_stopping_criteria)
            # print(output_text)
            # print('-----')
            intermediate_ans = self.extract_intermediate(output_text)
            path.append({'search_query': search_query, 'docs': search_docs, 'think': intermediate_ans})
                
            if ("So the final answer is:" in output_text) or (idx+1 == self.args.max_iter):
                # print('a')
                text += output_text
                break
        
            if intermediate_ans:
                text += f"{intermediate_ans}\n"
        
            search_query = self.extract_follow_up(output_text)
            if search_query:
                text += f"Follow up: {search_query}\nIntermediate answer: "
                search_docs = self.retriever.search(search_query)
                
        # Regenerate the last sentence if it is needed
        if "So the final answer is:" not in text:
            text += "\nSo the final answer is:"
            docs_tmp = path[-1]['docs']
            input_prompt = self.user_prompt_self_ask.format(
                documents = self.documents2string(docs_tmp),
                question=question
            )
            input_prompt += f"{follow_ups}\n{text}"
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": input_prompt}
            ]
            _, output_text = self.generator.generate(messages)
            pred_answer = self.extract_final_answer(output_text) if self.extract_final_answer(output_text) else output_text       
            path.append({'think': f'So the final answer is: {output_text}'})
        else:
            pred_answer = self.extract_final_answer(output_text)
            # print('b')

        # print(pred_answer)
    
        return pred_answer, path
   
   
class SearchR1_RAG(BasicRAG): 
    def __init__(self, args, device):
        super().__init__(args, device)
        self.curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'
        self.prompt = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    
    
    def get_think(self, text):
        pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            return matches[-1]
        else:
            return None

    def get_query(self, text):
        pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            return matches[-1]
        else:
            return None

    def get_answer(self, text):
        pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            return matches[-1]
        else:
            return None

    def inference(self, question):
        input_prompt = self.prompt.format(question=question)
        messages = [{"role": "user", "content": input_prompt}]
        
        path, cnt = [], 0
        while True:
            output_, output_text = self.generator.generate(messages)
            
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
            cnt += 1

        one_step_think = self.get_think(output_text)
        pred_answer = self.get_answer(output_text)
        path.append({'think': one_step_think, 'answer': pred_answer})
            
        return pred_answer, path
        
        
        
        
        
        
        
        
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















# Base on FlashRAG git code
class FLARE_RAG_V2(BasicRAG):
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