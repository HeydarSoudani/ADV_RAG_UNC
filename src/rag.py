
import re
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import spacy
import torch
import numpy as np
from math import exp

# from src.generate_dragin import BasicGenerator
from src.generate_chat_template import BasicGenerator
from src.retrieve import BM25, Rerank, PositiveRet, NegativeRet
from src.templetes import SYSTEM_PROMPT_LONGFORM, SYSTEM_PROMPT_REGENERATE, SYSTEM_PROMPT_SHORTFORM

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


class BasicRAG:
    def __init__(self, args):
        # args = args.__dict__ 
        self.args = args
        self.counter = Counter()
        self.generator = BasicGenerator(args)
        
        if args.retriever_model == 'bm25':
            self.retriever = BM25(args)
        elif args.retriever_model == 'rerank':
            self.retriever = Rerank(args)
        elif args.retriever_model == 'positive':
            self.retriever = PositiveRet(args)
        elif args.retriever_model == 'negative':
            self.retriever = NegativeRet(args)
        
    def retrieve(self, queries, qids, pos_contexts, neg_contexts, topk=1):
        self.counter.retrieve += 1
        docs, docids, scores = self.retriever.retrieve(
            queries=queries, qids=qids,
            pos_contexts=pos_contexts, neg_contexts=neg_contexts,
            topk=topk
        )
        return docs, docids, scores
        
    def get_top_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[0] if len(sentences) > 0 else ""

    def get_last_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[-1] if len(sentences) > 0 else "" 
       
    def reinference(self, question, fewshot_examplers, pred, max_new_tokens=20):
        user_prompt = self.generator.format_longform(question, fewshot_examplers, [])
        user_prompt += f" {pred}"
        text, _, _ = self.generator.generate(
            user_prompt, max_new_tokens,
            system_prompt=SYSTEM_PROMPT_REGENERATE
        )
        return text
            
    
class NoRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def inference(self, question, qid, fewshot_examplers, pos_contexts, neg_contexts):
        prompt = self.generator.format_longform(question, fewshot_examplers, [])
        text, _, _ = self.generator.generate(
            prompt, self.args.generate_max_length,
            system_prompt=SYSTEM_PROMPT_LONGFORM
        )
        if self.args.use_counter:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text, None, None
    
    
class SingleRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def inference(self, question, qid, fewshot_examplers, pos_contexts, neg_contexts):
        docs, _, _ = self.retrieve([question], [qid], [pos_contexts], [neg_contexts], topk=self.args.retrieve_topk)
        prompt = self.generator.format_longform(question, fewshot_examplers, docs[0])
        text, _, _ = self.generator.generate(
            prompt, self.args.generate_max_length,
            system_prompt=SYSTEM_PROMPT_LONGFORM
        )
        if self.args.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text, None, None


class FixLengthRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def inference(self, question, qid, fewshot_examplers, pos_contexts, neg_contexts):
        text = ""
        retrieve_question = question
        while True:
            old_len = len(text)
            docs, _, _ = self.retrieve([retrieve_question], [qid], [pos_contexts], [neg_contexts], topk=self.args.retrieve_topk)
            prompt = self.generator.format_longform(question, fewshot_examplers, docs[0])
            prompt += text
            
            if self.args.rag_method == "fix_length_retrieval":
                new_text, _, _ = self.generator.generate(prompt, self.args.generate_fix_length, system_prompt=SYSTEM_PROMPT_LONGFORM)
                if self.args.use_counter:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                text = text.strip() + " " + new_text.strip()
                retrieve_question = new_text.strip()
                
            elif self.args.rag_method == "fix_sentence_retrieval":
                new_text, _, _ = self.generator.generate(prompt, self.args.generate_max_length, system_prompt=SYSTEM_PROMPT_LONGFORM)
                if self.args.use_counter:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                new_text = new_text.strip()
                sentences = list(nlp(new_text).sents)
                sentences = [str(sent).strip() for sent in sentences]
                if len(sentences) == 0:
                    break
                text = text.strip() + " " + str(sentences[0])
                retrieve_question = sentences[0]

            else:
                raise NotImplementedError

            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.args.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        
        return text, None, None
  
    
class FLARE_RAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
        self.modifier = self.modifier_token if args.modifier_method=='token' else self.modifier_entity
        
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
    
    def inference(self, question, qid, fewshot_examplers, pos_contexts, neg_contexts):
        num_hallucination = 0
        generation_path = []
        
        text = ""
        while True:
            old_len = len(text)
            prompt = self.generator.format_longform(question, fewshot_examplers, [])
            new_text, tokens_text, logprobs = self.generator.generate(
                prompt,
                self.args.generate_max_length,
                system_prompt=SYSTEM_PROMPT_LONGFORM,
                return_logprobs=True
            )
            if self.args.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer)
            
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

                docs, _, _ = self.retrieve([retrieve_question], [qid], [pos_contexts], [neg_contexts], topk=self.args.retrieve_topk)
                prompt = self.generator.format_longform(question, fewshot_examplers, docs[0])
                prompt += " " + text + " " + ptext.strip()
                new_text, _, _ = self.generator.generate(
                    prompt,
                    self.args.generate_max_length,
                    system_prompt=SYSTEM_PROMPT_LONGFORM
                )
                if self.args.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                text = text.strip() + " " + ptext.strip() + " " + new_text.strip()
        
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.args.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        
        return text, num_hallucination, generation_path


class DRAGIN_RAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    
    def get_sentence_token_indices(self, text, tokens):
        doc = nlp(text)
        sentence_indices = []
        token_pointer = 0  # Keeps track of token position

        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_tokens = sent_text.split()  # Basic tokenization (ensure consistency with tokens)
            
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
    
    def inference(self, question, qid, fewshot_examplers, pos_contexts, neg_contexts):
        num_hallucination = 0
        generation_path = []
        
        case = f"Question: {question}\nAnswer:"
        text = ""
        while True:
            old_len = len(text)
            prompt = self.generator.format_longform(question, fewshot_examplers, [], add_case=False)
            tmp_li = [case, text]
            prompt += " ".join(s for s in tmp_li if len(s) > 0)
            
            new_text, tokens, attns, logprobs, entropies = self.generator.generate_attn(
                prompt, self.args.generate_max_length,
                system_prompt=SYSTEM_PROMPT_LONGFORM,
                use_entropy = self.args.rag_method == "dragin",
            )
            if self.args.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer)
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
                
                docs, _, _ = self.retrieve([retrieve_question], [qid], [pos_contexts], [neg_contexts], topk=self.args.retrieve_topk)
                prompt = self.generator.format_longform(question, fewshot_examplers, docs[0], add_case=False)
                tmp_li = [case, text, ptext.strip()]
                prompt += " ".join(s for s in tmp_li if len(s) > 0)
                new_text, _, _ = self.generator.generate(
                    prompt,
                    self.args.generate_max_length,
                    system_prompt=SYSTEM_PROMPT_LONGFORM
                )
                if self.args.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                text = text.strip() + " " + ptext.strip() + " " + new_text.strip()
            
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.args.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        
        return text, num_hallucination, generation_path
    
    
        