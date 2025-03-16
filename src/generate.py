
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import spacy
import torch
import numpy as np
from math import exp
from scipy.special import softmax
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.retrieve import BM25, Rerank, PositiveRet, NegativeRet
from utils.common_utils import fix_tokenizer_chat
from src.templetes import DEFAULT_SYSTEM_PROMPT, DEFAULT_REGENERATE_SYSTEM_PROMPT

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
        self.space_token = self.tokenizer.tokenize(' ')[0]
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    
    # Ths original DRAGIN code
    # def generate(self, input_text, max_new_tokens, return_logprobs=False):
    #     input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
    #     input_ids = input_ids.to(self.generator.device)
    #     input_length = input_ids.shape[1]
    #     attention_mask = torch.ones_like(input_ids)

    #     if return_logprobs:
    #         outputs = self.generator.generate(
    #             input_ids = input_ids, 
    #             attention_mask = attention_mask,
    #             max_new_tokens = max_new_tokens, 
    #             return_dict_in_generate = True, 
    #             output_scores = True,
    #         )
    #         transition_scores = self.generator.compute_transition_scores(
    #             outputs.sequences, outputs.scores, normalize_logits=True
    #         )

    #         generated_tokens = outputs.sequences[:, input_length:]
    #         text = self.tokenizer.decode(generated_tokens[0]) # text = "".join(tokens)
    #         tokens = [self.tokenizer.decode(t) for t in generated_tokens[0]]
    #         logprobs = transition_scores[0]
    #         logprobs = [p.cpu().numpy() for p in logprobs]
    #         assert len(tokens) == len(logprobs)
    #         return text, tokens, logprobs
        
    #     else:
    #         outputs = self.generator.generate(
    #             input_ids = input_ids, 
    #             max_new_tokens = max_new_tokens, 
    #             attention_mask = attention_mask,
    #         )
    #         generated_tokens = outputs[:, input_length:]
    #         text = self.tokenizer.decode(generated_tokens[0])
    #         return text, None, None
    
    
    # My code adopted with chat template
    def generate(self, input_text, max_new_tokens,
        system_prompt:str = DEFAULT_SYSTEM_PROMPT,
        return_logprobs=True, return_text=True,
        add_generation_prompt=True, continue_final_message=False
    ):
        messages = [{'role': 'system', 'content': system_prompt}]
        messages.append({'role': 'user', 'content': input_text})
        tokenizer, messages = fix_tokenizer_chat(self.tokenizer, messages)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message
        )
        # print(f"{text}\n")
        
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt").to(self.generator.device)
            input_ids = inputs['input_ids']
            model_output = self.generator.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_logits=True,
                output_scores = True,
                eos_token_id=self.eos_token_id
            )

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
                tokens_text = [[tokenizer.decode(token) for token in tokens_] for tokens_ in tokens]
                generated_texts = tokenizer.batch_decode(tokens, skip_special_tokens=True)
            
            if return_logprobs:
                logits_list = torch.stack(model_output.logits).cpu().permute(1, 0, 2)
                model_output.logits = None
                logprobs = torch.log_softmax(logits_list, dim=-1) #logprobs for each token
                logprobs = torch.gather(logprobs, dim=-1, index = model_output.sequences[:, len(input_ids[0]):].unsqueeze(-1))#logprobs for each token in the generated text
                logprobs = logprobs.squeeze(-1).tolist()#convert to list
                logprobs = [logprobs[i][:indices[i]+1] for i in range(len(logprobs))]
        
        assert len(tokens_text[0]) == len(logprobs[0])
        return generated_texts[0], tokens_text[0], logprobs[0]
    
    
    def generate_attn(self, input_text, max_new_tokens,
        solver="max", use_entropy = False, use_logprob = False,
        add_generation_prompt=True, continue_final_message=False
    ):
        
        messages = [{'role': 'system', 'content': DEFAULT_SYSTEM_PROMPT}]
        messages.append({'role': 'user', 'content': input_text})
        tokenizer, messages = fix_tokenizer_chat(self.tokenizer, messages)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message
        )
        
        with torch.no_grad():
            input_ids = self.tokenizer.encode(text, return_tensors="pt")
            input_ids = input_ids.to(self.generator.device)
            input_length = input_ids.shape[1]
            attention_mask = torch.ones_like(input_ids)
        
            model_output = self.generator.generate(
                input_ids = input_ids, 
                attention_mask = attention_mask,
                max_new_tokens = max_new_tokens, 
                return_dict_in_generate = True, 
                output_scores = True,
                eos_token_id=self.eos_token_id
            )
            generated_tokens = model_output.sequences[:, input_length:]
            tokens = self.tokenizer.convert_ids_to_tokens(generated_tokens[0])
            text = self.tokenizer.decode(generated_tokens[0])
        
            # merge tokens
            range_ = []
            for i, t in enumerate(tokens):
                if i == 0 or t.startswith(self.space_token) or generated_tokens[0][i] == 13 or tokens[i-1] == '</s>':
                    range_.append([i, i])
                else:
                    range_[-1][-1] += 1
        
            # attention
            atten = self.generator.generate(generated_tokens, return_dict_in_generate=True, output_attentions=True).attentions[-1][0]
            if solver == "max": 
                mean_atten, _ = torch.max(atten, dim=1)
                mean_atten = torch.mean(mean_atten, dim=0)
            elif solver == "avg":
                mean_atten = torch.sum(atten, dim=1)
                mean_atten = torch.mean(mean_atten, dim=0)
                for i in range(mean_atten.shape[0]):
                    mean_atten[i] /= (mean_atten.shape[0] - i)
            elif solver == "last_token":
                mean_atten = torch.mean(atten[:, -1], dim=0)
            else:
                raise NotImplementedError
            if mean_atten.shape[0] > 1 and tokens[0] == '</s>':
                mean_atten = mean_atten / sum(mean_atten[1:]).item()
        
            # regular tokens
            seqlist = []
            attns = []
            for r in range_:
                tokenseq = "".join(tokens[r[0]: r[1]+1]).replace(self.space_token, "")
                value = mean_atten[r[0]: r[1]+1].sum().item()
                seqlist.append(tokenseq)
                attns.append(value)

            # -log prob
            if use_logprob:
                transition_scores = self.generator.compute_transition_scores(
                    model_output.sequences, model_output.scores, normalize_logits=True
                )
                logprobs = transition_scores[0]
                logprobs = [p.cpu().numpy() for p in logprobs]
                assert len(tokens) == len(logprobs)
                seqlogprobs = []
                for r in range_:
                    logprobseq = sum(logprobs[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                    seqlogprobs.append(logprobseq)
            else:
                seqlogprobs = None

            # entropy
            if use_entropy:
                tmp = []
                for v in model_output.scores:
                    tmp.append(v.cpu())
                softmax_probs = softmax(tmp, axis=-1)
                entropies = -np.sum(softmax_probs * np.log(softmax_probs + 1e-10), axis=-1)
                entropies = [v[0] for v in entropies]
                seqentropies = []
                for r in range_:
                    entropyseq = sum(entropies[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                    seqentropies.append(entropyseq) 
            else:
                seqentropies = None 
        
        
        return text, seqlist, attns, seqlogprobs, seqentropies
    

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
   
    def format(self, question, fewshot_examplers, docs, add_case=True):
        prompt = ""
        for exp in fewshot_examplers:
            prompt += f"Question: {exp['question']}\n"
            prompt += f"Answer: {exp['cot']} So, the answer is {exp['answer']}.\n"
        prompt += "\n"
        
        if len(docs) > 0:
            prompt += "Context:\n"
            for i, doc in enumerate(docs):
                prompt += f"[{i+1}] {doc}\n"
            prompt += "\n"
        
        # prompt += "Answer in the same format as before.\n" 
        prompt += "Answer the following question by reasoning step-by-step, following the examples above.\n"
        if add_case:
            prompt += f"Question: {question}\nAnswer:"
        
        return prompt
     
    def regenerate(self, question, fewshot_examplers, pred, generate_max_length=10):
        prompt = self.format(question, fewshot_examplers, [])
        prompt += pred
        prompt += " So, the answer is"
        text, _, _ = self.generator.generate(prompt, generate_max_length, system_prompt=DEFAULT_REGENERATE_SYSTEM_PROMPT)
        return text
    
    
class NoRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def inference(self, question, qid, fewshot_examplers, pos_contexts, neg_contexts):
        prompt = self.format(question, fewshot_examplers, [])
        text, _, _ = self.generator.generate(prompt, self.args.generate_max_length)
        if self.args.use_counter:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text
    
    
class SingleRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def inference(self, question, qid, fewshot_examplers, pos_contexts, neg_contexts):
        docs, _, _ = self.retrieve([question], [qid], [pos_contexts], [neg_contexts], topk=self.args.retrieve_topk)
        prompt = self.format(question, fewshot_examplers, docs[0])
        text, _, _ = self.generator.generate(prompt, self.args.generate_max_length)
        if self.args.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text


class FixLengthRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def inference(self, question, qid, fewshot_examplers, pos_contexts, neg_contexts):
        text = ""
        retrieve_question = question
        while True:
            old_len = len(text)
            docs, _, _ = self.retrieve([retrieve_question], [qid], [pos_contexts], [neg_contexts], topk=self.args.retrieve_topk)
            prompt = self.format(question, fewshot_examplers, docs[0])
            prompt += text
            
            if self.args.rag_method == "fix_length_retrieval":
                new_text, _, _ = self.generator.generate(prompt, self.args.generate_fix_length)
                if self.args.use_counter:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                text = text.strip() + " " + new_text.strip()
                retrieve_question = new_text.strip()
                
            elif self.args.rag_method == "fix_sentence_retrieval":
                new_text, _, _ = self.generator.generate(prompt, self.args.generate_max_length)
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
        
        return text
  
    
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
        text = ""
        while True:
            old_len = len(text)
            prompt = self.format(question, fewshot_examplers, [])
            new_text, tokens_text, logprobs = self.generator.generate(prompt, self.args.generate_max_length, return_logprobs=True)
            if self.args.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer)
            ptext, curr, hallucination = self.modifier(new_text, tokens_text, logprobs)
            if not hallucination:
                text = text.strip() + " " + new_text.strip()
            else:
                if self.args.query_formulation == "direct":
                    retrieve_question = curr.replace("[xxx]", "")
                elif self.args.query_formulation == "forward_all":
                    tmp_all = [question, text, ptext]
                    retrieve_question = " ".join(s for s in tmp_all if len(s) > 0)
                else:
                    raise NotImplemented

                docs, _, _ = self.retrieve([retrieve_question], [qid], [pos_contexts], [neg_contexts], topk=self.args.retrieve_topk)
                prompt = self.format(question, fewshot_examplers, docs[0])
                prompt += text + " " + ptext.strip()
                new_text, _, _ = self.generator.generate(prompt, self.args.generate_max_length)
                if self.args.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                text = text.strip() + " " + ptext.strip() + " " + new_text.strip()
        
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.args.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        
        return text


class DRAGIN_RAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def modifier(self, text, tokens, attentions, weight):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        tid = 0
        for sid, sent in enumerate(sentences):
            tl, tr = tid, tid
            if sid == len(sentences) - 1:
                tl, tr = tid, len(tokens)
            else:
                for i in range(tid + 1, len(tokens)):
                    seq = " ".join(tokens[tl:i])
                    if sent in seq:
                        tr = i
                        break
                tid = tr
            # value = attenion * (-log prob)
            attns = attentions[tl:tr]
            attns = np.array(attns) / sum(attns)
            value = [attns[i-tl] * weight[i] * (tr-tl) for i in range(tl, tr)] 
            thres = [1 if v > self.args.hallucination_threshold else 0 for v in value]
            if 1 in thres:
                # hallucinated
                if "check_real_words" in self.__dict__ and self.check_real_words:
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
                
                prev = "" if sid == 0 else " ".join(sentences[:sid])
                # curr = " ".join(
                #     [tokens[i] if thres[i] == 0 else "[xxx]" for i in range(len(thres))]
                # )
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
        forward_attns = torch.zeros(len(tokens))
        hit_cnt = 0
        for i in range(len(curr_hit)):
            if curr_hit[i] == 1:
                forward_attns += attns[curr_st + i]
                hit_cnt += 1
        forward_attns /= hit_cnt
        forward_attns = forward_attns.tolist()

        # 分析词性，保留实词对应的 attns
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
        
        # if "retrieve_keep_top_k" in self.__dict__:
        #     top_k = min(self.retrieve_keep_top_k, len(real_pairs))
        # elif "retrieve_keep_ratio" in self.__dict__:
        #     top_k = int(len(real_pairs) * self.retrieve_keep_ratio)
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
        case = f"Question: {question}\nAnswer:"
        text = ""
        while True:
            old_len = len(text)
            prompt = self.format(question, fewshot_examplers, [], add_case=False)
            tmp_li = [case, text]
            prompt += " ".join(s for s in tmp_li if len(s) > 0)
            
            new_text, tokens, attns, logprobs, entropies = self.generator.generate_attn(
                prompt, self.args.generate_max_length,  
                use_entropy = self.args.rag_method == "dragin",
                # use_logprob = self.method == "attn_prob"
            )
            if self.args.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer)
            weight = entropies if self.args.rag_method == "dragin" else [-v for v in logprobs]
            hallucination, ptext, curr_tokens, curr_hit =  self.modifier(new_text, tokens, attns, weight)
            
            if not hallucination:
                text = text.strip() + " " + new_text.strip()
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
                    # assert "retrieve_keep_top_k" in self.__dict__
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
            
                docs, _, _ = self.retrieve([retrieve_question], [qid], [pos_contexts], [neg_contexts], topk=self.args.retrieve_topk)
                prompt = self.format(question, fewshot_examplers, docs[0], add_case=False)
                tmp_li = [case, text, ptext.strip()]
                prompt += " ".join(s for s in tmp_li if len(s) > 0)
                
                new_text, _, _ = self.generator.generate(prompt, self.args.generate_max_length)
                if self.args.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                text = text.strip() + " " + ptext.strip() + " " + new_text.strip()
            
            
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.args.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        
        return text
    
    
        