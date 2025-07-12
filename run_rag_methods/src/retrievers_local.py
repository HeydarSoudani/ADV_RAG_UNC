# ===========================
# === Src: https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/search/retrieval_server.py
# ===========================

import json
import faiss
import torch
import warnings
import datasets
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional
from sentence_transformers import CrossEncoder
from pyserini.search.lucene import LuceneSearcher
from transformers import AutoConfig, AutoTokenizer, AutoModel



def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset(
        'json', 
        data_files=corpus_path,
        split="train",
        num_proc=4
    )
    return corpus

def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results

def load_model(model_path: str, use_fp16: bool = False):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype="auto")
    model.eval()
    model.cuda()
    if use_fp16: 
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer

def pooling(
    pooler_output,
    last_hidden_state,
    attention_mask = None,
    pooling_method = "mean"
):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")

class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        # processing query for different encoders
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]

        inputs = self.tokenizer(query_list,
                                max_length=self.max_length,
                                padding=True,
                                truncation=True,
                                return_tensors="pt"
                                )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros(
                (inputs['input_ids'].shape[0], 1), dtype=torch.long
            ).to(inputs['input_ids'].device)
            output = self.model(
                **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
            )
            query_emb = output.last_hidden_state[:, 0, :]
        elif "ReasonIR" in type(self.model).__name__:
            # query_emb = self.model.encode(query_list)
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(None,
                                output.last_hidden_state,
                                inputs['attention_mask'],
                                self.pooling_method)
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(output.pooler_output,
                                output.last_hidden_state,
                                inputs['attention_mask'],
                                self.pooling_method)
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        
        del inputs, output
        torch.cuda.empty_cache()

        return query_emb

class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retriever_name = config.retriever_name
        self.topk = config.retrieval_topk
        
        self.index_path = config.index_path
        self.corpus_path = config.corpus_path
        self._docid_to_doc = None

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: List[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)
    
    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        return self._batch_search(query_list, num, return_score)

class BM25Retriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        self.searcher = LuceneSearcher(self.index_path)
        self.searcher.set_bm25(config.bm25_k1, config.bm25_b)
        
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)
        self.max_process_num = 8
    
    def _check_contain_doc(self):
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            if return_score:
                return [], []
            else:
                return []
        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn('Not enough documents retrieved!')
        else:
            hits = hits[:num]

        if self.contain_doc:
            all_contents = [
                json.loads(self.searcher.doc(hit.docid).raw())['contents'] 
                for hit in hits
            ]
            results = [
                {
                    'title': content.split("\n")[0].strip("\""),
                    'text': "\n".join(content.split("\n")[1:]),
                    'contents': content
                } 
                for content in all_contents
            ]
        else:
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num, True)
            results.append(item_result)
            scores.append(item_score)
        if return_score:
            return results, scores
        else:
            return results

class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        print('loading index ...')
        self.index = faiss.read_index(self.index_path)
        if config.faiss_gpu:
            print("Using FAISS with GPU ...")
            # --- Multi-GPUs
            # co = faiss.GpuMultipleClonerOptions()
            # co.useFloat16 = True
            # co.shard = True
            # self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

            # --- Single-GPU
            device_id = torch.cuda.current_device()
            print(f'Using faiss_gpu on process {device_id}...')
            res = faiss.StandardGpuResources() # Get GPU resource for this device
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            self.index = faiss.index_cpu_to_gpu(res, device_id, self.index, co)

        print('loading corpus ...')
        self.corpus = load_corpus(self.corpus_path)
        self.encoder = Encoder(
            model_name = config.retriever_name,
            model_path = config.retrieval_model_path,
            pooling_method = config.retrieval_pooling_method,
            max_length = config.retrieval_query_max_length,
            use_fp16 = config.retrieval_use_fp16
        )
        self.topk = config. retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]
        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores.tolist()
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk
        
        results = []
        scores = []
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc='Retrieval process: '):
            query_batch = query_list[start_idx:start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            # load_docs is not vectorized, but is a python list approach
            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            # chunk them back
            batch_results = [batch_results[i*num : (i+1)*num] for i in range(len(batch_idxs))]
            
            results.extend(batch_results)
            scores.extend(batch_scores)
            
            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results
            torch.cuda.empty_cache()
            
        if return_score:
            return results, scores
        else:
            return results

# TODO
class ContrieverRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
    
    def _search(self, query: str, num: int = None, return_score: bool = False):
        pass

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        pass

class RerankRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        
        # Frist-stage
        self.searcher = LuceneSearcher(self.index_path)
        self.searcher.set_bm25(config.bm25_k1, config.bm25_b)
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)
        self.max_process_num = 8
    
        # Second-stage
        self.cross_encoder = CrossEncoder(config.retrieval_model_path, max_length=config.retrieval_query_max_length)
    
    def set_topk(self, new_k):
        self.topk = new_k
      
    def _check_contain_doc(self):
        return self.searcher.doc(0).raw() is not None
    
    def _rerank_documents(self, query, contents):
        query_doc_pairs = [(query, doc['contents']) for doc in contents]
        scores = self.cross_encoder.predict(query_doc_pairs)        
        reranked_docs = sorted(zip(scores, contents), key=lambda x: x[0], reverse=True)[:self.topk]
        scores, sorted_contents = zip(*reranked_docs)
        return list(sorted_contents), list(scores)
    
    def _search(self, query: str, num: int = None, return_score: bool = False):
        first_stage_num = 1000
        
        if num is None:
            num = self.topk
            
        # First-stage
        hits = self.searcher.search(query, first_stage_num)
        if len(hits) < 1:
            if return_score:
                return [], []
            else:
                return []
        
        if len(hits) < first_stage_num:
            warnings.warn('Not enough documents retrieved for first-stage!')
        else:
            hits = hits[:first_stage_num]
        
        if self.contain_doc:
            all_contents = [
                json.loads(self.searcher.doc(hit.docid).raw())['contents'] 
                for hit in hits
            ]
        else:
            docids = [hit.docid for hit in hits]
            all_contents = load_docs(self.corpus, docids)
        
        # Second-stage
        if len(all_contents) > 0:
            results, scores = self._rerank_documents(query, all_contents)
        
        
        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        pass

