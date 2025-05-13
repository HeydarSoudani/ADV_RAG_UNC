
import random
import json
from tqdm import tqdm
from typing import List, Dict, Tuple
from sentence_transformers import CrossEncoder
from pyserini.search.lucene import LuceneSearcher

INDEX_DIR = "data/row_files/corpus"


class PositiveRet:
    def __init__(self, args):
        self.args = args
        
    def retrieve(self, queries: List[str], qids: List[str], pos_contexts, neg_contexts, topk: int = 1):
        docids, docs, scores = [], [], []
        for i, qid in enumerate(qids):
            docids_, docs_, scores_ = [], [], []
            q_contexts = pos_contexts[i] 
            if len(q_contexts) > 0:
                for ctx in q_contexts:
                    docs_.append(ctx['text']) 
                    docids_.append('0')
                    scores_.append(1)
            docs.append(docs_)
            docids.append(docids_)
            scores.append(scores_)
        return docs, docids, scores
     
class NegativeRet:
    def __init__(self, args):
        self.args = args
        
    def retrieve(self, queries: List[str], qids: List[str], pos_contexts, neg_contexts, topk: int = 1):
        docids, docs, scores = [], [], []
        for i, qid in enumerate(qids):
            docids_, docs_, scores_ = [], [], []
            q_contexts = neg_contexts[i] 
            if len(q_contexts) > 0:
                q_contexts_topk = random.sample(q_contexts, min(self.args.retrieve_topk, len(q_contexts)))
                for ctx in q_contexts_topk:
                    docs_.append(ctx['text']) 
                    docids_.append('0')
                    scores_.append(1)
            docs.append(docs_)
            docids.append(docids_)
            scores.append(scores_)
        return docs, docids, scores

class BM25:
    def __init__(self, args):
        self.args = args
        index_dir = f"{INDEX_DIR}/bm25_index"
        self.searcher = LuceneSearcher(index_dir)
        self.searcher.set_bm25(args.bm25_k1, args.bm25_b)
    
    def retrieve(self, queries: List[str], qids: List[str], pos_contexts, neg_contexts, topk: int = 1):
        bm25_hits = self.searcher.batch_search(queries, qids, k=1000, threads=20)

        docids, docs, scores = [], [], []
        for qid in qids:
            docids_, docs_, scores_ = [], [], []
            for hit in bm25_hits[qid][:self.args.retrieve_topk]:
                doc = self.searcher.doc(hit.docid)
                if doc is not None:
                    raw_content = doc.raw()
                    try:
                        content_json = json.loads(raw_content)
                        contents = content_json.get("contents") 
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON for document ID {hit.docid}")
                    docs_.append(contents)
                    docids_.append(hit.score)
                    scores_.append(doc.docid())
            docs.append(docs_)
            docids.append(docids_)
            scores.append(scores_)
        return docs, docids, scores
    

class Rerank:
    def __init__(self, args):
        self.args = args
        index_dir = f"{INDEX_DIR}/bm25_index"
        self.searcher = LuceneSearcher(index_dir)
        self.searcher.set_bm25(args.bm25_k1, args.bm25_b)
        
        rerank_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2" # cross-encoder/ms-marco-MiniLM-L12-v2
        self.cross_encoder = CrossEncoder(rerank_model_name, max_length=args.retrieve_max_query_length)
    
    def rerank_documents(self, query, docids, documents):
        query_doc_pairs = [(query, doc) for doc in documents]
        scores = self.cross_encoder.predict(query_doc_pairs)
        reranked_docs = sorted(zip(docids, scores, documents), key=lambda x: x[1], reverse=True)[:self.args.retrieve_topk]
        # return [{'docid': docid, 'context': doc, 'score': float(score)} for docid, score, doc in reranked_docs]
        docids, scores, docs = zip(*reranked_docs)
        return list(docs), list(docids), list(scores)
    
    def retrieve(self, queries: List[str], qids: List[str], pos_contexts, neg_contexts, topk: int = 1):
        bm25_hits = self.searcher.batch_search(queries, qids, k=1000, threads=20)
        
        docids, docs, scores = [], [], []
        for i, qid in enumerate(qids):
            query = queries[i]
            docids = [hit.docid for hit in bm25_hits[qid]]  # Get document IDs
            texts = []
            for docid in docids:
                doc = self.searcher.doc(docid)
                raw_content = doc.raw()
                content_json = json.loads(raw_content)
                contents = content_json.get("contents") 
                texts.append(contents)
            
            if len(texts) > 0:
                docs_, docids_, scores_ = self.rerank_documents(query, docids, texts)
            else:
                docs_ = ['' for _ in range(self.args.retrieve_topk)] 
                docids_ = ['' for _ in range(self.args.retrieve_topk)] 
                scores_ = [0 for _ in range(self.args.retrieve_topk)]
            
            docs.append(docs_)
            docids.append(docids_)
            scores.append(scores_)
        
        return docs, docids, scores
    

# TODO
class Contriever:
    pass



# TODO

# intfloat/e5-base-v2
class E5:
    pass

