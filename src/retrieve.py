
import json
from tqdm import tqdm
from typing import List, Dict, Tuple
from sentence_transformers import CrossEncoder
from pyserini.search.lucene import LuceneSearcher

INDEX_DIR = "data/row_files/corpus"

class BM25:
    def __init__(self, args):
        self.args = args
        index_dir = f"{INDEX_DIR}/bm25_index"
        self.searcher = LuceneSearcher(index_dir)
        self.searcher.set_bm25(args.bm25_k1, args.bm25_b)
    
    def retrieve(self, queries: List[str], qids: List[str], topk: int = 1):
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
        
        rerank_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.cross_encoder = CrossEncoder(rerank_model_name, max_length=args.retrieve_max_query_length)
    
    def rerank_documents(self, query, docids, documents):
        query_doc_pairs = [(query, doc) for doc in documents]
        scores = self.cross_encoder.predict(query_doc_pairs)
        reranked_docs = sorted(zip(docids, scores, documents), key=lambda x: x[1], reverse=True)[:self.args.retrieve_topk]
        # return [{'docid': docid, 'context': doc, 'score': float(score)} for docid, score, doc in reranked_docs]
        
        docids, scores, docs = zip(*reranked_docs)
        return list(docs), list(docids), list(scores)
    
    def retrieve(self, queries: List[str], qids: List[str], topk: int = 1):
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
            
            docs_, docids_, scores_ = self.rerank_documents(query, docids, texts)
            docs.append(docs_)
            docids.append(docids_)
            scores.append(scores_)
        
        return docs, docids, scores