#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json, csv
from tqdm import tqdm

from utils.utils import set_seed

csv.field_size_limit(10**8)

subset_percentage = 1.0
id_col= 0
text_col= 1
title_col = 2

def remove_nul_lines(file):
    for line in file:
        if '\x00' not in line:
            yield line

def convert_to_pyserini_file():
    corpus_file = f"data/row_files/corpus/wikipedia_psgs_w100.tsv"
    output_file = f"data/row_files/corpus/wikipedia_psgs_w100_pyserini_format.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(corpus_file, 'r') as input:
        filtered_file = remove_nul_lines(input)
        reader = csv.reader(filtered_file, delimiter="\t")
        
        with open(output_file, 'w') as output:
            for i, row in enumerate(tqdm(reader)):                
                if row[id_col] == "id":
                    continue
                obj = {
                    "id": f"psg_{row[id_col]}",
                    "contents": row[text_col],
                    "title": row[title_col]
                }
                output.write(json.dumps(obj, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever_model", type=str, default="bm25", choices=["bm25", "dpr", "ance"])
    parser.add_argument("--index_dir_base_path", type=str, default="data/row_files/corpus")
    parser.add_argument("--dataset_name", type=str, default='wikimultihopqa', choices=[
        'wikimultihopqa', 'hotpotqa', 'musique', 'iirc', 'multihop_rag',
        'nqgold', 'trivia', 'popqa',
        'factscore'
    ])
    parser.add_argument("--dataset_subsec", type=str, default="test", choices=["train", "dev", "test", "validation"])
    parser.add_argument("--bm25_k1", type=float, default="0.9")
    parser.add_argument("--bm25_b", type=float, default="0.4")
    parser.add_argument("--bm25_top_k", type=int, default="1000")
    parser.add_argument("--rerank_top_k", type=int, default="10")
    parser.add_argument("--rel_threshold", type=int, default="1")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # = Step 0) install library
    # pip install pyserini
    
    # = Step 1) convert tsv to pyserini file
    convert_to_pyserini_file()
    
    # = Step 2) BM25 corpus indexing using pyserini
    # python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator -threads 20 -input "data/row_files/corpus" -index "data/row_files/corpus/bm25_index" -storePositions -storeDocvectors -storeRaw
    
    
# python data/corpus_preparation.py

