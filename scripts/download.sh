#!/bin/bash

save_path=$HOME/ADV_RAG_UNC/data/search_r1_files

python $HOME/ADV_RAG_UNC/run_rag_methods/src/download.py --save_path $HOME/ADV_RAG_UNC/data

cat $HOME/ADV_RAG_UNC/data/search_r1_files/part_* > $HOME/ADV_RAG_UNC/data/search_r1_files/e5_Flat.index

gunzip $HOME/ADV_RAG_UNC/data/wiki-18.jsonl.gz

