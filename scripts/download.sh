#!/bin/bash

save_path=$HOME/ADV_RAG_UNC/data/search_r1_files

python $HOME/ADV_RAG_UNC/src_mcts/download.py --save_path $HOME/ADV_RAG_UNC/data/search_r1_files

cat $HOME/ADV_RAG_UNC/data/search_r1_files/part_* > $HOME/ADV_RAG_UNC/data/search_r1_files/e5_Flat.index

gunzip data/search_r1_files/wiki-18.jsonl.gz

