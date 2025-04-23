#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_a100
#SBATCH --time=0:05:00
#SBATCH --output=script_logging/slurm_%A.out

conda init
conda activate retriever

file_path=$HOME/ADV_RAG_UNC/data/search_r1_files
corpus_file=$file_path/wiki-18.jsonl

# For E5
retriever_name=e5
index_file=$file_path/e5_Flat.index
retriever_path=intfloat/e5-base-v2
python $HOME/ADV_RAG_UNC/src_mcts/retrieval_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 3 \
    --retriever_name $retriever_name \
    --retriever_model $retriever_path \
    --faiss_gpu


# For BM25
retriever_name=bm25
index_file=$file_path/bm25
python $HOME/ADV_RAG_UNC/src_mcts/retrieval_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --retriever_name $retriever_name \
    --topk 3
    


python $HOME/ADV_RAG_UNC/run_searchr1/retrieval_server.py \
    --index_path $HOME/ADV_RAG_UNC/data/search_r1_files/bm25 \
    --corpus_path $HOME/ADV_RAG_UNC/data/search_r1_files/wiki-18.jsonl \
    --topk 3 \
    --retriever_name bm25
    