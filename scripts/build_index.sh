#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_a100
#SBATCH --time=34:00:00
#SBATCH --mem=160GB
#SBATCH --output=script_logging/slurm_%A.out

module load 2024
module load Python/3.12.3-GCCcore-13.3.0

### === Set variables ==========================
corpus_file=data/search_r1_files/wiki-18.jsonl # jsonl
save_dir=data/search_r1_files
retriever_name=reasonir # this is for indexing naming
retriever_model=reasonir/ReasonIR-8B

# change faiss_type to HNSW32/64/128 for ANN indexing
# change retriever_name to bm25 for BM25 indexing
srun python $HOME/ADV_RAG_UNC/run_searchr1/index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 256 \
    --batch_size 64 \
    --pooling_method mean \
    --faiss_type Flat \
    --save_embedding