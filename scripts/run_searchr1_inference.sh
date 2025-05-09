#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_a100
#SBATCH --time=6:00:00
#SBATCH --mem=80GB
#SBATCH --output=script_logging/slurm_%A.out

module load 2024
module load Python/3.12.3-GCCcore-13.3.0


### === Set variables ==========================
model_name_or_path="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo"
dataset="musique"
subsec="dev"
fraction_of_data_to_use=1.0
retriever_name="e5"
index_path="data/search_r1_files/e5_Flat.index"
retrieval_model_path="intfloat/e5-base-v2"
run="run_20 (search_r1_full)"

srun python $HOME/ADV_RAG_UNC/run_searchr1/inference.py \
    --model_name_or_path "$model_name_or_path" \
    --dataset "$dataset" \
    --subsec "$subsec" \
    --fraction_of_data_to_use "$fraction_of_data_to_use" \
    --retriever_name "$retriever_name" \
    --index_path "$index_path" \
    --retrieval_model_path "$retrieval_model_path" \
    --run "$run"


### Datasets:
    # 'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'

### retriever_model:
    # 'bm25', 'contriever', 'rerank_l6', 'rerank_l12', 'e5'

