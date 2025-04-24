#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_a100
#SBATCH --time=6:00:00
#SBATCH --output=script_logging/slurm_%A.out

module load 2024
module load Python/3.12.3-GCCcore-13.3.0


### === Set variables ==========================
model_name_or_path="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo"
dataset="musique"
subsec="dev"
fraction_of_data_to_use=1.0
retriever_name="rerank"
run="run_4 (search_r1)"

srun python $HOME/ADV_RAG_UNC/run_searchr1/inference.py \
    --model_name_or_path "$model_name_or_path" \
    --dataset "$dataset" \
    --subsec "$subsec" \
    --fraction_of_data_to_use "$fraction_of_data_to_use" \
    --retriever_name "$retriever_name" \
    --run "$run"


### Datasets:
    # 'nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'

### retriever_model:
    # 'bm25', 'contriever', 'rerank', 'e5'

