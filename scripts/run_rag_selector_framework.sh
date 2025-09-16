#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_a100
#SBATCH --time=1:00:00
#SBATCH --mem=80GB
#SBATCH --output=script_logging/slurm_%A.out

module load 2024
module load Python/3.12.3-GCCcore-13.3.0


### === Set variables ==========================
selector_model_name_or_path="answerdotai/ModernBERT-large"
secondary_model_name_or_path="Qwen/Qwen2.5-7B-Instruct"
dataset="musique"
subsec="train"
retriever_name="rerank_l6"
prompt_format="x_o_c_g_dc"
consistency_method="rag_consistency"
get_ideal="True"
with_training="True"
with_clustering="True"
confidence_score_injection="in_input"
training_method="pairwise"
run_train="run_1 (rag_methods_2k)"
run_test="run_2 (rag_methods_1k)"

accelerate launch --multi_gpu $HOME/run_rag_selector/run_framework.py \
    --selector_model_name_or_path "$selector_model_name_or_path" \
    --secondary_model_name_or_path "$secondary_model_name_or_path" \
    --dataset "$dataset" \
    --subsec "$subsec" \
    --retriever_name "$retriever_name" \
    --prompt_format "$prompt_format" \
    --consistency_method "$consistency_method" \
    --get_ideal "$get_ideal"\
    --with_training "$with_training"\
    --with_clustering "$with_clustering"\
    --confidence_score_injection "$confidence_score_injection"\
    --training_method "$training_method"\
    --run_train "$run_train"\
    --run_test "$run_test"


### prompt_format:
# - 'x_o', 'x_o_sq', 'x_o_th', 'x_o_dc', 'x_o_g', 'x_o_g_sq', 'x_o_g_dc', 'x_o_sq_dc', 'x_o_sq_th_dc',
# - 'x_o_c', 'x_o_c_sq', 'x_o_c_th', 'x_o_c_dc', 'x_o_c_g', 'x_o_c_g_sq', 'x_o_c_g_dc', 'x_o_c_sq_dc', 'x_o_c_sq_th_dc',
