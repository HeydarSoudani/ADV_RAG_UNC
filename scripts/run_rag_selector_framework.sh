#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_h100
#SBATCH --time=0:30:00
#SBATCH --mem=20GB
#SBATCH --output=script_logging/slurm_%A.out

module load 2024
module load Python/3.12.3-GCCcore-13.3.0


### === Set variables ==========================
selector_model_name_or_path="answerdotai/ModernBERT-large"
secondary_model_name_or_path="Qwen/Qwen2.5-7B-Instruct"
dataset="hotpotqa"
subsec="dev"
retriever_name="rerank_l6"
consistency_method="rag_consistency"
ensemble_method="rag_ensemble"
with_clustering="False"
with_confidence="False"
is_encoder_frozen="False"
confidence_score_injection="in_input"
training_method="pairwise"
prompt_format="x_o_c_g_dc"
run_train="run_1 (rag_methods_2k)"
run_test="run_3 (rag_methods_500)"

args=(
  --selector_model_name_or_path "$selector_model_name_or_path"
  --secondary_model_name_or_path "$secondary_model_name_or_path"
  --dataset "$dataset"
  --subsec "$subsec"
  --retriever_name "$retriever_name"
  --consistency_method "$consistency_method"
  --ensemble_method "$ensemble_method"
  --confidence_score_injection "$confidence_score_injection"
  --training_method "$training_method"
  --prompt_format "$prompt_format"
  --run_train "$run_train"
  --run_test "$run_test"
)
[[ "$with_clustering"    == "True" ]] && args+=(--with_clustering)
[[ "$with_confidence"    == "True" ]] && args+=(--with_confidence)
[[ "$is_encoder_frozen"  == "True" ]] && args+=(--is_encoder_frozen)

accelerate launch --multi_gpu \
  "$HOME/ADV_RAG_UNC/run_rag_selector/run_framework.py" \
  "${args[@]}"


# accelerate launch --multi_gpu $HOME/ADV_RAG_UNC/run_rag_selector/run_framework.py \
#     --selector_model_name_or_path "$selector_model_name_or_path" \
#     --secondary_model_name_or_path "$secondary_model_name_or_path" \
#     --dataset "$dataset" \
#     --subsec "$subsec" \
#     --retriever_name "$retriever_name" \
#     --consistency_method "$consistency_method" \
#     --ensemble_method "$ensemble_method" \
#     --with_clustering "$with_clustering" \
#     --with_confidence "$with_confidence" \
#     --is_encoder_frozen "$is_encoder_frozen" \
#     --confidence_score_injection "$confidence_score_injection" \
#     --training_method "$training_method" \
#     --prompt_format "$prompt_format" \
#     --run_train "$run_train" \
#     --run_test "$run_test"

