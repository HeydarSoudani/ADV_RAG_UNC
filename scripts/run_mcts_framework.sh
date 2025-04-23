#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_a100
#SBATCH --time=3:30:00
#SBATCH --output=script_logging/slurm_%A.out

module load 2024
module load Python/3.12.3-GCCcore-13.3.0


### === Set variables ==========================
model_name_or_path="Qwen/Qwen2.5-7B-Instruct"
dataset="hotpotqa"
subsec="test"
fraction_of_data_to_use=1.0
retriever_model="rerank"
num_rollouts=4
max_depth_allowed=4
run="run_1 (rollout_4)"

srun python $HOME/ADV_RAG_UNC/run_mcts/run_framework.py \
    --model_name_or_path "$model_name_or_path" \
    --dataset "$dataset" \
    --subsec "$subsec" \
    --fraction_of_data_to_use "$fraction_of_data_to_use" \
    --retriever_model "$retriever_model" \
    --num_rollouts "$num_rollouts" \
    --max_depth_allowed "$max_depth_allowed" \
    --run "$run" \
    
    

### Datasets:
    # 'wikimultihopqa', 'hotpotqa', 'musique', 'iirc'

### rag_method:
    # 'no_retrieval', 'single_retrieval',
    # 'fix_length_retrieval', 'fix_sentence_retrieval',
    # 'flare', 'dragin'

### retriever_model:
    # 'negative', 'bm25', 'contriever', 'rerank', 'bge_m3', 'sgpt', 'positive'

### Model name:
    # GPT-4o:     "openai/gpt-4o"
    # GPT-3.5:    "openai/gpt-3.5-turbo-instruct"
    
    # llama3.2:   "meta-llama/Llama-3.2-3B-Instruct"
    # llama3.2:   "meta-llama/Llama-3.2-1B-Instruct"
    # llama3.1:   "meta-llama/Llama-3.1-8B-Instruct"
    # llama2(13B):"meta-llama/Llama-2-13b-chat-hf"
    # llama2(7B): "meta-llama/Llama-2-7b-chat-hf"
    
    # Qwen2.5:    "Qwen/Qwen2.5-7B-Instruct"
    # Gemma2:     "google/gemma-2-9b-it"
    # Phi4 (14B): "microsoft/phi-4"
    # Falcon3:    "tiiuae/Falcon3-10B-Instruct"
    # Mistral:    "mistralai/Mistral-7B-Instruct-v0.3"
    # Vicuna:     "lmsys/vicuna-13b-v1.5"

    # DS-7B: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    # DS-8B: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    # DS-14B: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B

