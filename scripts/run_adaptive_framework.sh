#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --time=1:20:00
#SBATCH --output=script_logging/slurm_%A.out

module load 2024
module load Python/3.12.3-GCCcore-13.3.0
# module load Java/21.0.2
# pip install --upgrade transformers


python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator -threads 20 -input "data/row_files/corpus" -index "data/row_files/corpus/bm25_index" -storePositions -storeDocvectors -storeRaw


# ### === Set variables ==========================
# model_name_or_path="meta-llama/Llama-3.1-8B-Instruct"
# dataset="2wikimultihopqa"
# subsec="test"
# rag_type="no_retrieval"
# retriever="bm25"
# fraction_of_data_to_use=0.6    # nqgold 0.104 | trivia 0.034 | popqa 0.021 | 2wikimultihopqa 0.6
# accuracy_metric="exact_match"    # model_judge | exact_match
# model_eval="gpt-3.5-turbo"
# run="run_1 (300s-EM)"          # run_0 (300s-G3.5) | run_1 (300s-EM)

# srun python $HOME/RAG_UNC/_truth_torch_framework/run/run_framework.py \
#     --model_name_or_path "$model" \
#     --dataset "$dataset" \
#     --subsec "$subsec" \
#     --prompt_format "$prompt_format" \
#     --fraction_of_data_to_use "$fraction_of_data_to_use"\
#     --accuracy_metric "$accuracy_metric" \
#     --model_eval "$model_eval" \
#     --run "$run" \



# prompt_format:
    # 'only_q', 'q_negative', 'q_positive',
    # 'bm25_retriever_top1', 'bm25_retriever_top5',
    # 'contriever_retriever_top1', 'contriever_retriever_top5',
    # 'rerank_retriever_top1', 'rerank_retriever_top5'

# rag_type:
    # 'no_retrieval', 'single_retrieval',
    # 'fix_length_retrieval', 'fix_sentence_retrieval',
    # 'flare', 'dragin'

# retriever:
    # 'positive', 'negative', 'bm25', 'contriever', 'rerank', 'bge_m3', 'sgpt'

# Datasets:
    # '2wikimultihopqa', 'hotpotqa', 'musique', 'iirc',
    # 'nqgold', 'trivia', 'popqa',
    # 

# Model name:
    # GPT-4o:     "openai/gpt-4o"
    # GPT-3.5:    "openai/gpt-3.5-turbo-instruct"
    
    # llama3.2:   "meta-llama/Llama-3.2-3B-Instruct"
    # llama3.2:   "meta-llama/Llama-3.2-1B-Instruct"
    # llama3.1:   "meta-llama/Llama-3.1-8B-Instruct"
    # llama2:     "meta-llama/Llama-2-7b-chat-hf"
    # Qwen2.5:    "Qwen/Qwen2.5-7B-Instruct"
    # Gemma2:     "google/gemma-2-9b-it"
    # Phi4 (14B): "microsoft/phi-4"
    # Mistral:    "mistralai/Mistral-7B-Instruct-v0.3"

    # DS-7B: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    # DS-8B: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    # DS-14B: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
    
    
    

