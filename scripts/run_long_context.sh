#!/usr/bin/env bash
#SBATCH --job-name=vllm_long_ctx
#SBATCH --output=vllm_long_context_%A.log
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=04:00:00

set -euo pipefail

# Environment Setup
cd ~/workspace/vllm
source .venv/bin/activate
export HF_HOME="/jet/home/rnagaraj/workspace/vllm/hf_cache"
export TRITON_CACHE_DIR="/jet/home/rnagaraj/workspace/vllm/triton_cache"
export XDG_CACHE_HOME="/jet/home/rnagaraj/workspace/vllm/xdg_cache"

mkdir -p benchmark_results

echo "Starting Priority 3: Long-Context Stress Test (4K -> 128K)..."
python scripts/benchmark_long_context.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --policies lru attention hybrid \
    --output benchmark_results/long_context_3b.json

echo "Complete!"
