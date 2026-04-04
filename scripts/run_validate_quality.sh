#!/usr/bin/env bash
#SBATCH --job-name=vllm_quality
#SBATCH --output=vllm_quality_%A.log
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=02:00:00

set -euo pipefail

# Environment Setup
cd ~/workspace/vllm
source .venv/bin/activate
export HF_HOME="/jet/home/rnagaraj/workspace/vllm/hf_cache"
export TRITON_CACHE_DIR="/jet/home/rnagaraj/workspace/vllm/triton_cache"
export XDG_CACHE_HOME="/jet/home/rnagaraj/workspace/vllm/xdg_cache"

mkdir -p benchmark_results

echo "Starting Priority 5: Output Quality Validation..."
python scripts/validate_output_quality.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --num-samples 100 \
    --output benchmark_results/quality_validation_3b.json

echo "Complete!"
