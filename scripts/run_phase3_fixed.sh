#!/usr/bin/env bash
#SBATCH -A cis250224p
#SBATCH --job-name=vllm_phase3
#SBATCH --output=vllm_phase3_%A.log
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=06:00:00

set -euo pipefail
cd ~/workspace/vllm
source .venv/bin/activate
export HF_HOME="/jet/home/rnagaraj/workspace/vllm/hf_cache"
export TRITON_CACHE_DIR="/jet/home/rnagaraj/workspace/vllm/triton_cache"
export XDG_CACHE_HOME="/jet/home/rnagaraj/workspace/vllm/xdg_cache"

mkdir -p benchmark_results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Starting Phase 3.1: Memory Pressure Sweep"
python scripts/benchmark_memory_pressure_sweep.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset sharegpt \
  --num-prompts 200 \
  --output benchmark_results/memory_sweep_${TIMESTAMP}.json

echo "Starting Phase 3.2: Hybrid Ablation"
python scripts/benchmark_hybrid_ablation.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset-name sharegpt \
  --num-prompts 200 \
  --gpu-mem-util 0.10 \
  --output benchmark_results/hybrid_ablation_${TIMESTAMP}.json

echo "Starting Phase 3.3: Failure Modes"
python scripts/benchmark_failure_modes.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --gpu-mem-util 0.10 \
  --output benchmark_results/failure_modes_${TIMESTAMP}.json

echo "Phase 3 Complete!"
