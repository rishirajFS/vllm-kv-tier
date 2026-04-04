#!/usr/bin/env bash
#SBATCH -A cis250224p
#SBATCH --job-name=vllm_phase4_7b
#SBATCH --output=vllm_phase4_7b_%A.log
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=04:00:00

set -euo pipefail
cd ~/workspace/vllm
source .venv/bin/activate
export HF_HOME="/jet/home/rnagaraj/workspace/vllm/hf_cache"
export TRITON_CACHE_DIR="/jet/home/rnagaraj/workspace/vllm/triton_cache"
export XDG_CACHE_HOME="/jet/home/rnagaraj/workspace/vllm/xdg_cache"

mkdir -p benchmark_results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Starting Phase 4.1: Qwen 7B ShareGPT"
python -m kv_cache_tiering.benchmarks.benchmark \
  --model Qwen/Qwen2.5-7B-Instruct \
  --policies lru attention hybrid \
  --num-prompts 200 \
  --dataset sharegpt \
  --dataset-path datasets/sharegpt.json \
  --gpu-mem-util 0.50 \
  --max-model-len 2048 \
  --output benchmark_results/results_qwen7b_sharegpt_${TIMESTAMP}.json

echo "Starting Phase 4.2: Qwen 7B MS-MARCO"
python -m kv_cache_tiering.benchmarks.benchmark \
  --model Qwen/Qwen2.5-7B-Instruct \
  --policies lru attention hybrid \
  --num-prompts 200 \
  --dataset msmarco \
  --dataset-path datasets/msmarco.json \
  --gpu-mem-util 0.50 \
  --max-model-len 2048 \
  --output benchmark_results/results_qwen7b_msmarco_${TIMESTAMP}.json

echo "Starting Phase 4.3: Qwen 7B HumanEval"
python -m kv_cache_tiering.benchmarks.benchmark \
  --model Qwen/Qwen2.5-7B-Instruct \
  --policies lru attention hybrid \
  --num-prompts 164 \
  --dataset humaneval \
  --dataset-path datasets/humaneval.json \
  --gpu-mem-util 0.50 \
  --max-model-len 2048 \
  --output benchmark_results/results_qwen7b_humaneval_${TIMESTAMP}.json

echo "Phase 4 Complete!"
