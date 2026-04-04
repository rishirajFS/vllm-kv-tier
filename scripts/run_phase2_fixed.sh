#!/usr/bin/env bash
#SBATCH -A cis250224p
#SBATCH --job-name=vllm_phase2
#SBATCH --output=vllm_phase2_%A.log
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

echo "Starting Phase 2.1: Qwen 1.5B ShareGPT"
python -m kv_cache_tiering.benchmarks.benchmark \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --policies lru attention hybrid \
  --num-prompts 200 \
  --dataset sharegpt \
  --dataset-path datasets/sharegpt.json \
  --gpu-mem-util 0.15 \
  --max-model-len 2048 \
  --output benchmark_results/results_qwen1.5b_sharegpt_fixed_${TIMESTAMP}.json

echo "Starting Phase 2.2: Qwen 3B ShareGPT"
python -m kv_cache_tiering.benchmarks.benchmark \
  --model Qwen/Qwen2.5-3B-Instruct \
  --policies lru attention hybrid \
  --num-prompts 200 \
  --dataset sharegpt \
  --dataset-path datasets/sharegpt.json \
  --gpu-mem-util 0.20 \
  --max-model-len 4096 \
  --output benchmark_results/results_qwen3b_sharegpt_fixed_${TIMESTAMP}.json

echo "Starting Phase 2.3: Qwen 3B MS-MARCO"
python -m kv_cache_tiering.benchmarks.benchmark \
  --model Qwen/Qwen2.5-3B-Instruct \
  --policies lru attention hybrid \
  --num-prompts 200 \
  --dataset msmarco \
  --dataset-path datasets/msmarco.json \
  --gpu-mem-util 0.20 \
  --max-model-len 4096 \
  --output benchmark_results/results_qwen3b_msmarco_fixed_${TIMESTAMP}.json

echo "Starting Phase 2.4: Qwen 3B HumanEval"
python -m kv_cache_tiering.benchmarks.benchmark \
  --model Qwen/Qwen2.5-3B-Instruct \
  --policies lru attention hybrid \
  --num-prompts 164 \
  --dataset humaneval \
  --dataset-path datasets/humaneval.json \
  --gpu-mem-util 0.20 \
  --max-model-len 4096 \
  --output benchmark_results/results_qwen3b_humaneval_fixed_${TIMESTAMP}.json

echo "Starting Phase 2.5: Llama 3.2-1B ShareGPT"
python -m kv_cache_tiering.benchmarks.benchmark \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --policies lru attention hybrid \
  --num-prompts 200 \
  --dataset sharegpt \
  --dataset-path datasets/sharegpt.json \
  --gpu-mem-util 0.15 \
  --max-model-len 2048 \
  --output benchmark_results/results_llama1b_sharegpt_fixed_${TIMESTAMP}.json

echo "Phase 2 Complete!"
