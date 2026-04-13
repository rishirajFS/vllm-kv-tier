#!/bin/bash
#SBATCH --job-name=week1_method1
#SBATCH --output=week1_method1_%j.out
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --account=cis250224p

set -e

echo "======================================================================="
echo "Week 1 - Method 1: Batch Inference with Long Sequences"
echo "======================================================================="
echo "Start time: $(date)"
echo ""

export HF_HOME=~/workspace/vllm/hf_cache
export TRITON_CACHE_DIR=~/workspace/vllm/triton_cache
export XDG_CACHE_HOME=~/workspace/vllm/xdg_cache
export VLLM_LOGGING_LEVEL=DEBUG

source ~/workspace/vllm/.venv/bin/activate
cd ~/workspace/vllm

python test_bypass_scheduler.py

echo ""
echo "End time: $(date)"
