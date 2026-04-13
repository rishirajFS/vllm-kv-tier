#!/bin/bash
#SBATCH --job-name=32_concurrent
#SBATCH --output=week1_32concurrent_%j.out
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --account=cis250224p

set -e

echo "======================================================================="
echo "WEEK 1 ALTERNATIVE TEST: 32 Concurrent Requests"
echo "======================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

export HF_HOME=~/workspace/vllm/hf_cache
export TRITON_CACHE_DIR=~/workspace/vllm/triton_cache
export XDG_CACHE_HOME=~/workspace/vllm/xdg_cache

source ~/workspace/vllm/.venv/bin/activate
cd ~/workspace/vllm

python test_32_concurrent.py

echo ""
echo "End time: $(date)"
exit $?
