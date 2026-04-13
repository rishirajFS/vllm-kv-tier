#!/bin/bash
#SBATCH --job-name=check_connector
#SBATCH --output=slurm-%j-check-connector.out
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --account=cis250224p

# Quick diagnostic to verify KV connector initialization

set -e

echo "======================================================================="
echo "KV Connector Diagnostic"
echo "======================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

module load cuda/12.4
export HF_HOME=~/workspace/vllm/hf_cache
export TRITON_CACHE_DIR=~/workspace/vllm/triton_cache
export XDG_CACHE_HOME=~/workspace/vllm/xdg_cache

source ~/workspace/vllm/.venv/bin/activate
cd ~/workspace/vllm

python3 scripts/check_kv_connector.py

echo ""
echo "End time: $(date)"
