#!/bin/bash
#SBATCH -A cis250224p
#SBATCH --job-name=eviction-test
#SBATCH --output=eviction_test_%j.out
#SBATCH --error=eviction_test_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# Quick test to find GPU memory settings that trigger evictions

set -e

echo "========================================="
echo "Eviction Trigger Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "========================================="
echo ""

# Load modules
module purge
module load cuda/12.4
module load python/3.11

# Activate environment
source ~/vllm-env/bin/activate

# Set environment
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=~/workspace/hf_cache

# GPU info
nvidia-smi

# Run the test
echo ""
echo "Running eviction trigger test..."
echo ""

python scripts/test_eviction_trigger.py

echo ""
echo "========================================="
echo "Test completed: $(date)"
echo "========================================="
