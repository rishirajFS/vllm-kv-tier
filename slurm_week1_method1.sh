#!/bin/bash -l
#SBATCH --job-name=week1_method1
#SBATCH --output=%x_%j.out
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

# Load modules
module load cuda/12.4
module load gcc/10.2

# Explicitly use the high-speed workspace path
export WORKSPACE_DIR=/ocean/projects/cis260009p/rnagaraj/vllm
cd $WORKSPACE_DIR

# Source the Python 3.12 virtual environment to avoid SyntaxErrors
source .venv/bin/activate

export HF_HOME=$WORKSPACE_DIR/hf_cache
export TRITON_CACHE_DIR=$WORKSPACE_DIR/triton_cache
export XDG_CACHE_HOME=$WORKSPACE_DIR/xdg_cache
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the patched test script
python3 test_bypass_scheduler.py

echo ""
echo "End time: $(date)"
echo "======================================================================="
