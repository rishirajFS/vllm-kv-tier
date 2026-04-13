#!/bin/bash
#SBATCH --job-name=quality_kv
#SBATCH --output=slurm-%j-quality.out
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --account=cis250224p

# Quality Validation Suite
set -e

echo "======================================================================="
echo "Quality Validation Suite - KV Cache Tiering"
echo "======================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

module load cuda/12.4
export HF_HOME=~/workspace/vllm/hf_cache
export TRITON_CACHE_DIR=~/workspace/vllm/triton_cache
export XDG_CACHE_HOME=~/workspace/vllm/xdg_cache

source ~/workspace/vllm/.venv/bin/activate
pip install rouge-score bert-score --quiet

cd ~/workspace/vllm
DATASET_DIR=~/workspace/vllm/datasets
OUTPUT_DIR=~/workspace/vllm/benchmark_results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL="Qwen/Qwen2.5-3B-Instruct"

# Setup datasets
python scripts/setup_quality_datasets.py --output $DATASET_DIR

# Run quality validation
python scripts/run_quality_validation.py \
    --model "$MODEL" \
    --dataset "$DATASET_DIR/quality_subset_sharegpt.json" \
    --policies lru attention hybrid \
    --max-samples 100 \
    --output "$OUTPUT_DIR/quality_validation_${TIMESTAMP}.json"

echo "Results: $OUTPUT_DIR/quality_validation_${TIMESTAMP}.json"
