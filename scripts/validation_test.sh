#!/bin/bash
#SBATCH --job-name=eviction_validation
#SBATCH --output=slurm-%j-validation.out
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --account=cis250224p

# Quick Validation Test - Confirm eviction fixes work with correct config
# Expected: total_evictions > 0
# Duration: ~20-30 minutes

set -e

echo "======================================================================="
echo "Eviction Fix Validation Test"
echo "======================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load modules
module load cuda/12.4
export HF_HOME=~/workspace/vllm/hf_cache
export TRITON_CACHE_DIR=~/workspace/vllm/triton_cache
export XDG_CACHE_HOME=~/workspace/vllm/xdg_cache

# Activate environment
source ~/workspace/vllm/.venv/bin/activate

# Navigate to benchmark directory
cd ~/workspace/vllm/kv_cache_tiering/benchmarks

# Configuration: AGGRESSIVE memory pressure to force evictions
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
DATASET="sharegpt"
DATASET_PATH="~/workspace/vllm/datasets/sharegpt.json"
GPU_MEM=0.08           # 8% GPU - very aggressive
MAX_LEN=2048           # 2K max length - force memory pressure
NUM_PROMPTS=50         # Small batch for quick test
MAX_TOKENS=512         # Short outputs

OUTPUT_DIR=~/workspace/vllm/benchmark_results
mkdir -p $OUTPUT_DIR
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$OUTPUT_DIR/validation_test_${TIMESTAMP}.json"

echo "Configuration:"
echo "  Model: $MODEL"
echo "  GPU Memory: $GPU_MEM (8% - aggressive)"
echo "  Max Model Len: $MAX_LEN (2K - force pressure)"
echo "  Num Prompts: $NUM_PROMPTS"
echo "  Max Tokens: $MAX_TOKENS"
echo "  Dataset: $DATASET"
echo ""

echo "Running validation benchmark..."
python benchmark.py \
    --model "$MODEL" \
    --eviction-policy lru \
    --dataset "$DATASET" \
    --dataset-path "$DATASET_PATH" \
    --num-prompts $NUM_PROMPTS \
    --gpu-memory-utilization $GPU_MEM \
    --max-model-len $MAX_LEN \
    --max-tokens $MAX_TOKENS \
    --output "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Validation test completed successfully"
    echo ""
    echo "Checking eviction count..."

    # Extract eviction count
    EVICTIONS=$(python3 -c "
import json
with open('$OUTPUT_FILE') as f:
    data = json.load(f)
    evictions = data[0].get('total_evictions', 0)
    print(evictions)
")

    echo "Total evictions: $EVICTIONS"
    echo ""

    if [ "$EVICTIONS" -gt 0 ]; then
        echo "🎉 SUCCESS! Eviction fixes are working!"
        echo "   Evictions detected: $EVICTIONS"
        echo ""
        echo "Next steps:"
        echo "  1. Submit LongBench job: sbatch scripts/slurm_longbench.sh"
        echo "  2. Submit Quality validation: sbatch scripts/slurm_quality_validation.sh"
        echo "  3. Run both in parallel if cluster capacity allows"
    else
        echo "⚠️  WARNING: Zero evictions detected"
        echo "   This may indicate:"
        echo "   - GPU memory still too generous (try 4-6%)"
        echo "   - max_model_len still too large (try 1024)"
        echo "   - Dataset prompts too short"
        echo ""
        echo "Debug info:"
        cat "$OUTPUT_FILE"
    fi
else
    echo "❌ Validation test failed"
    echo "Check error messages above"
fi

echo ""
echo "End time: $(date)"
echo "Results: $OUTPUT_FILE"
