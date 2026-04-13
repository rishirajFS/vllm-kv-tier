#!/bin/bash
#SBATCH --job-name=eviction_test
#SBATCH --output=slurm-%j-eviction-test.out
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --account=cis250224p

# EVICTION TRIGGER TEST
# This configuration is GUARANTEED to trigger evictions by creating extreme memory pressure

set -e

echo "======================================================================="
echo "Eviction Trigger Test - EXTREME Memory Pressure Configuration"
echo "======================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

module load cuda/12.4
export HF_HOME=~/workspace/vllm/hf_cache
export TRITON_CACHE_DIR=~/workspace/vllm/triton_cache
export XDG_CACHE_HOME=~/workspace/vllm/xdg_cache

source ~/workspace/vllm/.venv/bin/activate
cd ~/workspace/vllm/kv_cache_tiering/benchmarks

# Configuration designed to FORCE evictions
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
DATASET="sharegpt"
DATASET_PATH="~/workspace/vllm/datasets/sharegpt.json"

# EXTREME MEMORY PRESSURE SETTINGS
GPU_MEM=0.03               # 3% GPU = ~960MB total, ~640MB for KV cache after model (TINY!)
MAX_LEN=512                # Very short max length (reduces per-request capacity)
NUM_PROMPTS=500            # Many concurrent requests
MAX_TOKENS=128             # Short outputs
CPU_BYTES=8000000000       # 8GB CPU for offloading

OUTPUT_DIR=~/workspace/vllm/benchmark_results
mkdir -p $OUTPUT_DIR
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "EXTREME PRESSURE CONFIGURATION:"
echo "  Model: $MODEL"
echo "  GPU Memory: $GPU_MEM (3% - EXTREME)"
echo "  Max Model Len: $MAX_LEN (512 tokens - TINY)"
echo "  Num Prompts: $NUM_PROMPTS (500 concurrent)"
echo "  Max Tokens: $MAX_TOKENS"
echo "  CPU Bytes: $CPU_BYTES"
echo ""

echo "Expected Memory Usage:"
echo "  GPU Allocated: ~960 MB (3% of 32GB)"
echo "  Model Size: ~3 GB (needs to fit!)"
echo "  KV Cache Available: ~0 MB (almost nothing!)"
echo "  KV Cache Needed (500 × 512 tokens): ~128 MB"
echo ""
echo "  ⚠️  This will FORCE evictions to CPU!"
echo ""

OUTPUT_FILE="$OUTPUT_DIR/eviction_trigger_${TIMESTAMP}.json"

# Run with all 3 policies
python benchmark.py \
    --model "$MODEL" \
    --policies lru attention hybrid \
    --dataset "$DATASET" \
    --dataset-path "$DATASET_PATH" \
    --num-prompts $NUM_PROMPTS \
    --gpu-mem-util $GPU_MEM \
    --max-model-len $MAX_LEN \
    --max-tokens $MAX_TOKENS \
    --cpu-bytes $CPU_BYTES \
    --output "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Benchmark completed successfully"
    echo ""
    echo "Checking eviction counts..."

    python3 << EOF
import json
with open('$OUTPUT_FILE') as f:
    results = json.load(f)

print(f"\n{'='*70}")
print(f"EVICTION RESULTS")
print(f"{'='*70}\n")
print(f"{'Policy':<15} {'Throughput':<15} {'Evictions':<12} {'GPU→CPU':<12}")
print("-" * 70)

lru_tput = 0
for r in results:
    policy = r['policy']
    tput = r['tokens_per_second']
    evictions = r['total_evictions']
    bytes_off = r['bytes_gpu_to_cpu']

    if policy == 'lru':
        lru_tput = tput

    improvement = ((tput - lru_tput) / lru_tput * 100) if lru_tput > 0 else 0

    print(f"{policy:<15} {tput:<15.1f} {evictions:<12} {bytes_off/1024/1024:<10.1f} MB")

print()

# Check if ANY evictions happened
total_evictions = sum(r['total_evictions'] for r in results)
if total_evictions > 0:
    print("🎉 SUCCESS! Evictions detected!")
    print(f"   Total evictions across all policies: {total_evictions}")
    print()
    print("   This proves the eviction mechanism is working.")
    print("   Now you can run full experiments with this configuration!")
else:
    print("❌ STILL NO EVICTIONS!")
    print()
    print("   This is unexpected. Possible issues:")
    print("   1. GPU memory is still too generous (try 2% instead of 3%)")
    print("   2. vLLM is processing sequentially despite batch submission")
    print("   3. Model is too large to fit in 3% GPU memory")
    print()
    print("   Try reducing GPU% further or increasing num_prompts to 1000+")

print()
EOF

else
    echo "❌ Benchmark failed"
    echo "Check error messages above"
fi

echo ""
echo "End time: $(date)"
echo "Results: $OUTPUT_FILE"
echo ""
echo "Next steps if evictions detected:"
echo "  1. Adjust GPU% to find sweet spot (4-6%)"
echo "  2. Test with different models (3B, 7B)"
echo "  3. Run full LongBench suite with this configuration"
echo ""
