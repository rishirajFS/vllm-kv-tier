#!/bin/bash
#SBATCH --job-name=force_eviction
#SBATCH --output=slurm-%j-force-eviction.out
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60GB
#SBATCH --account=cis250224p

# FORCE EVICTION - Aggressive Dynamic Growth Strategy
#
# Key changes from "natural pressure":
# 1. LONG OUTPUTS (max_tokens=2048, not 256!) - forces dynamic KV growth
# 2. TIGHT max_model_len (2048, not 16K!) - limits static allocation
# 3. MANY concurrent requests (200+) - saturates scheduler
# 4. Moderate GPU% (35%) - not too tight, not too loose

set -e

echo "======================================================================="
echo "FORCE EVICTION - Dynamic Growth Strategy"
echo "======================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

module load cuda/12.4
export HF_HOME=~/workspace/vllm/hf_cache
export TRITON_CACHE_DIR=~/workspace/vllm/triton_cache
export XDG_CACHE_HOME=~/workspace/vllm/xdg_cache

source ~/workspace/vllm/.venv/bin/activate
cd ~/workspace/vllm/kv_cache_tiering/benchmarks

DATASET_DIR=~/workspace/vllm/datasets
OUTPUT_DIR=~/workspace/vllm/benchmark_results
mkdir -p $OUTPUT_DIR
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "FORCE EVICTION CONFIGURATION:"
echo ""
echo "Model: Qwen/Qwen2.5-3B-Instruct"
echo "GPU Memory: 35% (NOT 45%!)"
echo "Max Model Len: 2048 tokens (TIGHT - not 16K!)"
echo "Max Output Tokens: 2048 (LONG - not 256!)"
echo "Num Prompts: 200 (HIGH concurrency)"
echo ""
echo "Why this forces evictions:"
echo "  1. Tight max_model_len (2048) limits static allocation"
echo "  2. Long outputs (2048) force DYNAMIC KV cache growth during generation"
echo "  3. As generation progresses, KV cache expands beyond initial allocation"
echo "  4. Scheduler MUST preempt/evict to free blocks for new tokens"
echo "  5. Many concurrent requests (200) keep pressure constant"
echo ""
echo "Expected behavior:"
echo "  - Initial: Scheduler admits 10-20 requests"
echo "  - During generation: KV cache grows token-by-token"
echo "  - At ~1000 tokens output: Memory pressure hits"
echo "  - Eviction triggered to make room for continued generation"
echo ""
echo "======================================================================="
echo ""

MODEL="Qwen/Qwen2.5-3B-Instruct"
GPU_MEM=0.35              # 35% - moderate
MAX_LEN=4096              # TIGHT max length (fits prompt+generation)
MAX_TOKENS=2048           # LONG outputs!
NUM_PROMPTS=200           # High concurrency

# Use ShareGPT for this test (we know it works)
DATASET="sharegpt"
DATASET_PATH="$DATASET_DIR/sharegpt.json"

if [ ! -f "$DATASET_PATH" ]; then
    echo "❌ Dataset not found: $DATASET_PATH"
    echo "   Please ensure ShareGPT dataset exists"
    exit 1
fi

OUTPUT_FILE="$OUTPUT_DIR/results_force_eviction_${TIMESTAMP}.json"

echo "Running benchmark with LONG OUTPUTS..."
echo ""

python benchmark.py \
    --model "$MODEL" \
    --policies lru attention hybrid \
    --dataset "$DATASET" \
    --dataset-path "$DATASET_PATH" \
    --num-prompts $NUM_PROMPTS \
    --gpu-mem-util $GPU_MEM \
    --max-model-len $MAX_LEN \
    --max-tokens $MAX_TOKENS \
    --output "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Benchmark completed"
    echo ""
    echo "Analyzing evictions..."

    python3 << 'EOF'
import json
import sys

try:
    with open('$OUTPUT_FILE') as f:
        results = json.load(f)

    print(f"\n{'='*70}")
    print("EVICTION ANALYSIS")
    print(f"{'='*70}\n")

    print(f"{'Policy':<12} {'Throughput':<12} {'Evictions':<12} {'Improvement'}")
    print("-" * 60)

    lru_tput = 0
    total_evictions = 0

    for r in results:
        policy = r['policy']
        tput = r['tokens_per_second']
        evictions = r.get('total_evictions', 0)
        total_evictions += evictions

        if policy == 'lru':
            lru_tput = tput
            improvement = "-"
        else:
            improvement = f"{((tput - lru_tput) / lru_tput * 100):+.1f}%" if lru_tput > 0 else "N/A"

        print(f"{policy:<12} {tput:<12.1f} {evictions:<12} {improvement}")

    print()

    if total_evictions > 500:
        print(f"🎉 SUCCESS! {total_evictions} total evictions detected!")
        print("   Dynamic growth strategy worked!")
        print()
        print("   Key insight: Long outputs force KV cache to grow dynamically,")
        print("   triggering evictions as generation progresses.")
    elif total_evictions > 0:
        print(f"⚠️  Low evictions ({total_evictions}). Try:")
        print("   - Increase max_tokens to 4096")
        print("   - Reduce GPU% to 25%")
        print("   - Increase num_prompts to 500")
    else:
        print("❌ STILL ZERO EVICTIONS!")
        print()
        print("Possible causes:")
        print("  1. Scheduler still too conservative (try GPU% = 20%)")
        print("  2. Model + initial KV still fits (try smaller model)")
        print("  3. Outputs not actually long (check ignore_eos flag)")
        print("  4. OffloadingConnector not initialized (check vLLM logs)")
        print()
        print("Next steps:")
        print("  - Check SLURM output for vLLM initialization logs")
        print("  - Verify KVTransferConfig is active")
        print("  - Try OPT-125M model (much smaller)")

    sys.exit(0 if total_evictions > 0 else 1)

except Exception as e:
    print(f"Error analyzing results: {e}")
    sys.exit(1)
EOF

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✅ Evictions detected! Ready for full experiments."
        echo ""
        echo "Next steps:"
        echo "  1. Use these settings for LongBench:"
        echo "     GPU_MEM=0.35, MAX_LEN=2048, MAX_TOKENS=2048"
        echo "  2. Run across multiple models (1.5B, 3B, 7B)"
        echo "  3. Test on different datasets"
    else
        echo ""
        echo "⚠️  Still no evictions. Need deeper investigation."
        echo ""
        echo "Debug checklist:"
        echo "  1. Check vLLM logs for KVTransferConfig initialization"
        echo "  2. Verify OffloadingConnector is created"
        echo "  3. Check if V1 engine even uses KV transfer"
        echo "  4. Consider using V0 engine instead"
    fi

else
    echo "❌ Benchmark failed"
fi

echo ""
echo "End time: $(date)"
echo "Results: $OUTPUT_FILE"
