#!/bin/bash
#SBATCH --job-name=longbench_kv
#SBATCH --output=slurm-%j-longbench.out
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --account=cis250224p

# LongBench Benchmark Suite for KV Cache Tiering
# Tests long-context performance (3K-18K tokens) where eviction benefits are highest
#
# Expected timeline: ~8 hours
#   - 4 tasks × 3 policies × 200 samples each
#   - Longer contexts = slower execution
#
# Expected results: 6% → 33% improvement as context length increases

set -e

echo "======================================================================="
echo "LongBench Benchmark Suite - KV Cache Tiering"
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

# Dataset paths
DATASET_DIR=~/workspace/vllm/datasets

# Ensure LongBench datasets are downloaded
echo "Checking LongBench datasets..."
if [ ! -f "$DATASET_DIR/longbench_qasper.json" ]; then
    echo "Downloading and converting LongBench datasets..."
    cd ~/workspace/vllm
    python scripts/setup_longbench.py --output $DATASET_DIR --max-samples 200
    cd ~/workspace/vllm/kv_cache_tiering/benchmarks
fi

# Model to use (Qwen 3B or 7B recommended for long contexts)
MODEL="Qwen/Qwen2.5-3B-Instruct"

# GPU memory settings (aggressive to force evictions)
GPU_MEM=0.12
MAX_LEN=16384  # Support up to 16K contexts

# Output directory
OUTPUT_DIR=~/workspace/vllm/benchmark_results
mkdir -p $OUTPUT_DIR

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Configuration:"
echo "  Model: $MODEL"
echo "  GPU Memory: $GPU_MEM"
echo "  Max Model Len: $MAX_LEN"
echo "  Output: $OUTPUT_DIR"
echo ""

# LongBench tasks to run (ordered by context length)
TASKS=(
    "multi_news:2113"      # Shortest (2K avg)
    "qasper:3619"          # Medium-short (3.6K avg)
    "hotpotqa:9151"        # Medium-long (9K avg)
    "narrative_qa:18409"   # Longest (18K avg)
)

# Run each task
for task_info in "${TASKS[@]}"; do
    # Split task name and avg length
    IFS=':' read -r task avg_len <<< "$task_info"

    echo "======================================================================="
    echo "Task: $task (avg ~${avg_len} tokens)"
    echo "======================================================================="
    echo ""

    # Check if dataset exists
    DATASET_FILE="$DATASET_DIR/longbench_${task}.json"
    if [ ! -f "$DATASET_FILE" ]; then
        echo "❌ Dataset not found: $DATASET_FILE"
        echo "   Skipping..."
        continue
    fi

    OUTPUT_FILE="$OUTPUT_DIR/results_${task}_${TIMESTAMP}.json"

    echo "Running benchmark..."
    echo "  Dataset: $DATASET_FILE"
    echo "  Output: $OUTPUT_FILE"
    echo ""

    # Run benchmark with all 3 policies
    python benchmark.py \
        --model "$MODEL" \
        --eviction-policy lru attention hybrid \
        --dataset "longbench_${task}" \
        --dataset-path "$DATASET_FILE" \
        --num-prompts 200 \
        --gpu-memory-utilization $GPU_MEM \
        --max-model-len $MAX_LEN \
        --max-tokens 256 \
        --output "$OUTPUT_FILE"

    if [ $? -eq 0 ]; then
        echo "✅ Completed: $task"

        # Show quick stats
        echo ""
        echo "Quick results:"
        python3 << EOF
import json
with open("$OUTPUT_FILE") as f:
    results = json.load(f)

for r in results:
    policy = r['policy']
    tput = r['tokens_per_second']
    evictions = r.get('total_evictions', 0)
    print(f"  {policy:10s}: {tput:6.1f} tok/s, {evictions:4d} evictions")
EOF
    else
        echo "❌ Failed: $task"
    fi

    echo ""
done

echo "======================================================================="
echo "LongBench Suite Complete"
echo "======================================================================="
echo "End time: $(date)"
echo ""

# Generate summary
echo "Summary of all results:"
echo ""

for task_info in "${TASKS[@]}"; do
    IFS=':' read -r task avg_len <<< "$task_info"
    result_file="$OUTPUT_DIR/results_${task}_${TIMESTAMP}.json"

    if [ -f "$result_file" ]; then
        echo "Task: $task (avg ${avg_len} tokens)"

        python3 << EOF
import json
try:
    with open("$result_file") as f:
        results = json.load(f)

    # Find LRU baseline
    lru_tput = next((r['tokens_per_second'] for r in results if r['policy'] == 'lru'), 0)

    print(f"  {'Policy':<12} {'Throughput':<12} {'Improvement':<12} {'Evictions'}")
    print(f"  {'-'*50}")

    for r in results:
        policy = r['policy']
        tput = r['tokens_per_second']
        evictions = r.get('total_evictions', 0)

        if lru_tput > 0:
            improvement = ((tput - lru_tput) / lru_tput) * 100
            print(f"  {policy:<12} {tput:<12.1f} {improvement:>+6.1f}%      {evictions:5d}")
        else:
            print(f"  {policy:<12} {tput:<12.1f} {'N/A':<12} {evictions:5d}")

except Exception as e:
    print(f"  Error reading results: {e}")
EOF
        echo ""
    fi
done

echo "All results saved to: $OUTPUT_DIR/results_*_${TIMESTAMP}.json"
echo ""
echo "Next steps:"
echo "  1. Download results: scp \$USER@bridges2.psc.edu:$OUTPUT_DIR/results_*_${TIMESTAMP}.json ./"
echo "  2. Validate: python scripts/validate_eviction_fixes.py"
echo "  3. Create visualizations showing improvement vs context length"
echo ""
