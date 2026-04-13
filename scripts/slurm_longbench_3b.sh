#!/bin/bash
#SBATCH --job-name=longbench_3b
#SBATCH --output=slurm-%j-longbench-3b.out
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60GB
#SBATCH --account=cis250224p

# LongBench with Qwen 3B - Natural Pressure Configuration
# Demonstrates scaling across model sizes

set -e

echo "======================================================================="
echo "LongBench - Qwen 3B (Scaling Validation)"
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

echo "Configuration:"
echo "  Model: Qwen/Qwen2.5-3B-Instruct"
echo "  GPU Memory: 30% (natural pressure for 3B)"
echo "  Max Model Len: 16384 tokens"
echo ""
echo "Memory Analysis:"
echo "  - Model weights: ~6 GB"
echo "  - GPU allocated: 9.6 GB (30% of 32GB)"
echo "  - KV cache @ 15K tokens: ~7.8 GB needed"
echo "  - Total needed: 13.8 GB"
echo "  - Must evict ~4.2 GB to CPU! ✓"
echo ""

MODEL="Qwen/Qwen2.5-3B-Instruct"
GPU_MEM=0.30           # 30% for 3B model
MAX_LEN=16384
MAX_TOKENS=256
NUM_PROMPTS=100

# Same tasks as 7B for comparison
TASKS=(
    "narrative_qa"
    "qasper"
    "multi_news"
)

for task in "${TASKS[@]}"; do
    echo ""
    echo "Task: $task"
    echo ""

    DATASET_FILE="$DATASET_DIR/longbench_${task}.json"

    if [ ! -f "$DATASET_FILE" ]; then
        echo "⚠️  Dataset not found: $DATASET_FILE"
        echo "   Run slurm_longbench_natural.sh first to download datasets"
        continue
    fi

    OUTPUT_FILE="$OUTPUT_DIR/results_longbench_3b_${task}_${TIMESTAMP}.json"

    python benchmark.py \
        --model "$MODEL" \
        --policies lru attention hybrid \
        --dataset "longbench_${task}" \
        --dataset-path "$DATASET_FILE" \
        --num-prompts $NUM_PROMPTS \
        --gpu-mem-util $GPU_MEM \
        --max-model-len $MAX_LEN \
        --max-tokens $MAX_TOKENS \
        --output "$OUTPUT_FILE"

    if [ $? -eq 0 ]; then
        echo "✅ Completed: $task"

        python3 << EOF
import json
with open("$OUTPUT_FILE") as f:
    results = json.load(f)

lru_tput = next((r['tokens_per_second'] for r in results if r['policy'] == 'lru'), 0)

for r in results:
    policy = r['policy']
    tput = r['tokens_per_second']
    evictions = r.get('total_evictions', 0)
    improvement = ((tput - lru_tput) / lru_tput * 100) if lru_tput > 0 else 0

    print(f"  {policy:<12} {tput:>8.1f} tok/s  {evictions:>6} evictions  {improvement:>+6.1f}%")
EOF

    else
        echo "❌ Failed: $task"
    fi

    echo ""
done

echo "Complete! Results: $OUTPUT_DIR/results_longbench_3b_*_${TIMESTAMP}.json"
