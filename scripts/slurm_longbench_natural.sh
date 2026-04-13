#!/bin/bash
#SBATCH --job-name=longbench_natural
#SBATCH --output=slurm-%j-longbench-natural.out
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60GB
#SBATCH --account=cis250224p

# LongBench with NATURAL Memory Pressure (RECOMMENDED APPROACH)
#
# Strategy: Use realistic GPU settings (45%) with long sequences (16K tokens)
# Long contexts create NATURAL evictions without artificial constraints
#
# Expected: 15-25% throughput improvement from content-aware eviction

set -e

echo "======================================================================="
echo "LongBench - Natural Memory Pressure Configuration"
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
OUTPUT_DIR=~/workspace/vllm/benchmark_results
mkdir -p $OUTPUT_DIR

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "======================================================================="
echo "NATURAL PRESSURE CONFIGURATION"
echo "======================================================================="
echo ""
echo "Model: Qwen/Qwen2.5-7B-Instruct"
echo "GPU Memory: 45% (REALISTIC - not extreme!)"
echo "Max Model Len: 16384 tokens (ALLOW LONG SEQUENCES)"
echo "Dataset: LongBench (avg 12K-15K tokens naturally)"
echo ""
echo "Why this works:"
echo "  - Model weights: ~14 GB (Qwen 7B)"
echo "  - GPU allocated: 14.4 GB (45% of 32GB)"
echo "  - KV cache @ 15K tokens: ~7.8 GB needed"
echo "  - Total needed: 21.8 GB"
echo "  - Must evict ~7.4 GB to CPU! ✓"
echo ""
echo "Expected Results:"
echo "  - Evictions: 800-1500 per run"
echo "  - LRU: ~450 tok/s (baseline)"
echo "  - Attention: ~550 tok/s (+22% improvement!) ⭐"
echo "  - Hybrid: ~540 tok/s (+20% improvement!)"
echo ""
echo "======================================================================="

# Model configuration
MODEL="Qwen/Qwen2.5-7B-Instruct"
GPU_MEM=0.45           # 45% - REALISTIC setting
MAX_LEN=16384          # Allow long sequences
MAX_TOKENS=256         # Output length
NUM_PROMPTS=100        # 100 samples (each 12K-15K tokens)

# LongBench tasks (pick 2-3 high-value ones)
TASKS=(
    "narrative_qa"       # Book understanding (18K avg)
    "qasper"            # Scientific papers (3.6K avg)
    "multi_news"        # Multi-doc summarization (2K avg)
)

# Run each task
for task in "${TASKS[@]}"; do
    echo ""
    echo "======================================================================="
    echo "Task: $task"
    echo "======================================================================="
    echo ""

    # Check if dataset exists
    DATASET_FILE="$DATASET_DIR/longbench_${task}.json"
    if [ ! -f "$DATASET_FILE" ]; then
        echo "⚠️  Dataset not found: $DATASET_FILE"
        echo "   Please sync the raw .json payloads to the cluster first!"
        continue
    fi

    OUTPUT_FILE="$OUTPUT_DIR/results_longbench_${task}_${TIMESTAMP}.json"

    echo "Running benchmark..."
    echo "  Dataset: $DATASET_FILE"
    echo "  Output: $OUTPUT_FILE"
    echo ""

    # Run benchmark with all 3 policies
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
        echo ""
        echo "✅ Completed: $task"
        echo ""
        echo "Quick results:"

        python3 << EOF
import json

try:
    with open("$OUTPUT_FILE") as f:
        results = json.load(f)

    print(f"\n{'Policy':<12} {'Throughput':<12} {'Evictions':<10} {'Improvement':<12}")
    print("-" * 50)

    lru_tput = next((r['tokens_per_second'] for r in results if r['policy'] == 'lru'), 0)

    for r in results:
        policy = r['policy']
        tput = r['tokens_per_second']
        evictions = r.get('total_evictions', 0)

        improvement = ((tput - lru_tput) / lru_tput * 100) if lru_tput > 0 else 0
        improvement_str = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"

        print(f"{policy:<12} {tput:<12.1f} {evictions:<10} {improvement_str:<12}")

    total_evictions = sum(r.get('total_evictions', 0) for r in results)

    print()
    if total_evictions > 500:
        print(f"🎉 SUCCESS! {total_evictions} total evictions detected!")
        print("   Natural memory pressure is working!")
    elif total_evictions > 0:
        print(f"⚠️  Low evictions ({total_evictions}). Context may be shorter than expected.")
    else:
        print(f"❌ ZERO evictions! Long sequences not creating pressure.")

except Exception as e:
    print(f"Error reading results: {e}")
EOF

    else
        echo "❌ Failed: $task"
    fi

    echo ""
done

echo "======================================================================="
echo "LongBench Natural Pressure - Complete"
echo "======================================================================="
echo "End time: $(date)"
echo ""

# Generate summary across all tasks
echo "Summary of Results:"
echo ""

python3 << EOF
import json
from pathlib import Path
from collections import defaultdict

results_dir = Path('$OUTPUT_DIR')
pattern = f"results_longbench_*_{TIMESTAMP}.json"

all_results = defaultdict(list)

for result_file in results_dir.glob(f"results_longbench_*_$TIMESTAMP.json"):
    task = result_file.stem.replace(f'results_longbench_', '').replace(f'_$TIMESTAMP', '')

    try:
        with open(result_file) as f:
            data = json.load(f)

        for r in data:
            all_results[r['policy']].append({
                'task': task,
                'throughput': r['tokens_per_second'],
                'evictions': r.get('total_evictions', 0)
            })
    except Exception as e:
        print(f"Warning: couldn't read {result_file}: {e}")

if all_results:
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS")
    print(f"{'='*70}\n")

    for policy in ['lru', 'attention', 'hybrid']:
        if policy not in all_results:
            continue

        results = all_results[policy]
        avg_tput = sum(r['throughput'] for r in results) / len(results)
        total_evictions = sum(r['evictions'] for r in results)

        print(f"{policy.upper()}")
        print(f"  Avg Throughput: {avg_tput:.1f} tok/s")
        print(f"  Total Evictions: {total_evictions}")
        print()

    # Calculate improvement
    if 'lru' in all_results and 'attention' in all_results:
        lru_avg = sum(r['throughput'] for r in all_results['lru']) / len(all_results['lru'])
        attn_avg = sum(r['throughput'] for r in all_results['attention']) / len(all_results['attention'])
        improvement = ((attn_avg - lru_avg) / lru_avg) * 100

        print(f"🎯 KEY FINDING: Attention-weighted achieves {improvement:+.1f}% improvement!")

        if improvement > 15:
            print("   ⭐ EXCELLENT! This is publishable!")
        elif improvement > 8:
            print("   ✅ GOOD! Shows clear benefit.")
        elif improvement > 0:
            print("   ⚠️  Modest improvement. May need tuning.")
        else:
            print("   ❌ Attention performing worse. Check eviction counts.")
else:
    print("No results found to summarize.")
EOF

echo ""
echo "All results saved to: $OUTPUT_DIR/results_longbench_*_${TIMESTAMP}.json"
echo ""
echo "Next steps:"
echo "  1. Download results: scp \$USER@bridges2.psc.edu:$OUTPUT_DIR/results_longbench_*_${TIMESTAMP}.json ./"
echo "  2. Create visualizations (context length vs improvement)"
echo "  3. Run quality validation (ROUGE-L, BERTScore)"
echo ""
