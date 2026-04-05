#!/bin/bash
#SBATCH --job-name=longbench_7b_evict
#SBATCH --output=vllm_longbench_7b_%j.log
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60GB
#SBATCH --account=cis250224p

# =============================================================================
# LongBench-v2 on Qwen 7B — Tuned to FORCE KV cache evictions
# 
# Memory math (V100-32GB):
#   Model weights (7B fp16): ~14GB
#   gpu_mem_util=0.55 → 0.55 * 32GB = 17.6GB total
#   KV cache budget: 17.6 - 14 = ~3.6GB
#   At max_model_len=16384 (16K tokens), each block ~384KB
#   → ~9,000 max blocks before eviction kicks in
#   LongBench-v2 prompts at 16K tokens → GUARANTEES evictions
# =============================================================================

echo "======================================================================="
echo "LongBench-v2 7B Eviction Benchmark"
echo "======================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

# Environment
export HF_HOME=~/workspace/vllm/hf_cache
export TRITON_CACHE_DIR=~/workspace/vllm/triton_cache
export XDG_CACHE_HOME=~/workspace/vllm/xdg_cache

cd ~/workspace/vllm/kv_cache_tiering/benchmarks

# Activate venv
source ~/workspace/vllm/.venv/bin/activate

# Dataset paths
DATASET_DIR=~/workspace/vllm/datasets

# Config tuned for maximum eviction pressure
MODEL="Qwen/Qwen2.5-7B-Instruct"
GPU_MEM=0.55            # Leaves only ~3.6GB for KV cache — forces evictions
MAX_LEN=16384           # 16K context — fills KV pool fast
CPU_BYTES=8000000000    # 8GB CPU KV cache to absorb evicted blocks
OUTPUT_DIR=~/workspace/vllm/benchmark_results

echo "Configuration:"
echo "  Model: $MODEL"
echo "  GPU Memory: $GPU_MEM  (deliberately low to force evictions)"
echo "  Max Model Len: $MAX_LEN"
echo "  CPU KV Cache: $((CPU_BYTES / 1000000000))GB"
echo "  Output: $OUTPUT_DIR"
echo ""

# LongBench-v2 tasks ordered by context length (shortest first)
TASKS=(
    "dialogue_history:44942"
    "multi_doc_qa:72861"
    "single_doc_qa:89544"
    "long_in_context:108703"
    "structured_data:104239"
)

for task_info in "${TASKS[@]}"; do
    IFS=':' read -r task avg_len <<< "$task_info"

    DATASET_FILE="$DATASET_DIR/longbench_${task}.json"
    if [ ! -f "$DATASET_FILE" ]; then
        echo "❌ Dataset not found: $DATASET_FILE — skipping"
        continue
    fi

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_FILE="$OUTPUT_DIR/longbench_7b_${task}_${TIMESTAMP}.json"

    echo "======================================================================="
    echo "Task: $task (avg ctx ~${avg_len} words)"
    echo "======================================================================="

    python benchmark.py \
        --model "$MODEL" \
        --policies lru attention hybrid \
        --dataset "longbench_${task}" \
        --dataset-path "$DATASET_FILE" \
        --num-prompts 30 \
        --gpu-mem-util $GPU_MEM \
        --max-model-len $MAX_LEN \
        --max-tokens 256 \
        --cpu-bytes $CPU_BYTES \
        --output "$OUTPUT_FILE"

    if [ $? -eq 0 ]; then
        echo "✅ Completed: $task"
        # Quick stats
        python3 << EOF
import json
try:
    with open("$OUTPUT_FILE") as f:
        results = json.load(f)
    lru = next((r['tokens_per_second'] for r in results if r['policy'] == 'lru'), 1)
    print(f"  {'Policy':<12} {'Throughput':>12} {'vs LRU':>10} {'Evictions':>12}")
    for r in results:
        p = r['policy']
        thr = r['tokens_per_second']
        evictions = r.get('total_evictions', 0)
        imp = f"{((thr-lru)/lru*100):+.1f}%" if p != 'lru' else "baseline"
        print(f"  {p:<12} {thr:>10.1f}   {imp:>9} {evictions:>12,}")
except Exception as e:
    print(f"  Could not parse results: {e}")
EOF
    else
        echo "❌ Failed: $task"
        tail -20 "$OUTPUT_FILE" 2>/dev/null
    fi

    echo ""
done

# Skip code_repo — 330K word prompts will be heavily truncated; not representative
echo "Note: code_repo task skipped (avg 330K words, context truncated to 16K)"
echo ""
echo "======================================================================="
echo "LongBench-v2 7B Suite Complete: $(date)"
echo "======================================================================="
echo ""
echo "Results in: $OUTPUT_DIR/longbench_7b_*.json"
echo ""

# Final summary across all tasks
python3 << 'PYEOF'
import json, glob, os

base = os.path.expanduser("~/workspace/vllm/benchmark_results")
files = sorted(glob.glob(f"{base}/longbench_7b_*.json"))

if not files:
    print("No LongBench-v2 7B result files found yet.")
else:
    print("=" * 70)
    print("LONGBENCH-V2 7B FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Task':<22} {'Policy':<12} {'Throughput':>10} {'vs LRU':>9} {'Evictions':>11}")
    print("-" * 70)
    for f in files:
        task = os.path.basename(f).split("_longbench_7b_")[-1].rsplit("_", 2)[0]
        try:
            with open(f) as fp:
                results = json.load(fp)
            lru_thr = next((r['tokens_per_second'] for r in results if r['policy'] == 'lru'), 1)
            for r in results:
                p = r['policy']
                thr = r['tokens_per_second']
                evictions = r.get('total_evictions', 0)
                imp = f"{((thr-lru_thr)/lru_thr*100):+.1f}%" if p != 'lru' else "baseline"
                print(f"{task:<22} {p:<12} {thr:>10.1f} {imp:>9} {evictions:>11,}")
        except Exception as e:
            print(f"{task}: error — {e}")
PYEOF
