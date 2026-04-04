#!/bin/bash
#SBATCH -A cis250224p
#SBATCH --job-name=vllm-long-context
#SBATCH --output=benchmark_results/slurm_long_context_%j.out
#SBATCH --error=benchmark_results/slurm_long_context_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER@psc.edu

# Priority 3: Long-Context Stress Test
# Tests performance scaling from 4K to 128K tokens

set -e
set -x

echo "========================================="
echo "Priority 3: Long-Context Stress Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "========================================="

# Load modules
module purge
module load cuda/12.4
module load python/3.11
module load gcc/12.2.0

# Activate virtual environment
source ~/vllm-env/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=~/workspace/hf_cache
export TMPDIR=/local/scratch/$USER
mkdir -p $TMPDIR

# GPU info
nvidia-smi

# Configuration
MODEL="Qwen/Qwen2.5-7B-Instruct"
CONTEXT_LENGTHS="4096 8192 16384 32768 65536 131072"
NUM_SAMPLES=50
POLICIES="lru attention hybrid"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="benchmark_results/long_context_qwen7b_${TIMESTAMP}.json"

echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Context Lengths: $CONTEXT_LENGTHS"
echo "  Samples per length: $NUM_SAMPLES"
echo "  Policies: $POLICIES"
echo "  Output: $OUTPUT_FILE"
echo ""

# Create output directory
mkdir -p benchmark_results

# Run benchmark
python scripts/benchmark_long_context.py \
    --model "$MODEL" \
    --context-lengths $CONTEXT_LENGTHS \
    --num-samples $NUM_SAMPLES \
    --max-new-tokens 256 \
    --gpu-mem-util 0.12 \
    --policies $POLICIES \
    --output "$OUTPUT_FILE"

# Check if output file was created
if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "✓ Benchmark completed successfully"
    echo "  Results saved to: $OUTPUT_FILE"
    echo "  File size: $(du -h $OUTPUT_FILE | cut -f1)"

    # Print summary
    echo ""
    echo "Results Summary:"
    python -c "
import json
with open('$OUTPUT_FILE') as f:
    results = json.load(f)

# Group by context length
by_context = {}
for r in results:
    ctx = r['context_length']
    if ctx not in by_context:
        by_context[ctx] = {}
    by_context[ctx][r['policy']] = r['tokens_per_second']

# Print table
print(f\"{'Context':>10} | {'LRU':>8} | {'Attention':>10} | {'Hybrid':>8} | {'Attn Impr':>10}\")
print('-' * 65)
for ctx in sorted(by_context.keys()):
    lru = by_context[ctx].get('lru', 0)
    attn = by_context[ctx].get('attention', 0)
    hyb = by_context[ctx].get('hybrid', 0)
    impr = ((attn - lru) / lru * 100) if lru > 0 else 0
    print(f\"{ctx:>10,} | {lru:>8.1f} | {attn:>10.1f} | {hyb:>8.1f} | {impr:>9.1f}%\")
"
else
    echo ""
    echo "✗ Benchmark failed - output file not created"
    exit 1
fi

echo ""
echo "Completed: $(date)"
echo "========================================="
