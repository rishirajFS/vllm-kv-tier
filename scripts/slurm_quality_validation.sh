#!/bin/bash
#SBATCH -A cis250224p
#SBATCH --job-name=vllm-quality
#SBATCH --output=benchmark_results/slurm_quality_%j.out
#SBATCH --error=benchmark_results/slurm_quality_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER@psc.edu

# Priority 5: Quality Validation
# Proves that KV cache tiering doesn't hurt generation quality

set -e
set -x

echo "========================================="
echo "Priority 5: Quality Validation"
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

# Install quality metrics packages if needed
pip install -q rouge-score bert-score 2>/dev/null || true

# Configuration
MODEL="Qwen/Qwen2.5-7B-Instruct"
DATASET="sharegpt"
DATASET_PATH="$HOME/workspace/vllm/datasets/sharegpt.json"
NUM_SAMPLES=200
MAX_TOKENS=256
POLICIES="lru attention hybrid"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="benchmark_results/quality_qwen7b_${TIMESTAMP}.json"

echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET ($DATASET_PATH)"
echo "  Samples: $NUM_SAMPLES"
echo "  Max tokens: $MAX_TOKENS"
echo "  Policies: $POLICIES"
echo "  Output: $OUTPUT_FILE"
echo ""

# Check dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "⚠ Warning: Dataset not found at $DATASET_PATH"
    echo "  Will use synthetic prompts instead"
    DATASET_PATH=""
fi

# Create output directory
mkdir -p benchmark_results

# Run quality validation
if [ -z "$DATASET_PATH" ]; then
    # No dataset - use synthetic prompts
    python scripts/validate_output_quality.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --num-samples $NUM_SAMPLES \
        --max-tokens $MAX_TOKENS \
        --policies $POLICIES \
        --output "$OUTPUT_FILE"
else
    # Use real dataset
    python scripts/validate_output_quality.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --num-samples $NUM_SAMPLES \
        --max-tokens $MAX_TOKENS \
        --policies $POLICIES \
        --output "$OUTPUT_FILE"
fi

# Check if output file was created
if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "✓ Quality validation completed successfully"
    echo "  Results saved to: $OUTPUT_FILE"
    echo "  File size: $(du -h $OUTPUT_FILE | cut -f1)"

    # Print summary
    echo ""
    echo "Quality Metrics Summary:"
    python -c "
import json
with open('$OUTPUT_FILE') as f:
    results = json.load(f)

print(f\"{'Policy':>10} | {'ROUGE-L':>8} | {'BERTScore':>10} | {'Exact Match':>11} | {'Status':>8}\")
print('-' * 60)
for r in results:
    policy = r['policy']
    rouge = r['rouge_l_f1']
    bert = r['bertscore_f1']
    exact = r['exact_match_rate']
    status = '✓ PASS' if rouge > 0.95 else ('⚠ CHECK' if rouge > 0.90 else '✗ FAIL')
    print(f\"{policy:>10} | {rouge:>8.4f} | {bert:>10.4f} | {exact:>10.1%} | {status:>8}\")

# Check for quality degradation
min_rouge = min(r['rouge_l_f1'] for r in results)
if min_rouge < 0.90:
    print()
    print('⚠ WARNING: Quality degradation detected (ROUGE-L < 0.90)')
    print('  This may indicate a bug in the tiering implementation.')
elif min_rouge > 0.98:
    print()
    print('✓ Excellent quality preservation (ROUGE-L > 0.98)')
    print('  KV cache tiering does not hurt generation quality.')
"
else
    echo ""
    echo "✗ Quality validation failed - output file not created"
    exit 1
fi

echo ""
echo "Completed: $(date)"
echo "========================================="
