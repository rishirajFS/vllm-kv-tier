#!/bin/bash
#SBATCH --job-name=priority3
#SBATCH --output=slurm-%j-priority3.out
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --account=cis250224p

# Priority 3: Publication-Level (Comprehensive Comparison)
#
# Tests: 14 total (~8-10 hours)
# - Policy comparison (7 tests): LRU, FIFO, LFU, Random, ARC, Attention, Hybrid
# - Hyperparameter sensitivity (7 tests): Different α, β, γ combinations

set -e

echo "======================================================================="
echo "PRIORITY 3: PUBLICATION-LEVEL EXPERIMENTS"
echo "======================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

export HF_HOME=~/workspace/vllm/hf_cache
export TRITON_CACHE_DIR=~/workspace/vllm/triton_cache
export XDG_CACHE_HOME=~/workspace/vllm/xdg_cache

source ~/workspace/vllm/.venv/bin/activate
cd ~/workspace/vllm

echo "Running Priority 3 tests:"
echo "  - Policy comparison (7 tests)"
echo "  - Hyperparameter sensitivity (7 tests)"
echo ""

RESULTS_DIR=~/workspace/vllm/test_results
mkdir -p $RESULTS_DIR
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$RESULTS_DIR/priority3_results_${TIMESTAMP}.txt"

python3 -m pytest tests/kv_offload/test_priority3_publication_level.py \
    -v -s --tb=short \
    2>&1 | tee "$OUTPUT_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "Extracting results..."
grep "^RESULT," "$OUTPUT_FILE" > "$RESULTS_DIR/priority3_parsed_${TIMESTAMP}.csv" 2>/dev/null || true

echo ""
echo "======================================================================="
echo "Priority 3 Complete"
echo "======================================================================="
echo ""
echo "Results saved to:"
echo "  Full output: $OUTPUT_FILE"
echo "  Parsed CSV: priority3_parsed_${TIMESTAMP}.csv"
echo ""
echo "Key comparisons:"
echo "  - Attention vs LRU vs FIFO vs LFU vs Random vs ARC"
echo "  - Optimal hyperparameters (α, β, γ)"
echo ""
echo "Download with:"
echo "  scp bridges2.psc.edu:~/workspace/vllm/test_results/priority3_*.csv ."
echo ""
echo "End time: $(date)"

exit $EXIT_CODE
