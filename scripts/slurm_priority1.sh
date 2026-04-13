#!/bin/bash
#SBATCH --job-name=priority1
#SBATCH --output=slurm-%j-priority1.out
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --account=cis250224p

# Priority 1: Workload Variation (Easy Wins)
#
# Tests: 11 total (~2-4 hours)
# - System:Content ratio (4 tests): 5%, 10%, 20%, 50%
# - Attention distributions (4 tests): uniform, zipfian, bimodal, random
# - Longer sequences (3 tests): 20, 50, 100 blocks

set -e

echo "======================================================================="
echo "PRIORITY 1: WORKLOAD VARIATION EXPERIMENTS"
echo "======================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

export HF_HOME=~/workspace/vllm/hf_cache
export TRITON_CACHE_DIR=~/workspace/vllm/triton_cache
export XDG_CACHE_HOME=~/workspace/vllm/xdg_cache

source ~/workspace/vllm/.venv/bin/activate
cd ~/workspace/vllm

echo "Running Priority 1 tests:"
echo "  - System:Content ratio (4 tests)"
echo "  - Attention distributions (4 tests)"
echo "  - Longer sequences (3 tests)"
echo ""

RESULTS_DIR=~/workspace/vllm/test_results
mkdir -p $RESULTS_DIR
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$RESULTS_DIR/priority1_results_${TIMESTAMP}.txt"

python3 -m pytest tests/kv_offload/test_priority1_workload_variation.py \
    -v -s --tb=short \
    2>&1 | tee "$OUTPUT_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "Extracting results..."
grep "^RESULT," "$OUTPUT_FILE" > "$RESULTS_DIR/priority1_parsed_${TIMESTAMP}.csv" 2>/dev/null || true

echo ""
echo "======================================================================="
echo "Priority 1 Complete"
echo "======================================================================="
echo ""
echo "Results saved to:"
echo "  Full output: $OUTPUT_FILE"
echo "  Parsed CSV: priority1_parsed_${TIMESTAMP}.csv"
echo ""
echo "Download with:"
echo "  scp bridges2.psc.edu:~/workspace/vllm/test_results/priority1_*.csv ."
echo ""
echo "End time: $(date)"

exit $EXIT_CODE
