#!/bin/bash
#SBATCH --job-name=priority2
#SBATCH --output=slurm-%j-priority2.out
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --account=cis250224p

# Priority 2: Performance Metrics (More Convincing)
#
# Tests: 8 total (~4-6 hours)
# - Recomputation cost (4 tests): 16, 32, 64, 128 tokens/block
# - Multi-turn conversation (3 tests): 5, 10, 20 turns
# - Cumulative benefit (1 test): 100 requests

set -e

echo "======================================================================="
echo "PRIORITY 2: PERFORMANCE METRICS EXPERIMENTS"
echo "======================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

export HF_HOME=~/workspace/vllm/hf_cache
export TRITON_CACHE_DIR=~/workspace/vllm/triton_cache
export XDG_CACHE_HOME=~/workspace/vllm/xdg_cache

source ~/workspace/vllm/.venv/bin/activate
cd ~/workspace/vllm

echo "Running Priority 2 tests:"
echo "  - Recomputation cost savings (4 tests)"
echo "  - Multi-turn conversation (3 tests)"
echo "  - Cumulative benefit (1 test)"
echo ""

RESULTS_DIR=~/workspace/vllm/test_results
mkdir -p $RESULTS_DIR
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$RESULTS_DIR/priority2_results_${TIMESTAMP}.txt"

python3 -m pytest tests/kv_offload/test_priority2_performance_metrics.py \
    -v -s --tb=short \
    2>&1 | tee "$OUTPUT_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "Extracting results..."
grep "^RESULT," "$OUTPUT_FILE" > "$RESULTS_DIR/priority2_parsed_${TIMESTAMP}.csv" 2>/dev/null || true

echo ""
echo "======================================================================="
echo "Priority 2 Complete"
echo "======================================================================="
echo ""
echo "Results saved to:"
echo "  Full output: $OUTPUT_FILE"
echo "  Parsed CSV: priority2_parsed_${TIMESTAMP}.csv"
echo ""
echo "Key metrics:"
echo "  - Recomputation savings (% reduction)"
echo "  - Multi-turn hit rate (% cache hits)"
echo "  - Cumulative benefit (total hits over time)"
echo ""
echo "Download with:"
echo "  scp bridges2.psc.edu:~/workspace/vllm/test_results/priority2_*.csv ."
echo ""
echo "End time: $(date)"

exit $EXIT_CODE
