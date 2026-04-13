#!/bin/bash
#SBATCH --job-name=extended_tests
#SBATCH --output=slurm-%j-extended-tests.out
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --account=cis250224p

# Extended Eviction Policy Experiments
#
# Explores:
# 1. Hybrid policy performance
# 2. Capacity scaling (5, 10, 20, 50 blocks)
# 3. Attention score sensitivity (2x, 10x, 100x, 1000x ratios)
# 4. Score decay impact (0.9, 0.95, 0.99, 1.0)
# 5. Workload stress test (30 system blocks, 20 capacity)

set -e

echo "======================================================================="
echo "Extended Eviction Policy Experiments"
echo "======================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

module load cuda/12.4
export HF_HOME=~/workspace/vllm/hf_cache
export TRITON_CACHE_DIR=~/workspace/vllm/triton_cache
export XDG_CACHE_HOME=~/workspace/vllm/xdg_cache

source ~/workspace/vllm/.venv/bin/activate
cd ~/workspace/vllm

echo "Running extended experiments..."
echo ""
echo "Experiments:"
echo "  1. Hybrid vs LRU vs Attention (3-way comparison)"
echo "  2. Capacity scaling (5, 10, 20, 50 blocks)"
echo "  3. Attention score sensitivity (2x to 1000x ratios)"
echo "  4. Score decay impact (0.9 to 1.0)"
echo "  5. Workload stress test (30 system blocks)"
echo ""
echo "======================================================================="
echo ""

# Run extended tests with verbose output
python3 -m pytest tests/kv_offload/test_eviction_extended.py -v -s --tb=short

EXIT_CODE=$?

echo ""
echo "======================================================================="
echo "Summary of Results"
echo "======================================================================="
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ ALL EXTENDED TESTS PASSED!"
    echo ""
    echo "Review the output above for detailed results on:"
    echo "  - Hybrid policy performance vs LRU and Attention"
    echo "  - How capacity affects the advantage"
    echo "  - Sensitivity to attention score magnitude"
    echo "  - Impact of score decay parameter"
    echo "  - Behavior under stress (more system blocks than capacity)"
    echo ""
    echo "Key insights to look for:"
    echo "  1. Does Hybrid beat both LRU and Attention?"
    echo "  2. At what capacity does attention matter most? (expect: tight capacity)"
    echo "  3. Do higher score ratios help? (expect: yes, up to saturation)"
    echo "  4. Does decay rate matter? (expect: minimal impact in this test)"
    echo "  5. Can attention handle stress? (expect: yes, better than LRU)"
else
    echo "❌ SOME TESTS FAILED (exit code: $EXIT_CODE)"
    echo ""
    echo "This is okay - we're exploring parameter space!"
    echo "Failed tests reveal interesting edge cases."
fi

echo ""
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"

# Exit with success even if some tests fail (we're exploring)
exit 0
