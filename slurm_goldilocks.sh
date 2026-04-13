#!/bin/bash
#SBATCH --job-name=goldilocks
#SBATCH --output=week1_goldilocks_%j.out
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --account=cis250224p

# Week 1 - Method 2.5: Goldilocks Zone (55% GPU)
#
# Finding the sweet spot between:
# - Method 1 (80% GPU): Scheduler blocked (0 evictions)
# - Method 2 (35% GPU): Too tight (crashed)
#
# Strategy:
# - 55% GPU memory (Goldilocks zone)
# - 24 concurrent requests (max pressure)
# - 10K tokens per request × 24 = 240K total tokens

set -e

echo "======================================================================="
echo "WEEK 1 - METHOD 2.5: Goldilocks Zone (55% GPU)"
echo "======================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

export HF_HOME=~/workspace/vllm/hf_cache
export TRITON_CACHE_DIR=~/workspace/vllm/triton_cache
export XDG_CACHE_HOME=~/workspace/vllm/xdg_cache

source ~/workspace/vllm/.venv/bin/activate
cd ~/workspace/vllm

echo "Configuration:"
echo "  - GPU memory: 55% (sweet spot)"
echo "  - Concurrent requests: 24 (max pressure)"
echo "  - Tokens per request: ~10,000"
echo "  - Total workload: ~240,000 tokens"
echo ""

python test_method2_goldilocks.py

EXIT_CODE=$?

echo ""
echo "======================================================================="
echo "Goldilocks Test Complete"
echo "======================================================================="
echo ""
echo "Exit code: $EXIT_CODE"
echo "  0 = Success (>1000 evictions)"
echo "  1 = Failure (<1000 evictions)"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ GOLDILOCKS ZONE FOUND!"
    echo "   Proceed with Week 2 LongBench using 55% GPU config"
else
    echo "⚠️  Goldilocks test failed"
    echo "   Next steps: Custom scheduler or Path B (workshop paper)"
fi

echo ""
echo "End time: $(date)"

exit $EXIT_CODE
