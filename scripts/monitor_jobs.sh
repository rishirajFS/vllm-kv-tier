#!/bin/bash
# Monitor the two master SLURM jobs in real-time

PHASE2_JOB=38433031
PHASE3_JOB=38433032

echo "======================================================================="
echo "Monitoring Master Jobs"
echo "======================================================================="
echo ""

# Job status
echo "Current Status:"
squeue -j $PHASE2_JOB,$PHASE3_JOB -o "%.18i %.12P %.20j %.8u %.2t %.10M %.6D %R" 2>/dev/null || echo "Jobs not in queue (completed or not found)"
echo ""

# Detailed info
echo "Detailed Info:"
echo ""
echo "Phase 2 Master (Job $PHASE2_JOB):"
scontrol show job $PHASE2_JOB 2>/dev/null | grep -E "JobState|RunTime|TimeLimit|StartTime|EndTime|Reason" || echo "  Not found"
echo ""

echo "Phase 3 Master (Job $PHASE3_JOB):"
scontrol show job $PHASE3_JOB 2>/dev/null | grep -E "JobState|RunTime|TimeLimit|StartTime|EndTime|Reason" || echo "  Not found"
echo ""

# Check output files
echo "======================================================================="
echo "Output Files"
echo "======================================================================="
echo ""

if [[ -f "slurm-${PHASE2_JOB}.out" ]]; then
    PHASE2_SIZE=$(wc -l < "slurm-${PHASE2_JOB}.out")
    PHASE2_RECENT=$(tail -3 "slurm-${PHASE2_JOB}.out" | head -1)
    echo "Phase 2 Log: $PHASE2_SIZE lines"
    echo "  Recent: $PHASE2_RECENT"
else
    echo "Phase 2 Log: Not created yet"
fi

if [[ -f "slurm-${PHASE3_JOB}.out" ]]; then
    PHASE3_SIZE=$(wc -l < "slurm-${PHASE3_JOB}.out")
    PHASE3_RECENT=$(tail -3 "slurm-${PHASE3_JOB}.out" | head -1)
    echo "Phase 3 Log: $PHASE3_SIZE lines"
    echo "  Recent: $PHASE3_RECENT"
else
    echo "Phase 3 Log: Not created yet"
fi

echo ""

# Check benchmark results
echo "======================================================================="
echo "Benchmark Results"
echo "======================================================================="
echo ""

RESULTS_FIXED=$(ls benchmark_results/*_fixed_*.json 2>/dev/null | wc -l)
RESULTS_SWEEP=$(ls benchmark_results/memory_sweep_*.json 2>/dev/null | wc -l)
RESULTS_ABLATION=$(ls benchmark_results/hybrid_ablation_*.json 2>/dev/null | wc -l)
RESULTS_FAILURE=$(ls benchmark_results/failure_modes_*.json 2>/dev/null | wc -l)

TOTAL_RESULTS=$((RESULTS_FIXED + RESULTS_SWEEP + RESULTS_ABLATION + RESULTS_FAILURE))

echo "Phase 2 (Core Benchmarks): $RESULTS_FIXED / 5 expected"
echo "Phase 3 (Analytical):"
echo "  Memory Sweep: $RESULTS_SWEEP / 1 expected"
echo "  Hybrid Ablation: $RESULTS_ABLATION / 1 expected"
echo "  Failure Modes: $RESULTS_FAILURE / 1 expected"
echo ""
echo "Total: $TOTAL_RESULTS / 8 expected"

if [[ $TOTAL_RESULTS -eq 8 ]]; then
    echo "✅ All results generated!"
    echo ""
    echo "Next steps:"
    echo "  bash scripts/collect_results.sh"
elif [[ $TOTAL_RESULTS -gt 0 ]]; then
    echo "⏳ Partial results (jobs still running)"
else
    echo "⏳ No results yet (waiting for jobs to start)"
fi

echo ""
echo "======================================================================="
echo "Commands"
echo "======================================================================="
echo ""
echo "Watch Phase 2 live output:"
echo "  tail -f slurm-${PHASE2_JOB}.out"
echo ""
echo "Watch Phase 3 live output:"
echo "  tail -f slurm-${PHASE3_JOB}.out"
echo ""
echo "Cancel jobs if needed:"
echo "  scancel $PHASE2_JOB $PHASE3_JOB"
echo ""
echo "Re-run this monitor:"
echo "  bash scripts/monitor_jobs.sh"
echo ""
