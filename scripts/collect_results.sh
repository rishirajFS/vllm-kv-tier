#!/bin/bash
# Collect all benchmark results and SLURM logs after jobs complete

echo "======================================================================="
echo "Collecting Benchmark Results"
echo "======================================================================="
echo ""

# Check if jobs are still running
PHASE2_STATUS=$(squeue -j 38433031 -h -o "%T" 2>/dev/null || echo "NOT_FOUND")
PHASE3_STATUS=$(squeue -j 38433032 -h -o "%T" 2>/dev/null || echo "NOT_FOUND")

echo "Job Status:"
echo "  Phase 2 (38433031): $PHASE2_STATUS"
echo "  Phase 3 (38433032): $PHASE3_STATUS"
echo ""

if [[ "$PHASE2_STATUS" == "RUNNING" || "$PHASE3_STATUS" == "RUNNING" ]]; then
    echo "⚠️  Warning: Jobs still running! Results may be incomplete."
    echo ""
fi

# Create results archive
ARCHIVE_DIR="results_archive_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ARCHIVE_DIR"

echo "Collecting files to: $ARCHIVE_DIR"
echo ""

# Copy SLURM output logs
echo "📋 Collecting SLURM logs..."
cp slurm-38433031.out "$ARCHIVE_DIR/phase2_master.log" 2>/dev/null && echo "  ✅ Phase 2 log" || echo "  ❌ Phase 2 log not found"
cp slurm-38433032.out "$ARCHIVE_DIR/phase3_master.log" 2>/dev/null && echo "  ✅ Phase 3 log" || echo "  ❌ Phase 3 log not found"

# Copy JSON results
echo ""
echo "📊 Collecting benchmark results..."
RESULTS_COUNT=0

for file in benchmark_results/*_fixed_*.json benchmark_results/memory_sweep_*.json benchmark_results/hybrid_ablation_*.json benchmark_results/failure_modes_*.json; do
    if [[ -f "$file" ]]; then
        cp "$file" "$ARCHIVE_DIR/"
        echo "  ✅ $(basename $file)"
        RESULTS_COUNT=$((RESULTS_COUNT + 1))
    fi
done

if [[ $RESULTS_COUNT -eq 0 ]]; then
    echo "  ⚠️  No result files found yet!"
fi

# Create summary
echo ""
echo "📝 Creating summary..."

cat > "$ARCHIVE_DIR/SUMMARY.txt" <<EOF
Benchmark Results Collection
Generated: $(date)

Jobs:
  Phase 2 Master (38433031): $PHASE2_STATUS
  Phase 3 Master (38433032): $PHASE3_STATUS

Files Collected: $RESULTS_COUNT JSON files

Contents:
  - phase2_master.log: SLURM output from Phase 2 (core benchmarks)
  - phase3_master.log: SLURM output from Phase 3 (analytical scripts)
  - results_*_fixed_*.json: Re-run benchmark results with fixed instrumentation
  - memory_sweep_*.json: Memory pressure sweep (Priority 6)
  - hybrid_ablation_*.json: Hybrid weight ablation (Priority 7)
  - failure_modes_*.json: Failure mode analysis (Priority 8)

Expected Files (8 total):
  Phase 2 (5 files):
    1. results_qwen1.5b_sharegpt_fixed_*.json
    2. results_qwen3b_sharegpt_fixed_*.json
    3. results_qwen3b_msmarco_fixed_*.json
    4. results_qwen3b_humaneval_fixed_*.json
    5. results_llama1b_sharegpt_fixed_*.json

  Phase 3 (3 files):
    6. memory_sweep_*.json
    7. hybrid_ablation_*.json
    8. failure_modes_*.json

Next Steps:
  1. Validate fixes: python scripts/validate_eviction_fixes.py
  2. Download results: scp -r $(whoami)@bridges2.psc.edu:~/vllm/$ARCHIVE_DIR ./
  3. Analyze data and update documentation
EOF

echo "  ✅ Summary created"

# Create tarball for easy download
echo ""
echo "📦 Creating archive..."
tar -czf "${ARCHIVE_DIR}.tar.gz" "$ARCHIVE_DIR"
echo "  ✅ ${ARCHIVE_DIR}.tar.gz"

# Show file sizes
echo ""
echo "📏 Archive size:"
du -h "${ARCHIVE_DIR}.tar.gz"

# Quick validation
echo ""
echo "======================================================================="
echo "Quick Validation"
echo "======================================================================="
echo ""

if [[ -f "$ARCHIVE_DIR/phase2_master.log" ]]; then
    echo "Phase 2 Log (last 20 lines):"
    echo "---"
    tail -20 "$ARCHIVE_DIR/phase2_master.log"
    echo ""
fi

echo "To download results:"
echo "  scp $(whoami)@bridges2.psc.edu:~/vllm/${ARCHIVE_DIR}.tar.gz ./"
echo ""

echo "To extract locally:"
echo "  tar -xzf ${ARCHIVE_DIR}.tar.gz"
echo "  cd $ARCHIVE_DIR"
echo ""

echo "To validate eviction fixes worked:"
echo "  python scripts/validate_eviction_fixes.py"
echo ""

echo "✅ Collection complete!"
