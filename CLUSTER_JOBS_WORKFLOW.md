# Cluster Jobs Workflow - Jobs 38433031 & 38433032

## Your Smart Consolidation Strategy ✅

Instead of 8 individual jobs fighting for scheduler priority, you consolidated into **2 master jobs** that will backfill overnight:

| Job ID | Type | Duration | Runs | Purpose |
|--------|------|----------|------|---------|
| **38433031** | Phase 2 Master | 6 hours | 5 benchmarks | Core ShareGPT/MS-MARCO/HumanEval verification |
| **38433032** | Phase 3 Master | 6 hours | 3 scripts | Memory sweep, hybrid ablation, failure modes |

**Benefit**: Two 6-hour GPU blocks backfill ~10x faster than 8 separate 1-hour jobs in the PD queue.

---

## Job Monitoring

### Check Job Status

```bash
ssh bridges2.psc.edu

# Quick status
squeue -u $USER

# Detailed info
bash scripts/monitor_jobs.sh
```

### Watch Live Output

```bash
# Phase 2 (Core benchmarks)
tail -f slurm-38433031.out

# Phase 3 (Analytical scripts)
tail -f slurm-38433032.out
```

### Expected Progress

**Phase 2 Master (Job 38433031)** - Sequential execution:
```
[Hour 0-1] Qwen 1.5B ShareGPT (3 policies)
[Hour 1-2] Qwen 3B ShareGPT (3 policies)
[Hour 2-3] Qwen 3B MS-MARCO (3 policies)
[Hour 3-4] Qwen 3B HumanEval (3 policies)
[Hour 4-5] Llama 3.2-1B ShareGPT (3 policies)
[Hour 5-6] Cleanup and validation
```

**Phase 3 Master (Job 38433032)** - Sequential execution:
```
[Hour 0-2] Memory Pressure Sweep (6 GPU configs × 3 policies = 18 runs)
[Hour 2-4] Hybrid Ablation (9 alpha values × varying beta/gamma = 27 runs)
[Hour 4-6] Failure Modes (6 scenarios × 3 policies = 18 runs)
```

---

## Expected Output Files

### Phase 2 Outputs (5 JSON files)

```
benchmark_results/
├── results_qwen1.5b_sharegpt_fixed_YYYYMMDD_HHMMSS.json
├── results_qwen3b_sharegpt_fixed_YYYYMMDD_HHMMSS.json
├── results_qwen3b_msmarco_fixed_YYYYMMDD_HHMMSS.json
├── results_qwen3b_humaneval_fixed_YYYYMMDD_HHMMSS.json
└── results_llama1b_sharegpt_fixed_YYYYMMDD_HHMMSS.json
```

**Critical validation**: All should have `"total_evictions" > 0`

### Phase 3 Outputs (3 JSON files)

```
benchmark_results/
├── memory_sweep_YYYYMMDD_HHMMSS.json
├── hybrid_ablation_YYYYMMDD_HHMMSS.json
└── failure_modes_YYYYMMDD_HHMMSS.json
```

**Purpose**: Deep analysis for midterm report charts and tables

### SLURM Logs

```
slurm-38433031.out  (Phase 2 master log)
slurm-38433032.out  (Phase 3 master log)
```

---

## Results Collection (Run After Jobs Complete)

### 1. Collect All Results

```bash
ssh bridges2.psc.edu
cd ~/vllm

# Wait for both jobs to complete
squeue -j 38433031,38433032

# Collect results into archive
bash scripts/collect_results.sh
```

This creates:
- `results_archive_YYYYMMDD_HHMMSS/` directory with all files
- `results_archive_YYYYMMDD_HHMMSS.tar.gz` compressed archive

### 2. Download to Local Machine

```bash
# On your local machine
scp -r <username>@bridges2.psc.edu:~/vllm/results_archive_*.tar.gz ./

# Extract
tar -xzf results_archive_*.tar.gz
cd results_archive_*
```

### 3. Validate Fixes Worked

```bash
# On local machine (Mac)
cd /Users/rishi/Downloads/LLMsys_Project/vllm

# Move downloaded results to benchmark_results/
cp path/to/results_archive_*/*.json benchmark_results/

# Run validation
python3 scripts/validate_eviction_fixes.py
```

**Expected output**:
```
✅ SUCCESS: All files show evictions!
Total 1,234 evictions recorded across all runs.
```

---

## What Each Job Validates

### Phase 2 Master (Job 38433031)

**Purpose**: Confirm fixes work across models and workloads

| File | Validates | Key Comparison |
|------|-----------|----------------|
| qwen1.5b_sharegpt | Small model, conversational | vs original (0 evictions, +3.09%) |
| qwen3b_sharegpt | Large model, conversational | vs original (0 evictions, +1.37%) |
| qwen3b_msmarco | Large model, RAG queries | vs original (0 evictions, **-11.61%**) ← Overhead |
| qwen3b_humaneval | Large model, code completion | New workload |
| llama1b_sharegpt | Different architecture | vs original (0 evictions, +9.24%) ← Mystery |

**Critical Questions**:
1. Do all runs show `total_evictions > 0`?
2. Does attention policy beat LRU when evictions happen?
3. Was Llama's +9.24% real evictions or noise?

### Phase 3 Master (Job 38433032)

**Purpose**: Deep analysis for final report

| File | Provides | For Report |
|------|----------|------------|
| memory_sweep | Performance across 6 GPU% configs | "Sweet spot" analysis: where attention helps most |
| hybrid_ablation | Optimal alpha/beta/gamma weights | "Tuning guide" for production use |
| failure_modes | 6 adversarial scenarios | "When NOT to use" recommendations |

**Critical Questions**:
1. At what GPU% do evictions start?
2. What's optimal attention weight (alpha)?
3. Which workloads show no benefit?

---

## Troubleshooting

### If Jobs Fail

```bash
# Check error in SLURM logs
cat slurm-38433031.out | grep -i error
cat slurm-38433032.out | grep -i error

# Check node info
scontrol show job 38433031
scontrol show job 38433032
```

Common issues:
- **OOM**: GPU memory too low → Increase to 10-12%
- **Timeout**: Job exceeded 6 hours → Some models too slow
- **Module not found**: Environment issue → Check conda env

### If Still Zero Evictions

```bash
# Run single test manually to debug
python scripts/test_eviction_trigger.py
```

This systematically tests 4%, 6%, 8%, 10% GPU until evictions trigger.

---

## Timeline Expectations

### Backfill Scheduling

**Typical**: 2-8 hours in queue before starting (overnight is ideal)

Check priority:
```bash
squeue -j 38433031,38433032 -o "%.18i %.10P %.8u %.2t %.10M %.5D %.15R %Q"
```

### Execution Time

- **Phase 2**: ~5-6 hours (5 models × 3 policies × ~20 min each)
- **Phase 3**: ~5-6 hours (deep sweeps with many configs)
- **Total wall time**: Queue time + 6 hours each (can run parallel)

**Best case**: Jobs start 8pm, complete by 8am next day

---

## Next Steps After Validation

Once `validate_eviction_fixes.py` confirms evictions > 0:

### 1. Update Documentation

```bash
# Generate comparison tables
python scripts/validate_eviction_fixes.py > VALIDATION_REPORT.txt

# Update midterm report Section 4.4
# - Replace zero-eviction results with actual eviction data
# - Add comparison table (old vs new)
# - Explain why original runs failed
```

### 2. Create Visualizations

Key charts needed:
- **Memory sweep**: Throughput vs GPU% for all 3 policies
- **Hybrid ablation**: Throughput vs alpha weight
- **Failure modes**: Bar chart showing which scenarios hurt

### 3. Scientific Analysis

Compare predictions to results:
- Did attention policy show 8-12% improvement?
- Did MS-MARCO show -11% overhead without evictions?
- What's the real story behind Llama's +9.24%?

---

## Quick Reference Commands

```bash
# Monitor jobs
bash scripts/monitor_jobs.sh

# Watch live output
tail -f slurm-38433031.out
tail -f slurm-38433032.out

# Collect results
bash scripts/collect_results.sh

# Download (from local machine)
scp user@bridges2.psc.edu:~/vllm/results_archive_*.tar.gz ./

# Validate
python3 scripts/validate_eviction_fixes.py

# Compare old vs new
python3 scripts/validate_eviction_fixes.py | grep "COMPARISON" -A 50
```

---

## Success Criteria

**Minimum to prove fixes worked**:
- ✅ At least 1 file with `total_evictions > 0`
- ✅ At least 1 file with `bytes_gpu_to_cpu > 0`
- ✅ All files have `log_evictions_enabled: true`

**Full validation**:
- ✅ All 8 result files generated
- ✅ All Phase 2 files show evictions
- ✅ Attention policy beats LRU when evictions happen
- ✅ Memory sweep shows clear transition point
- ✅ Failure modes validate overhead predictions

---

## Status Tracking

Track completion:
- [ ] Phase 2 Master (38433031) started
- [ ] Phase 2 Master (38433031) completed
- [ ] Phase 3 Master (38433032) started
- [ ] Phase 3 Master (38433032) completed
- [ ] Results collected (8 JSON files)
- [ ] Results downloaded to local machine
- [ ] Validation script confirms evictions > 0
- [ ] Documentation updated
- [ ] Visualizations created

**Current Status**: Jobs submitted, waiting in queue for backfill allocation.
