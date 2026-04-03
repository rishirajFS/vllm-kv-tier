# Qwen 2.5 7B Benchmark Suite

This guide explains how to run comprehensive benchmarks on **Qwen 2.5 7B** to validate workload generalization across model families.

---

## Why Qwen 2.5?

**Model Family Generalization**: Your current results use LLaMA-3.2-1B. Testing on Qwen 2.5 7B proves your attention-aware eviction method works across:
- **Different architectures** (LLaMA vs Qwen)
- **Different model sizes** (1B vs 7B parameters)
- **Different training datasets** (LLaMA trained on English-heavy, Qwen includes multilingual + code)

**Advantages of Qwen 2.5 7B**:
1. **Strong code performance** - better HumanEval baseline than LLaMA
2. **128K context support** - enables long-context stress tests
3. **Recent release** (late 2024) - reviewers can't dismiss as "old model"
4. **Well-documented** - excellent vLLM support, active community

---

## Benchmark Suite Overview

The `run_qwen7b_benchmark_suite.sh` script runs **three complete benchmarks**:

| Benchmark | Workload Type | Prompts | Max Tokens | Expected Improvement |
|-----------|---------------|---------|------------|---------------------|
| **ShareGPT** | Conversational | 200 | 1024 | +8-10% (attention) |
| **MS-MARCO** | RAG (retrieval) | 200 | 1024 | +7-9% (attention) |
| **HumanEval** | Code completion | 164 | 512 | +3-5% (hybrid/attention) |

**Memory Pressure**: All benchmarks use **12% GPU memory** (same as LLaMA tests) to force eviction.

---

## Prerequisites

### 1. Datasets Downloaded

Ensure you have all three datasets in `~/workspace/vllm/datasets/`:

```bash
ls ~/workspace/vllm/datasets/
# Expected output:
# sharegpt.json
# msmarco.json
# humaneval.json
```

If missing, download them:
```bash
cd ~/workspace/vllm/datasets
# Follow instructions in kv_cache_tiering/benchmarks/DATASETS.md
```

### 2. GPU Access

**Minimum Requirements**:
- **GPU**: NVIDIA V100 (32GB), A100 (40GB/80GB), or H100
- **VRAM**: ≥24 GB (Qwen 7B model + KV cache @ 12% = ~20GB total)
- **CUDA**: 12.0+

**Check your GPU**:
```bash
nvidia-smi
# Verify: V100-32GB or better
```

### 3. vLLM Installation

```bash
# Should already be installed from previous work
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
```

---

## Running the Benchmark Suite

### Quick Start (SLURM)

If on PSC Bridges-2 or similar HPC cluster:

```bash
cd /Users/rishi/Downloads/LLMsys_Project/vllm

# Submit as SLURM job (recommended)
sbatch scripts/run_qwen7b_benchmark_suite.sh

# Check job status
squeue -u $USER

# Monitor progress
tail -f vllm_qwen7b_suite.log
```

**Estimated Runtime**: 3-4 hours total
- ShareGPT: ~60-90 minutes
- MS-MARCO: ~60-90 minutes
- HumanEval: ~30-45 minutes

### Local Execution (Non-SLURM)

If running on a local GPU workstation:

```bash
cd /Users/rishi/Downloads/LLMsys_Project/vllm

bash scripts/run_qwen7b_benchmark_suite.sh
```

**Note**: This will block your terminal for 3-4 hours. Consider using `screen` or `tmux`:

```bash
# Start a screen session
screen -S qwen_bench

# Run benchmark
bash scripts/run_qwen7b_benchmark_suite.sh

# Detach: Ctrl+A, then D
# Re-attach later: screen -r qwen_bench
```

---

## Configuration Options

### Environment Variables

Customize the benchmark via environment variables:

```bash
# Change GPU memory utilization (default: 12%)
export GPU_MEM_UTIL=0.10  # 10% = more memory pressure

# Change CPU offload memory (default: 8GB)
export CPU_BYTES=4000000000  # 4GB

# Change dataset location
export DATASET_DIR=/path/to/datasets

# Run only specific policies
export POLICIES="lru attention"  # Skip hybrid

# Run the suite
sbatch scripts/run_qwen7b_benchmark_suite.sh
```

### Manual Per-Dataset Execution

Run benchmarks individually:

```bash
# ShareGPT only
MODEL=Qwen/Qwen2.5-7B-Instruct \
python -m kv_cache_tiering.benchmarks.benchmark \
    --model Qwen/Qwen2.5-7B-Instruct \
    --policies lru attention hybrid \
    --dataset sharegpt \
    --dataset-path ~/workspace/vllm/datasets/sharegpt.json \
    --num-prompts 200 \
    --max-tokens 1024 \
    --gpu-mem-util 0.12 \
    --cpu-bytes 8000000000 \
    --output benchmark_results/results_qwen7b_sharegpt.json

# MS-MARCO only
python -m kv_cache_tiering.benchmarks.benchmark \
    --model Qwen/Qwen2.5-7B-Instruct \
    --policies lru attention hybrid \
    --dataset msmarco \
    --dataset-path ~/workspace/vllm/datasets/msmarco.json \
    --num-prompts 200 \
    --max-tokens 1024 \
    --gpu-mem-util 0.12 \
    --cpu-bytes 8000000000 \
    --output benchmark_results/results_qwen7b_msmarco.json

# HumanEval only
python -m kv_cache_tiering.benchmarks.benchmark \
    --model Qwen/Qwen2.5-7B-Instruct \
    --policies lru attention hybrid \
    --dataset humaneval \
    --dataset-path ~/workspace/vllm/datasets/humaneval.json \
    --num-prompts 164 \
    --max-tokens 512 \
    --gpu-mem-util 0.12 \
    --cpu-bytes 8000000000 \
    --output benchmark_results/results_qwen7b_humaneval.json
```

---

## Output and Results

### Result Files

After completion, find results in `benchmark_results/`:

```bash
ls -lh benchmark_results/results_qwen7b_*

# Expected output (timestamps will vary):
# results_qwen7b_sharegpt_20260402_120000.json
# results_qwen7b_msmarco_20260402_130000.json
# results_qwen7b_humaneval_20260402_140000.json
```

### Quick Results Check

View results immediately:

```bash
# Pretty-print JSON
cat benchmark_results/results_qwen7b_sharegpt_*.json | jq '.[0]'

# Extract throughput comparison
cat benchmark_results/results_qwen7b_sharegpt_*.json | jq '.[] | {policy, throughput: .tokens_per_second, ttft: .avg_ttft_ms}'
```

### Generate Consolidated Summary

Create markdown summary with all results:

```bash
python scripts/generate_results_summary.py \
    --results-dir benchmark_results \
    --output benchmark_results/SUMMARY.md

# View summary
cat benchmark_results/SUMMARY.md
```

**Expected Summary Format**:

```markdown
## SHAREGPT Dataset (Qwen 7B)

| Policy | Throughput | Improvement | Avg Latency | P95 Latency | Avg TTFT | P95 TTFT |
|--------|------------|-------------|-------------|-------------|----------|----------|
| **attention** | **XXX tok/s** | **+Y%** | XXX ms | XXX ms | XXX ms | XXX ms |
| hybrid | XXX tok/s | +Y% | XXX ms | XXX ms | XXX ms | XXX ms |
| lru | XXX tok/s | — | XXX ms | XXX ms | XXX ms | XXX ms |
```

---

## Expected Results

### Predictions Based on LLaMA Results

| Workload | LLaMA-3.2-1B Result | Qwen 7B Prediction | Rationale |
|----------|---------------------|-------------------|-----------|
| **ShareGPT** | +9.24% (attention) | **+8-10%** | Similar conversational patterns |
| **MS-MARCO** | +8.27% (attention) | **+7-9%** | RAG attention patterns consistent |
| **HumanEval** | +3.04% (hybrid) | **+4-6%** | Larger model → more cache pressure |

**Why Higher for Qwen 7B?**
- **7B parameters** → larger KV cache → more eviction opportunities
- **More layers** (32 vs 16) → deeper attention hierarchy
- **Larger hidden dim** (4096 vs 2048) → more discriminative attention scores

### TTFT Metrics (New!)

With the updated benchmark harness, you'll now see **Time-To-First-Token** metrics:

**Expected TTFT patterns**:
- **LRU**: Higher TTFT (must evict/reload important blocks before first token)
- **Attention**: Lower TTFT (important blocks stay on GPU)
- **Typical TTFT range**: 50-200ms depending on context length

---

## Comparing LLaMA vs Qwen Results

### Side-by-Side Comparison

```bash
# Extract LLaMA ShareGPT results
echo "LLaMA-3.2-1B ShareGPT:"
cat benchmark_results/results_sharegpt_20260401_*.json | jq '.[] | {policy, throughput: .tokens_per_second}'

# Extract Qwen 7B ShareGPT results
echo "Qwen 7B ShareGPT:"
cat benchmark_results/results_qwen7b_sharegpt_*.json | jq '.[] | {policy, throughput: .tokens_per_second}'
```

### Expected Insights

1. **Absolute Throughput**: Qwen 7B will be **slower** than LLaMA 1B (larger model)
   - LLaMA-1B: ~500-600 tok/s
   - Qwen-7B: ~200-400 tok/s (estimate)

2. **Relative Improvement**: Qwen 7B should show **similar or higher %** improvement
   - Both: +8-10% on ShareGPT, +7-9% on MS-MARCO, +3-6% on HumanEval

3. **TTFT**: Qwen 7B will have **higher absolute TTFT** but similar relative improvement
   - Attention-aware should reduce TTFT by 10-20% vs LRU

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Error**: `CUDA out of memory`

**Solution 1**: Reduce GPU memory utilization
```bash
export GPU_MEM_UTIL=0.10  # Try 10% instead of 12%
sbatch scripts/run_qwen7b_benchmark_suite.sh
```

**Solution 2**: Reduce batch size (prompts)
```bash
# Edit the script to use fewer prompts:
# ShareGPT: 100 instead of 200
# MS-MARCO: 100 instead of 200
# HumanEval: 100 instead of 164
```

**Solution 3**: Reduce max context length
```bash
export MAX_MODEL_LEN=16384  # Instead of 32768
```

### Issue: Model Download Timeout

**Error**: `Connection timeout downloading Qwen/Qwen2.5-7B-Instruct`

**Solution**: Pre-download the model
```bash
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', device_map='cpu')
"
```

### Issue: Dataset Not Found

**Error**: `ERROR: Dataset file not found: sharegpt.json`

**Solution**: Download datasets following `kv_cache_tiering/benchmarks/DATASETS.md`

### Issue: TTFT Still Zero

**Possible causes**:
1. **vLLM version too old** - RequestOutput.metrics not available
2. **Offline API limitation** - Batched generation doesn't track per-request TTFT

**Workaround**: Check vLLM version and update if needed
```bash
pip install --upgrade vllm
```

If TTFT remains 0, the latency metrics are still valid and improvements will still be visible in throughput and avg_latency.

---

## Analysis and Next Steps

### After Benchmark Completion

1. **Verify Results Match Predictions**:
   - ShareGPT: +8-10% improvement? ✓
   - MS-MARCO: +7-9% improvement? ✓
   - HumanEval: +3-6% improvement? ✓

2. **Update Documentation**:
   - Add Qwen results to `BENCHMARK_RESULTS.md`
   - Update `MIDTERM_REPORT.md` with model generalization findings
   - Regenerate `SUMMARY.md`

3. **Paper Narrative**:
   > "We validate our approach across two model families: LLaMA-3.2-1B and Qwen-2.5-7B. Consistent improvements (8-10% for conversational, 7-9% for RAG, 3-6% for code) demonstrate that attention-aware eviction generalizes across architectures and model sizes."

### Optional: Extended Experiments

**Long-Context Scaling** (Qwen 2.5 supports 128K context):

```bash
# Test with longer contexts
python -m kv_cache_tiering.benchmarks.benchmark \
    --model Qwen/Qwen2.5-7B-Instruct \
    --policies lru attention \
    --dataset sharegpt \
    --dataset-path ~/workspace/vllm/datasets/sharegpt_long.json \
    --num-prompts 50 \
    --max-model-len 65536 \
    --max-tokens 2048 \
    --gpu-mem-util 0.08 \
    --output results_qwen7b_longcontext.json
```

**Expected**: Even larger improvements (+15-25%) at long contexts due to extreme eviction pressure.

---

## Contact

For issues or questions:
- **Researcher**: Rishi Nagaraj (rnagaraj@andrew.cmu.edu)
- **Course**: 11-868 LLM Systems, CMU Spring 2026
- **GitHub Issues**: (if public repo)

---

## Summary Checklist

Before running the benchmark suite:

- [ ] All datasets downloaded (`sharegpt.json`, `msmarco.json`, `humaneval.json`)
- [ ] GPU available (V100-32GB or better)
- [ ] vLLM installed with GPU support
- [ ] Sufficient disk space (~50GB for model + results)
- [ ] SLURM/GPU access configured (if on HPC cluster)

After running:

- [ ] All 3 benchmarks completed successfully
- [ ] Results files generated (`results_qwen7b_*.json`)
- [ ] Summary generated (`SUMMARY.md`)
- [ ] Results match predictions (±2-3%)
- [ ] TTFT metrics captured (if supported)
- [ ] Compare with LLaMA results for validation

Good luck with your benchmarks! 🚀
