# Benchmark Results Index

This directory contains JSON result files from KV cache eviction policy benchmarks.

**Latest Update**: April 1, 2026

---

## File Naming Convention

`results_<dataset>_<timestamp>.json`

- `<dataset>`: Dataset name (sharegpt, msmarco, humaneval, synthetic)
- `<timestamp>`: YYYYMMDD_HHMMSS format

Example: `results_sharegpt_20260401_230944.json`

---

## Current Results

### Primary Result: ShareGPT (Memory Pressure Benchmark)

**File**: [`results_sharegpt_20260401_230944.json`](results_sharegpt_20260401_230944.json)

**Key Finding**: **9.2% throughput improvement** with attention-weighted eviction vs LRU baseline

| Policy | Throughput | Improvement | Configuration |
|--------|------------|-------------|---------------|
| LRU | 535.1 tok/s | baseline | 12% GPU, 8GB CPU, 200 prompts |
| Attention | 584.5 tok/s | +9.2% | 12% GPU, 8GB CPU, 200 prompts |
| Hybrid | 581.5 tok/s | +8.6% | 12% GPU, 8GB CPU, 200 prompts |

**Why This Matters**: 12% GPU memory utilization forced real KV cache evictions, demonstrating that content-aware eviction outperforms recency-based heuristics under memory pressure.

**Reproduction Command**:
```bash
python -m kv_cache_tiering.benchmarks.benchmark \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --policies lru attention hybrid \
    --dataset sharegpt \
    --dataset-path /path/to/sharegpt.json \
    --num-prompts 200 \
    --max-tokens 1024 \
    --gpu-mem-util 0.12 \
    --cpu-bytes 8000000000 \
    --block-size 48 \
    --output results_sharegpt_$(date +%Y%m%d_%H%M%S).json
```

---

### Control Experiments (No Memory Pressure)

All three control experiments show **zero evictions** and **identical performance** across policies (within 1% variance).

#### Control 1: OPT-125M, Synthetic Dataset

**File**: [`results_20260401_013454.json`](results_20260401_013454.json)

| Policy | Throughput | Variance from LRU | Configuration |
|--------|------------|-------------------|---------------|
| LRU | 541.3 tok/s | baseline | 20% GPU, 2GB CPU, 300 prompts |
| Attention | 537.1 tok/s | -0.8% | 20% GPU, 2GB CPU, 300 prompts |
| Hybrid | 536.5 tok/s | -0.9% | 20% GPU, 2GB CPU, 300 prompts |

**Observation**: All policies within measurement noise (±1%). GPU memory utilization of 20% provided ample KV cache headroom, so eviction logic was never invoked.

**Reproduction Command**:
```bash
python -m kv_cache_tiering.benchmarks.benchmark \
    --model facebook/opt-125m \
    --policies lru attention hybrid \
    --dataset synthetic \
    --num-prompts 300 \
    --max-tokens 256 \
    --gpu-mem-util 0.2 \
    --cpu-bytes 2000000000 \
    --output results_opt125m_$(date +%Y%m%d_%H%M%S).json
```

#### Control 2: Llama-3.2-1B, Synthetic Dataset (Lower Throughput)

**File**: [`results_20260401_123047.json`](results_20260401_123047.json)

| Policy | Throughput | Variance from LRU | Configuration |
|--------|------------|-------------------|---------------|
| LRU | 195.0 tok/s | baseline | 30% GPU, 4GB CPU, 200 prompts |
| Attention | 194.1 tok/s | -0.5% | 30% GPU, 4GB CPU, 200 prompts |
| Hybrid | 194.1 tok/s | -0.5% | 30% GPU, 4GB CPU, 200 prompts |

**Observation**: All policies within 0.5% (measurement noise). No evictions occurred.

#### Control 3: Llama-3.2-1B, Synthetic Dataset (Higher Throughput)

**File**: [`results_20260401_151159.json`](results_20260401_151159.json)

| Policy | Throughput | Variance from LRU | Configuration |
|--------|------------|-------------------|---------------|
| LRU | 2155.1 tok/s | baseline | 30% GPU, 4GB CPU, 200 prompts |
| Attention | 2145.6 tok/s | -0.4% | 30% GPU, 4GB CPU, 200 prompts |
| Hybrid | 2145.0 tok/s | -0.5% | 30% GPU, 4GB CPU, 200 prompts |

**Observation**: All policies within 0.5% (measurement noise). No evictions occurred.

**Key Insight**: Without memory pressure (20-30% GPU utilization), eviction policies perform identically because eviction is never needed.

---

## Future Workloads

### MS-MARCO (RAG Workload) - PENDING

**Dataset**: Microsoft Machine Reading Comprehension (passage retrieval queries)
**Status**: Not yet run
**Expected Result**: 5-15% improvement (lower than ShareGPT due to less prefix-sharing)

**Planned Configuration**:
- Model: `meta-llama/Llama-3.2-1B-Instruct`
- Dataset: MS-MARCO queries
- Prompts: 200
- Max tokens: 1024
- GPU memory: 12% (same memory pressure as ShareGPT)

**Reproduction Command**:
```bash
# Download MS-MARCO dataset first (see kv_cache_tiering/benchmarks/DATASETS.md)

bash scripts/run_msmarco_benchmark.sh
# OR
python -m kv_cache_tiering.benchmarks.benchmark \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --policies lru attention hybrid \
    --dataset msmarco \
    --dataset-path /path/to/msmarco.json \
    --num-prompts 200 \
    --max-tokens 1024 \
    --gpu-mem-util 0.12 \
    --cpu-bytes 8000000000 \
    --output results_msmarco_$(date +%Y%m%d_%H%M%S).json
```

### HumanEval (Code Completion) - PENDING

**Dataset**: OpenAI HumanEval (164 Python programming problems)
**Status**: Not yet run
**Expected Result**: 3-7% improvement (moderate, due to less prefix-sharing than conversation)

**Planned Configuration**:
- Model: `meta-llama/Llama-3.2-1B-Instruct` or `codellama`
- Dataset: HumanEval problems
- Prompts: 164 (all problems)
- Max tokens: 512
- GPU memory: 12%

**Reproduction Command**:
```bash
# Download HumanEval dataset first (see kv_cache_tiering/benchmarks/DATASETS.md)

bash scripts/run_humaneval_benchmark.sh
# OR
python -m kv_cache_tiering.benchmarks.benchmark \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --policies lru attention hybrid \
    --dataset humaneval \
    --dataset-path /path/to/humaneval.json \
    --num-prompts 164 \
    --max-tokens 512 \
    --gpu-mem-util 0.12 \
    --cpu-bytes 8000000000 \
    --output results_humaneval_$(date +%Y%m%d_%H%M%S).json
```

---

## Results Summary Table

| Dataset | Model | GPU Util | Prompts | LRU (tok/s) | Attention (tok/s) | Hybrid (tok/s) | Improvement | Status |
|---------|-------|----------|---------|-------------|-------------------|----------------|-------------|--------|
| **ShareGPT** | Llama-3.2-1B | **12%** | 200 | 535.1 | **584.5** | 581.5 | **+9.2%** | COMPLETE |
| Synthetic | opt-125m | 20% | 300 | 541.3 | 537.1 | 536.5 | ±0% (no evict) | COMPLETE |
| Synthetic | Llama-3.2-1B | 30% | 200 | 195.0 | 194.1 | 194.1 | ±0% (no evict) | COMPLETE |
| Synthetic | Llama-3.2-1B | 30% | 200 | 2155.1 | 2145.6 | 2145.0 | ±0% (no evict) | COMPLETE |
| MS-MARCO | Llama-3.2-1B | 12% | 200 | TBD | TBD | TBD | TBD | PENDING |
| HumanEval | Llama-3.2-1B | 12% | 164 | TBD | TBD | TBD | TBD | PENDING |

---

## Analysis Reference

For comprehensive analysis of these results, see:
- **[kv_cache_tiering/BENCHMARK_RESULTS.md](../kv_cache_tiering/BENCHMARK_RESULTS.md)** - Full experimental report with statistical analysis, root cause analysis, and future work recommendations
- **[kv_cache_tiering/MIDTERM_REPORT.md](../kv_cache_tiering/MIDTERM_REPORT.md)** - Academic midterm report documenting the full project

---

## Hardware Platform

All benchmarks run on:
- **System**: Pittsburgh Supercomputing Center (PSC) Bridges-2
- **GPU**: NVIDIA V100-32GB (Volta, compute capability 7.0)
- **CPU**: Intel Xeon Gold 6248
- **Memory**: 512 GB DDR4
- **CUDA**: 12.4.0
- **Attention Backend**: FlashInfer (V100 does not support FlashAttention2)

---

## Reproducing Results

### Prerequisites

1. NVIDIA GPU with ≥24 GB VRAM
2. CUDA 12.4+
3. Python 3.10+
4. vLLM (commit `003800536` or later)

### Quick Start

```bash
# 1. Install vLLM with KV tiering support
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 003800536
pip install -e ".[dev]"

# 2. Download datasets (see kv_cache_tiering/benchmarks/DATASETS.md)

# 3. Run ShareGPT benchmark (primary result)
python -m kv_cache_tiering.benchmarks.benchmark \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --policies lru attention hybrid \
    --dataset sharegpt \
    --dataset-path sharegpt.json \
    --num-prompts 200 \
    --max-tokens 1024 \
    --gpu-mem-util 0.12 \
    --cpu-bytes 8000000000 \
    --output results_sharegpt.json

# Expected: Attention ~9% faster than LRU
```

---

## File Inventory

| File | Size | Description | Status |
|------|------|-------------|--------|
| `results_sharegpt_20260401_230944.json` | 5 KB | ShareGPT benchmark (9.2% improvement) | PRIMARY |
| `results_20260401_013454.json` | 5 KB | OPT-125M synthetic (no improvement) | CONTROL |
| `results_20260401_123047.json` | 5 KB | Llama synthetic low throughput (no improvement) | CONTROL |
| `results_20260401_151159.json` | 5 KB | Llama synthetic high throughput (no improvement) | CONTROL |
| `results_msmarco_*.json` | - | MS-MARCO RAG workload | PENDING |
| `results_humaneval_*.json` | - | HumanEval code completion | PENDING |

---

## Contact

**Researcher**: Rishi Nagaraj (rnagaraj@andrew.cmu.edu)
**Course**: 11-868 Large Language Model Systems, Spring 2026
**Institution**: Carnegie Mellon University
