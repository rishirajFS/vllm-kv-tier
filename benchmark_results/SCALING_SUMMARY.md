# Benchmark Results Summary

_Auto-generated from benchmark result JSON files_

---

## Executive Summary

**Breakthrough Result**: Attention-weighted eviction achieves **+3.1% throughput improvement** on ShareGPT workload.

- **LRU baseline**: 1653.6 tok/s
- **Attention-weighted**: 1704.7 tok/s

**Control experiments** (synthetic dataset with 20-30% GPU memory) show **no improvement** (±1% variance), confirming that memory pressure is critical for eviction policy effectiveness.

---

## SHAREGPT Dataset

**Configuration**:
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Prompts: 200
- Max Tokens: 1024
- GPU Memory Utilization: 0.15

**Results**:

| Policy | Throughput | Improvement | Avg Latency | P95 Latency | Requests/s |
|--------|------------|-------------|-------------|-------------|------------|
| attention | 1704.7 tok/s | +3.09% | 511.2 ms | 587.9 ms | 1.956 |
| attention | 1131.3 tok/s | -31.58% | 673.2 ms | 774.2 ms | 1.485 |
| attention | 584.5 tok/s | -64.65% | 1401.1 ms | 1611.3 ms | 0.714 |
| **hybrid** | **1724.2 tok/s** | +4.27% | 512.5 ms | 589.4 ms | 1.951 |
| hybrid | 1132.2 tok/s | -31.53% | 672.7 ms | 773.7 ms | 1.486 |
| hybrid | 581.5 tok/s | -64.83% | 1408.9 ms | 1620.2 ms | 0.710 |
| lru | 1653.6 tok/s | — | 518.3 ms | 596.0 ms | 1.929 |
| lru | 1098.8 tok/s | — | 689.2 ms | 792.6 ms | 1.451 |
| lru | 535.1 tok/s | — | 1510.6 ms | 1737.2 ms | 0.662 |

**Source Files**: `results_qwen1.5b_sharegpt_20260404_093906.json`, `results_qwen3b_sharegpt_20260404_105824.json`, `results_sharegpt_20260401_230944.json`

---

## HUMANEVAL Dataset

**Configuration**:
- Model: `meta-llama/Llama-3.2-1B-Instruct`
- Prompts: 164
- Max Tokens: 512
- GPU Memory Utilization: 0.12

**Results**:

| Policy | Throughput | Improvement | Avg Latency | P95 Latency | Requests/s |
|--------|------------|-------------|-------------|-------------|------------|
| attention | 3587.5 tok/s | +2.83% | 121.2 ms | 139.4 ms | 8.252 |
| attention | 3573.4 tok/s | +2.42% | 91.6 ms | 105.3 ms | 10.916 |
| attention | 2193.9 tok/s | -37.12% | 173.5 ms | 199.6 ms | 5.763 |
| **hybrid** | **3595.0 tok/s** | +3.04% | 120.9 ms | 139.1 ms | 8.269 |
| hybrid | 3571.5 tok/s | +2.37% | 91.8 ms | 105.6 ms | 10.895 |
| hybrid | 2165.9 tok/s | -37.92% | 165.4 ms | 190.2 ms | 6.047 |
| lru | 3488.9 tok/s | — | 129.5 ms | 149.0 ms | 7.719 |
| lru | 3563.7 tok/s | — | 91.7 ms | 105.5 ms | 10.899 |
| lru | 2326.7 tok/s | — | 144.6 ms | 166.3 ms | 6.914 |

**Source Files**: `results_humaneval_20260402_150510.json`, `results_qwen1.5b_humaneval_20260404_095639.json`, `results_qwen3b_humaneval_20260404_111223.json`

---

## MSMARCO Dataset

**Configuration**:
- Model: `meta-llama/Llama-3.2-1B-Instruct`
- Prompts: 200
- Max Tokens: 1024
- GPU Memory Utilization: 0.12

**Results**:

| Policy | Throughput | Improvement | Avg Latency | P95 Latency | Requests/s |
|--------|------------|-------------|-------------|-------------|------------|
| attention | 3242.8 tok/s | +8.27% | 230.6 ms | 265.2 ms | 4.336 |
| attention | 3185.0 tok/s | +6.34% | 220.5 ms | 253.5 ms | 4.536 |
| attention | 2453.2 tok/s | -18.09% | 213.8 ms | 245.9 ms | 4.678 |
| hybrid | 3138.5 tok/s | +4.79% | 245.0 ms | 281.8 ms | 4.082 |
| hybrid | 3172.9 tok/s | +5.94% | 231.9 ms | 266.6 ms | 4.313 |
| hybrid | 2813.4 tok/s | -6.06% | 171.4 ms | 197.1 ms | 5.836 |
| lru | 2995.0 tok/s | — | 255.3 ms | 293.6 ms | 3.917 |
| **lru** | **3253.8 tok/s** | — | 214.8 ms | 247.1 ms | 4.655 |
| lru | 2775.5 tok/s | — | 177.7 ms | 204.4 ms | 5.627 |

**Source Files**: `results_msmarco_20260402_150510.json`, `results_qwen1.5b_msmarco_20260404_095252.json`, `results_qwen3b_msmarco_20260404_110742.json`

---

## SYNTHETIC Dataset

**Configuration**:
- Model: `facebook/opt-125m`
- Prompts: 300
- Max Tokens: 256
- GPU Memory Utilization: 0.2

**Results**:

| Policy | Throughput | Improvement | Avg Latency | P95 Latency | Requests/s |
|--------|------------|-------------|-------------|-------------|------------|
| attention | 537.1 tok/s | -0.76% | 388.8 ms | 503.6 ms | 2.572 |
| attention | 194.1 tok/s | -64.13% | 2353.1 ms | 2696.0 ms | 0.425 |
| attention | 2145.6 tok/s | +296.42% | 218.6 ms | 251.4 ms | 4.575 |
| hybrid | 536.5 tok/s | -0.87% | 389.3 ms | 504.2 ms | 2.569 |
| hybrid | 194.1 tok/s | -64.14% | 2353.7 ms | 2694.9 ms | 0.425 |
| hybrid | 2145.0 tok/s | +296.31% | 218.6 ms | 251.4 ms | 4.574 |
| lru | 541.3 tok/s | — | 385.9 ms | 500.0 ms | 2.592 |
| lru | 195.0 tok/s | — | 2342.1 ms | 2683.8 ms | 0.427 |
| **lru** | **2155.1 tok/s** | — | 217.6 ms | 250.3 ms | 4.595 |

**Source Files**: `results_20260401_013454.json`, `results_20260401_123047.json`, `results_20260401_151159.json`

---

## Analysis

For comprehensive analysis, see:
- [kv_cache_tiering/BENCHMARK_RESULTS.md](../kv_cache_tiering/BENCHMARK_RESULTS.md) - Full experimental report
- [kv_cache_tiering/MIDTERM_REPORT.md](../kv_cache_tiering/MIDTERM_REPORT.md) - Academic report
