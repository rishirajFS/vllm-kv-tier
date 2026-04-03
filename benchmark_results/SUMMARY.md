# Benchmark Results Summary

_Auto-generated from benchmark result JSON files_

---

## Executive Summary

**Breakthrough Result**: Attention-weighted eviction achieves **+9.2% throughput improvement** on ShareGPT workload.

- **LRU baseline**: 535.1 tok/s
- **Attention-weighted**: 584.5 tok/s

**Control experiments** (synthetic dataset with 20-30% GPU memory) show **no improvement** (±1% variance), confirming that memory pressure is critical for eviction policy effectiveness.

---

## SHAREGPT Dataset

**Configuration**:
- Model: `meta-llama/Llama-3.2-1B-Instruct`
- Prompts: 200
- Max Tokens: 1024
- GPU Memory Utilization: 0.12

**Results**:

| Policy | Throughput | Improvement | Avg Latency | P95 Latency | Requests/s |
|--------|------------|-------------|-------------|-------------|------------|
| **attention** | **584.5 tok/s** | +9.24% | 1401.1 ms | 1611.3 ms | 0.714 |
| hybrid | 581.5 tok/s | +8.69% | 1408.9 ms | 1620.2 ms | 0.710 |
| lru | 535.1 tok/s | — | 1510.6 ms | 1737.2 ms | 0.662 |

**Source Files**: `results_sharegpt_20260401_230944.json`

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
| **hybrid** | **3595.0 tok/s** | +3.04% | 120.9 ms | 139.1 ms | 8.269 |
| lru | 3488.9 tok/s | — | 129.5 ms | 149.0 ms | 7.719 |

**Source Files**: `results_humaneval_20260402_150510.json`

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
| **attention** | **3242.8 tok/s** | +8.27% | 230.6 ms | 265.2 ms | 4.336 |
| hybrid | 3138.5 tok/s | +4.79% | 245.0 ms | 281.8 ms | 4.082 |
| lru | 2995.0 tok/s | — | 255.3 ms | 293.6 ms | 3.917 |

**Source Files**: `results_msmarco_20260402_150510.json`

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
