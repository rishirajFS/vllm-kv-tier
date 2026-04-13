# Fix: Eviction Instrumentation

## Problem Summary

All benchmark results show `total_evictions: 0` despite some showing throughput improvements. Root cause analysis identified TWO separate bugs:

### Bug 1: Missing total_evictions Counter

**Issue**: OffloadingConnectorStats does not track or expose total_evictions metric.

**Current State**:
- `OffloadingConnectorStats` only tracks transfer operations (gpu_to_cpu, cpu_to_gpu bytes/time)
- `AttentionBlockManager.get_stats()` returns block counts and scores, but NOT eviction count
- Eviction data exists in `eviction_log` but no aggregated counter

**Evidence**:
```python
# vllm/v1/kv_offload/attention_manager.py:265
def get_stats(self) -> dict:
    return {
        "total_blocks": total,
        "ready_blocks": ready,
        "avg_attention_score": avg_score,
        "free_backend_blocks": self.backend.get_num_free_blocks(),
        # ❌ total_evictions missing
    }
```

### Bug 2: Eviction Logging Disabled in Benchmark

**Issue**: benchmark.py doesn't enable `log_evictions` flag in connector config.

**Current State**:
```python
# kv_cache_tiering/benchmarks/benchmark.py:85-99
def build_kv_connector_config(config: BenchmarkConfig) -> dict:
    extra_config = {
        "cpu_bytes_to_use": config.cpu_bytes_to_use,
        "block_size": config.block_size,
        "eviction_policy": config.eviction_policy,
        # ❌ "log_evictions": True MISSING
    }
```

**Evidence from Results**:
All 7 JSON files show: `"log_evictions_enabled": false, "eviction_log": false`

---

## Fix Plan

### Fix 1: Add total_evictions Counter to Manager Stats

**File**: `vllm/v1/kv_offload/attention_manager.py`

**Current implementation** (lines 11-30):
```python
class AttentionBlockManager:
    def __init__(...):
        self.blocks: dict[BlockHash, BlockMetadata] = {}
        self.eviction_log: list[EvictionRecord] = []
        self.log_evictions = log_evictions
        # ❌ No eviction counter
```

**Required changes**:

1. **Add counter field** (after line 29):
```python
class AttentionBlockManager:
    def __init__(...):
        self.blocks: dict[BlockHash, BlockMetadata] = {}
        self.eviction_log: list[EvictionRecord] = []
        self.log_evictions = log_evictions
        self._total_evictions = 0  # ✅ NEW: Track total evictions
```

2. **Increment counter during eviction** (line 199, in prepare_store method):
```python
# Current code at line 189-196:
if self.log_evictions:
    self.eviction_log.append(
        EvictionRecord(
            block_hash=block_hash,
            score=meta.cumulative_attention_score,
            access_count=meta.access_count,
            timestamp=time.monotonic(),
        )
    )

# Add AFTER this block (line 197):
self._total_evictions += 1  # ✅ NEW: Always increment counter
```

3. **Expose counter in get_stats()** (line 265-280):
```python
# Current:
def get_stats(self) -> dict:
    total = len(self.blocks)
    ready = sum(1 for m in self.blocks.values() if m.status.is_ready)
    avg_score = (
        sum(m.cumulative_attention_score for m in self.blocks.values())
        / total
        if total > 0
        else 0.0
    )
    return {
        "total_blocks": total,
        "ready_blocks": ready,
        "avg_attention_score": avg_score,
        "free_backend_blocks": self.backend.get_num_free_blocks(),
    }

# Updated:
def get_stats(self) -> dict:
    total = len(self.blocks)
    ready = sum(1 for m in self.blocks.values() if m.status.is_ready)
    avg_score = (
        sum(m.cumulative_attention_score for m in self.blocks.values())
        / total
        if total > 0
        else 0.0
    )
    return {
        "total_blocks": total,
        "ready_blocks": ready,
        "avg_attention_score": avg_score,
        "free_backend_blocks": self.backend.get_num_free_blocks(),
        "total_evictions": self._total_evictions,  # ✅ NEW
    }
```

**Also apply same changes to** `vllm/v1/kv_offload/hybrid_manager.py` (inherits from AttentionBlockManager):
- The hybrid_manager already calls `super().get_stats()` so it will inherit the total_evictions field
- Just need to ensure counter increment happens in its eviction logic too

---

### Fix 2: Enable Eviction Logging in Benchmark

**File**: `kv_cache_tiering/benchmarks/benchmark.py`

**Change at lines 85-99**:

```python
# Current:
def build_kv_connector_config(config: BenchmarkConfig) -> dict:
    extra_config = {
        "cpu_bytes_to_use": config.cpu_bytes_to_use,
        "block_size": config.block_size,
        "eviction_policy": config.eviction_policy,
    }

    # Set policy-specific parameters
    if config.eviction_policy == "attention":
        extra_config["score_decay"] = config.score_decay
    elif config.eviction_policy == "hybrid":
        extra_config["attention_weight"] = config.attention_weight
        extra_config["recency_weight"] = config.recency_weight
        extra_config["frequency_weight"] = config.frequency_weight
        extra_config["score_decay"] = config.score_decay

    return extra_config

# Updated:
def build_kv_connector_config(config: BenchmarkConfig) -> dict:
    extra_config = {
        "cpu_bytes_to_use": config.cpu_bytes_to_use,
        "block_size": config.block_size,
        "eviction_policy": config.eviction_policy,
        "log_evictions": True,  # ✅ NEW: Enable eviction logging
    }

    # Set policy-specific parameters
    if config.eviction_policy == "attention":
        extra_config["score_decay"] = config.score_decay
    elif config.eviction_policy == "hybrid":
        extra_config["attention_weight"] = config.attention_weight
        extra_config["recency_weight"] = config.recency_weight
        extra_config["frequency_weight"] = config.frequency_weight
        extra_config["score_decay"] = config.score_decay

    return extra_config
```

---

### Fix 3: Add get_stats() Method to OffloadingConnector

**File**: `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`

The benchmark calls `llm.llm_engine.engine_core.kv_connector.get_stats()`, but this method doesn't exist on the connector. Need to add it:

**Add after line 229** (after get_eviction_log method):

```python
def get_stats(self) -> dict:
    """
    Get comprehensive stats including transfer metrics and eviction data.

    Returns:
        Dictionary with:
        - total_evictions: Number of blocks evicted to CPU
        - bytes_gpu_to_cpu: Total bytes transferred GPU → CPU
        - bytes_cpu_to_gpu: Total bytes transferred CPU → GPU
        - avg_transfer_time_gpu_to_cpu: Average transfer time (ms)
        - avg_transfer_time_cpu_to_gpu: Average transfer time (ms)
        - Plus manager-specific stats (blocks, scores, etc.)
    """
    stats = {}

    # Get transfer stats from worker
    worker = self.connector_worker
    if worker:
        kv_stats = worker.get_kv_connector_stats()
        if kv_stats:
            reduced = kv_stats.reduce()

            # Extract bytes transferred
            stats["bytes_gpu_to_cpu"] = reduced.get("gpu_to_cpu_total_bytes", 0)
            stats["bytes_cpu_to_gpu"] = reduced.get("cpu_to_gpu_total_bytes", 0)

            # Calculate average transfer times
            gpu_to_cpu_time = reduced.get("gpu_to_cpu_total_time", 0)
            cpu_to_gpu_time = reduced.get("cpu_to_gpu_total_time", 0)
            gpu_to_cpu_bytes = stats["bytes_gpu_to_cpu"]
            cpu_to_gpu_bytes = stats["bytes_cpu_to_gpu"]

            # Avg time per operation (if we had N operations, time/N)
            # For now use total time in ms
            stats["avg_transfer_time_gpu_to_cpu_ms"] = gpu_to_cpu_time * 1000
            stats["avg_transfer_time_cpu_to_gpu_ms"] = cpu_to_gpu_time * 1000

        # Get manager stats (includes total_evictions after Fix 1)
        if hasattr(worker, 'manager'):
            manager_stats = worker.manager.get_stats()
            stats.update(manager_stats)

    return stats
```

---

## Testing the Fixes

### Step 1: Apply All Three Fixes

```bash
cd /Users/rishi/Downloads/LLMsys_Project/vllm

# Implement fixes (manual edits or use Edit tool)
# Fix 1: attention_manager.py (add counter, increment, expose)
# Fix 2: benchmark.py (enable log_evictions)
# Fix 3: offloading_connector.py (add get_stats method)
```

### Step 2: Run Local Test (No GPU Required)

```bash
python3 scripts/check_instrumentation_local.py
```

**Expected output changes**:
```diff
- ❌ Connector doesn't track total_evictions in stats
+ ✅ Connector tracks total_evictions in stats
```

### Step 3: Run Minimal Benchmark

```bash
cd kv_cache_tiering/benchmarks

python3 benchmark.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --eviction-policy lru \
  --num-prompts 10 \
  --dataset sharegpt \
  --gpu-memory-utilization 0.08 \
  --max-model-len 2048 \
  --output /tmp/test_eviction_fix.json
```

**Expected results**:
```json
{
  "total_evictions": 15,  // ✅ Non-zero!
  "bytes_gpu_to_cpu": 73728,  // ✅ Non-zero!
  "bytes_cpu_to_gpu": 49152,  // ✅ Non-zero!
  "eviction_log": true,  // ✅ Enabled!
  "log_evictions_enabled": true  // ✅ Enabled!
}
```

### Step 4: Verify Eviction Log Data

```python
import json

with open("/tmp/test_eviction_fix.json") as f:
    results = json.load(f)

assert results[0]["total_evictions"] > 0, "Evictions should be counted"
assert results[0]["bytes_gpu_to_cpu"] > 0, "GPU→CPU transfers should happen"
print(f"✅ Fix validated: {results[0]['total_evictions']} evictions recorded")
```

---

## Re-Running Benchmarks with Fixes

Once fixes are validated, re-run all 7 benchmarks with corrected settings:

### Qwen 1.5B ShareGPT
```bash
python3 benchmark.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --eviction-policy lru attention hybrid \
  --num-prompts 200 \
  --dataset sharegpt \
  --gpu-memory-utilization 0.08 \  # ✅ Reduced from 0.15
  --max-model-len 2048 \  # ✅ Reduced from 16384
  --output benchmark_results/results_qwen1.5b_sharegpt_fixed_$(date +%Y%m%d_%H%M%S).json
```

### Qwen 3B ShareGPT
```bash
python3 benchmark.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --eviction-policy lru attention hybrid \
  --num-prompts 200 \
  --dataset sharegpt \
  --gpu-memory-utilization 0.12 \  # ✅ Reduced from 0.25
  --max-model-len 4096 \  # ✅ Reduced from 32768
  --output benchmark_results/results_qwen3b_sharegpt_fixed_$(date +%Y%m%d_%H%M%S).json
```

**Expected improvements**:
- total_evictions > 0 for all runs
- Clear performance difference between policies (if evictions happening)
- OR confirmation that overhead exists when memory is ample (validating failure mode analysis)

---

## Root Cause Summary

| Bug | Impact | Fix |
|-----|--------|-----|
| **Bug 1**: No total_evictions counter in manager | Evictions happen but aren't counted | Add `_total_evictions` field, increment during eviction, expose in get_stats() |
| **Bug 2**: Eviction logging not enabled in benchmark | eviction_log stays empty, log_evictions_enabled=false | Add `"log_evictions": True` to kv_connector_extra_config |
| **Bug 3**: No get_stats() method in connector | Benchmark can't retrieve stats | Add get_stats() method that aggregates transfer + manager stats |

**Key Insight**: The evictions MAY have been happening in some runs (like Llama +9.24%), but we had no instrumentation to confirm. After fixes, we can definitively measure whether throughput gains came from better eviction policy or other factors.

---

## Next Steps

1. **Apply all three fixes** (attention_manager.py, benchmark.py, offloading_connector.py)
2. **Run local validation** with check_instrumentation_local.py
3. **Run minimal benchmark** (10 prompts) to verify non-zero evictions
4. **Re-run full benchmark suite** with reduced GPU memory (4-12%) and max_model_len (2K-4K)
5. **Analyze results** to determine:
   - Did original Llama +9.24% results actually have evictions? (We'll never know without re-running)
   - Do attention-aware policies help when evictions DO occur?
   - How much overhead exists when evictions DON'T occur? (Validates failure mode analysis)

**Status**: Ready to implement fixes.
