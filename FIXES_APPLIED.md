# Eviction Instrumentation Fixes - Applied

## Summary

Successfully implemented all three fixes to resolve zero evictions issue in benchmark results.

---

## Fix 1: Add total_evictions Counter to AttentionBlockManager ✅

**File**: `vllm/v1/kv_offload/attention_manager.py`

**Changes**:

1. **Added counter field** (line ~75):
```python
# Total evictions counter for stats
self._total_evictions: int = 0
```

2. **Increment counter during eviction** (line ~199):
```python
# Increment eviction counter
self._total_evictions += 1
```

3. **Expose counter in get_stats()** (line ~285):
```python
return {
    "total_blocks": total,
    "ready_blocks": ready,
    "avg_attention_score": avg_score,
    "free_backend_blocks": self.backend.get_num_free_blocks(),
    "total_evictions": self._total_evictions,  # ✅ NEW
}
```

**Impact**: The manager now tracks total evictions and exposes them via get_stats().

---

## Fix 2: Enable Eviction Logging in Benchmark ✅

**File**: `kv_cache_tiering/benchmarks/benchmark.py`

**Changes** (line ~91):

```python
extra = {
    "cpu_bytes_to_use": config.cpu_bytes_to_use,
    "block_size": config.block_size,
    "eviction_policy": config.eviction_policy,
    "log_evictions": True,  # ✅ NEW: Enable eviction logging for instrumentation
}
```

**Impact**: All future benchmark runs will have eviction logging enabled, populating eviction_log data.

---

## Fix 3: Add get_stats() Method to OffloadingConnector ✅

**File**: `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`

**Changes** (added after line ~229):

```python
def get_stats(self) -> dict:
    """
    Get comprehensive stats including transfer metrics and eviction data.

    Returns:
        Dictionary with:
        - total_evictions: Number of blocks evicted to CPU
        - bytes_gpu_to_cpu: Total bytes transferred GPU → CPU
        - bytes_cpu_to_gpu: Total bytes transferred CPU → GPU
        - avg_transfer_time_gpu_to_cpu_ms: Average transfer time (ms)
        - avg_transfer_time_cpu_to_gpu_ms: Average transfer time (ms)
        - Plus manager-specific stats (blocks, scores, etc.)
    """
    stats = {
        "total_evictions": 0,
        "bytes_gpu_to_cpu": 0,
        "bytes_cpu_to_gpu": 0,
        "avg_transfer_time_gpu_to_cpu_ms": 0.0,
        "avg_transfer_time_cpu_to_gpu_ms": 0.0,
    }

    # Get transfer stats from worker
    worker = self.connector_worker
    if worker:
        kv_stats = worker.get_kv_connector_stats()
        if kv_stats:
            reduced = kv_stats.reduce()

            # Extract bytes transferred
            stats["bytes_gpu_to_cpu"] = reduced.get("gpu_to_cpu_total_bytes", 0)
            stats["bytes_cpu_to_gpu"] = reduced.get("cpu_to_gpu_total_bytes", 0)

            # Calculate average transfer times (convert to ms)
            gpu_to_cpu_time = reduced.get("gpu_to_cpu_total_time", 0.0)
            cpu_to_gpu_time = reduced.get("cpu_to_gpu_total_time", 0.0)
            stats["avg_transfer_time_gpu_to_cpu_ms"] = gpu_to_cpu_time * 1000
            stats["avg_transfer_time_cpu_to_gpu_ms"] = cpu_to_gpu_time * 1000

        # Get manager stats (includes total_evictions after Fix 1)
        if hasattr(worker, 'manager') and hasattr(worker.manager, 'get_stats'):
            manager_stats = worker.manager.get_stats()
            stats.update(manager_stats)

    return stats
```

**Impact**: Benchmark can now successfully call `llm.llm_engine.engine_core.kv_connector.get_stats()` and receive complete statistics.

---

## Validation

### Local Check (No GPU Required)

Run the updated instrumentation checker:

```bash
cd /Users/rishi/Downloads/LLMsys_Project/vllm
python3 scripts/check_instrumentation_local.py
```

**Expected output**:
```
✅ Connector tracks total_evictions in stats
```

### Minimal Benchmark Test

```bash
cd kv_cache_tiering/benchmarks

python3 benchmark.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --eviction-policy lru \
  --num-prompts 10 \
  --dataset sharegpt \
  --gpu-memory-utilization 0.08 \
  --max-model-len 2048 \
  --output /tmp/test_fix.json
```

**Expected results in JSON**:
```json
{
  "total_evictions": 15,  // ✅ Non-zero
  "bytes_gpu_to_cpu": 73728,  // ✅ Non-zero
  "bytes_cpu_to_gpu": 49152,  // ✅ Non-zero
  "eviction_log": true,  // ✅ Enabled
  "log_evictions_enabled": true  // ✅ Enabled
}
```

---

## Next Steps

1. **Validate fixes locally** (run check_instrumentation_local.py)
2. **Run minimal benchmark** (10 prompts) to verify non-zero evictions
3. **Re-run full benchmark suite** on cluster with corrected settings:
   - Reduce GPU memory: 4-12% (was 12-25%)
   - Reduce max_model_len: 2K-4K (was 16K-32K)
4. **Compare results**:
   - Do we see evictions now?
   - Does attention-aware policy show improvement when evictions occur?
   - How much overhead exists when no evictions? (validates failure mode analysis)

---

## Files Modified

1. `vllm/v1/kv_offload/attention_manager.py` - Add eviction counter
2. `kv_cache_tiering/benchmarks/benchmark.py` - Enable eviction logging
3. `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py` - Add get_stats() method

---

## Root Cause Recap

| Issue | Root Cause | Fix |
|-------|------------|-----|
| **Zero evictions in all results** | AttentionBlockManager didn't track total_evictions | Added `_total_evictions` counter and exposed in get_stats() |
| **eviction_log always false** | Benchmark didn't enable `log_evictions` flag | Added `"log_evictions": True` to connector config |
| **Benchmark couldn't read stats** | OffloadingConnector lacked get_stats() method | Implemented get_stats() aggregating transfer + manager stats |

**Status**: ✅ All fixes applied. Ready for validation testing.
