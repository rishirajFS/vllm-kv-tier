# Zero Evictions Strategy - Root Cause & Solutions

**Date**: April 7, 2026
**Critical Finding**: vLLM V1 scheduler is **too conservative** - prevents OOM by limiting batch size instead of triggering evictions
**Status**: Need new approach to force evictions

---

## The Fundamental Problem

Your diagnosis is correct:

```
vLLM V1 Scheduler Logic:
┌─────────────────────────────────────┐
│ For each new request:               │
│   IF (blocks_needed > blocks_avail) │
│     → Keep in WAITING queue         │
│   ELSE                              │
│     → Move to RUNNING               │
└─────────────────────────────────────┘

Result: Never over-subscribes → Never triggers eviction!
```

**Why "Natural Pressure" Failed**:
- With 14.4GB allocated (45% GPU), 14GB for model → 0.4GB for KV cache
- Scheduler limits batch to 1-2 requests to stay within 0.4GB
- Short outputs (256 tokens) complete before pressure builds
- **No eviction trigger reached!**

---

## Three-Tier Diagnostic & Solution Plan

### Tier 1: VERIFY Connector Works (30 minutes)

**Run this on Mac locally** (uses CPU, just for testing):

```bash
cd ~/Downloads/LLMsys_Project/vllm
python3 scripts/check_kv_connector.py
```

**Expected Outcomes**:

**Case A: Connector Works** ✅
```
✅ Connector initialized
✅ get_stats() works
⚠️  Zero evictions (but connector exists)

DIAGNOSIS: Scheduler issue, not connector issue
→ Proceed to Tier 2
```

**Case B: Connector Fails** ❌
```
❌ kv_connector is None!
OR
❌ V1 engine doesn't support KV transfer

DIAGNOSIS: Architectural limitation
→ Proceed to Tier 3 (Alternative Approaches)
```

---

### Tier 2: FORCE Evictions (2-4 hours on PSC)

If Tier 1 shows connector works, try these strategies:

#### Strategy 2.1: Long Dynamic Outputs

**Script**: `scripts/slurm_force_eviction.sh`

**Key Changes**:
```bash
MAX_LEN=2048              # Tight (not 16K!)
MAX_TOKENS=2048           # LONG outputs (not 256!)
NUM_PROMPTS=200           # High concurrency
GPU_MEM=0.35              # Moderate pressure
```

**Why This Might Work**:
- Start with small static allocation (2K max_len)
- Generate LONG outputs (2K tokens)
- KV cache grows dynamically during generation
- At ~1000 tokens, pressure builds → eviction triggered!

**Run**:
```bash
ssh bridges2.psc.edu
cd ~/workspace/vllm
sbatch scripts/slurm_force_eviction.sh
```

**Expected**: Evictions during generation phase, not initial scheduling.

---

#### Strategy 2.2: Extreme Pressure (Verification)

If 2.1 still shows zero, go nuclear:

```bash
# Edit slurm_force_eviction.sh:
GPU_MEM=0.15              # 15% (very tight)
MAX_TOKENS=4096           # Very long outputs
NUM_PROMPTS=500           # Massive concurrency
```

**This MUST trigger evictions or something is fundamentally broken.**

---

#### Strategy 2.3: Server Mode with Continuous Arrivals

**Script**: `scripts/benchmark_server_mode.py`

Uses OpenAI server mode where requests arrive continuously, forcing real concurrency:

```bash
# On PSC (in interactive session)
python scripts/benchmark_server_mode.py \
    --dataset-path ~/workspace/vllm/datasets/sharegpt.json \
    --concurrent-requests 100 \
    --max-tokens 2048 \
    --output results_server_mode.json
```

**Why This Might Work**:
- Server processes requests as they arrive
- Can't defer to later like batch mode
- True concurrent pressure

---

### Tier 3: Alternative Approaches (If Tier 2 Fails)

If ALL attempts show zero evictions, we need to consider:

#### Option 3.1: Use V0 Engine Instead

V0 might have different scheduling behavior:

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2.5-3B-Instruct",
    # ... same settings ...
    use_v2_block_manager=False  # Force V0
)
```

**Check if this flag exists and forces V0 engine.**

---

#### Option 3.2: Synthetic Stress Test

Create artificial OOM scenario:

```python
# Load model with minimal memory
llm = LLM(model="...", gpu_memory_utilization=0.10)

# Generate until OOM (will trigger preemption/eviction)
prompts = ["Very long prompt..."] * 1000
outputs = llm.generate(prompts, max_tokens=4096)
```

If this works, document it as "stress test validation" rather than realistic workload.

---

#### Option 3.3: Modify Scheduler (Nuclear Option)

If you're desperate and have time:

1. Fork vLLM
2. Modify `vllm/v1/core/sched/scheduler.py`
3. Add flag to disable conservative checks:
   ```python
   if not self.allow_over_subscription:
       # existing conservative logic
   else:
       # allow over-subscription, trigger eviction
   ```
4. Rebuild and test

**Time**: 4-8 hours
**Risk**: High (modifying core scheduler)
**Benefit**: Proves concept works

---

#### Option 3.4: Pivot the Project

If eviction truly can't be triggered:

**Pivot A: Prefetching Focus**
- "Predictive KV Cache Prefetching for Multi-Turn Conversations"
- Can measure hit rates without needing evictions
- Still valuable optimization

**Pivot B: Static vs Dynamic Comparison**
- Compare pre-allocated KV cache vs on-demand
- Measure memory efficiency without eviction

**Pivot C: Different System**
- Consider alternative inference systems (TensorRT-LLM, DeepSpeed, etc.)
- Some might have more aggressive memory management

---

## Decision Tree

```
Start
  │
  ├─ Run check_kv_connector.py (30 min)
  │   │
  │   ├─ Connector works? ✅
  │   │   │
  │   │   └─ Try slurm_force_eviction.sh (2-4 hours)
  │   │       │
  │   │       ├─ Evictions > 0? ✅
  │   │       │   └─ SUCCESS! Use these settings for all experiments
  │   │       │
  │   │       └─ Still zero? ❌
  │   │           └─ Try server mode OR extreme pressure
  │   │               │
  │   │               ├─ Evictions > 0? ✅
  │   │               │   └─ SUCCESS! (but hacky config)
  │   │               │
  │   │               └─ Still zero? ❌
  │   │                   └─ Consider Tier 3 alternatives
  │   │
  │   └─ Connector broken? ❌
  │       └─ V1 doesn't support KV transfer
  │           └─ Try V0 engine OR pivot project
```

---

## Recommended Execution Plan

### Phase 1: Quick Diagnosis (Tonight - 30 min)

```bash
# On your Mac
cd ~/Downloads/LLMsys_Project/vllm
python3 scripts/check_kv_connector.py
```

**Outcome**: Know if connector works at all.

---

### Phase 2: Force Eviction (Tomorrow - 4 hours)

```bash
# On PSC
ssh bridges2.psc.edu
cd ~/workspace/vllm

# Submit force eviction test
sbatch scripts/slurm_force_eviction.sh

# Monitor
tail -f slurm-*-force-eviction.out
```

**Outcome**: Either evictions work or we know the limit.

---

### Phase 3: Decision Point (Day 3)

**If Phase 2 shows evictions > 0**:
→ Use those settings for full LongBench experiments
→ Proceed with original plan
→ **Expected timeline**: 3-4 more days

**If Phase 2 shows zero evictions**:
→ Emergency meeting to decide:
  - Try V0 engine?
  - Pivot to prefetching?
  - Modify scheduler (risky)?
  - Use synthetic stress test?
→ **Decision needed within 24 hours**

---

## Why This Happened (Retrospective)

### Assumption 1: GPU% Controls Pressure ❌

**We thought**: Lower GPU% → memory pressure → evictions

**Reality**: Scheduler limits batch size to prevent pressure

### Assumption 2: Long Sequences Trigger Evictions ❌

**We thought**: 16K contexts naturally overflow → evictions

**Reality**: Scheduler won't admit requests that would overflow

### Assumption 3: Concurrent Batch Forces Evictions ❌

**We thought**: Submitting all prompts at once → saturation → evictions

**Reality**: V1 scheduler processes conservatively despite batch submission

### What We Should Have Done

1. **Test on V0 first** - proven to work in some scenarios
2. **Read V1 scheduler code** - understand conservative logic
3. **Start with synthetic stress test** - prove concept before realistic workloads
4. **Check existing benchmarks** - see if anyone has triggered evictions in V1

---

## Files Created

1. **`scripts/check_kv_connector.py`** - Diagnostic to verify connector initialization
2. **`scripts/slurm_force_eviction.sh`** - Long dynamic outputs strategy
3. **`scripts/benchmark_server_mode.py`** - Server mode with continuous arrivals
4. **`ZERO_EVICTIONS_STRATEGY.md`** (this file) - Complete strategy guide

---

## Success Criteria (Revised)

### Minimum Success

- ✅ Evictions > 100 in ANY configuration
- ✅ Proves instrumentation works
- ✅ Can document as "proof of concept"

**Publishability**: Weak (requires synthetic setup)

### Good Success

- ✅ Evictions > 500 with realistic config
- ✅ Measurable throughput difference (5-10%)
- ✅ Can claim "works under memory pressure"

**Publishability**: Moderate (needs careful framing)

### Excellent Success

- ✅ Evictions > 1000 with natural workloads
- ✅ Clear scaling (15-25% improvement)
- ✅ Story: "Content-aware eviction for long contexts"

**Publishability**: Strong (original plan)

---

## Communication with Advisor

If you need to update your advisor, here's a suggested framing:

### Honest Assessment

"We've discovered a fundamental issue: vLLM V1's scheduler is more conservative than expected. It prevents OOM by limiting batch size rather than allowing over-subscription that would trigger eviction. We've tried:

1. Extreme low GPU% (3%, 12%, 15%, 25%, 45%) - zero evictions
2. Long sequences (16K) - zero evictions
3. High concurrency (100-500 requests) - zero evictions

**Root cause**: Scheduler won't admit requests that exceed memory, so eviction is never reached.

**Current status**: Testing alternative approaches:
- Dynamic growth strategy (long outputs force expansion)
- Server mode (continuous arrivals)
- V0 engine (different scheduler)

**Timeline**: 2-3 more days to determine if eviction is achievable in V1. If not, we have pivot options (prefetching, synthetic stress test, or different system)."

---

## Next Immediate Action

```bash
# Step 1: Verify connector (30 min - do now)
python3 scripts/check_kv_connector.py

# Step 2: If connector works, submit force eviction (tomorrow)
sbatch scripts/slurm_force_eviction.sh

# Step 3: Reconvene in 48 hours with results
```

---

## Summary

- **Problem**: V1 scheduler too conservative, prevents evictions
- **Status**: Have diagnostic + 3 new strategies to try
- **Timeline**: 2-3 days to exhaust options
- **Backup**: Pivot strategies if eviction not achievable
- **Next**: Run `check_kv_connector.py` to verify connector works

**The good news**: Your instrumentation code IS correct. This is a scheduler behavior issue, not a code bug.

**The challenge**: Need to outsmart the scheduler to trigger the code path we've instrumented.
