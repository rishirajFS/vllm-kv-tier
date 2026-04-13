#!/usr/bin/env python3
"""
Week 1 Alternative Test: 32 Concurrent Requests @ 55% GPU

Alternative approach:
- Keep 55% GPU (known to work from Goldilocks)
- Increase concurrency from 24 → 32 requests
- More simultaneous pressure

Goal: Force more evictions through higher concurrency
"""

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
import asyncio

async def run_32concurrent_test():
    """32 concurrent requests at 55% GPU."""

    print("=" * 70)
    print("WEEK 1 ALTERNATIVE TEST: 32 Concurrent Requests")
    print("=" * 70)

    engine_args = AsyncEngineArgs(
        model="Qwen/Qwen2.5-7B-Instruct",
        gpu_memory_utilization=0.55,  # Same as Goldilocks
        max_num_seqs=32,                # Higher than Goldilocks (24)
        max_model_len=16384,
        enable_prefix_caching=True,
        enforce_eager=True,
        disable_log_stats=False,
        kv_cache_metrics=True,
        trust_remote_code=True
    )

    print("\nConfiguration:")
    print(f"  GPU memory: 55% (same as Goldilocks)")
    print(f"  Concurrent requests: 32 (vs Goldilocks 24)")
    print(f"  Expected KV cache: ~53,712 tokens")
    print(f"  Workload: ~320,000 tokens (32 × 10K)")
    print(f"  Expected overflow: ~266,000 tokens")

    print("\nInitializing engine...")
    try:
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("✓ Engine initialized successfully!")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return 0

    long_prompts = [
        f"Request {i}: Complete analysis required. " + "word " * 10000
        for i in range(32)  # 32 concurrent
    ]

    print(f"\nProcessing {len(long_prompts)} concurrent requests...")
    sampling_params = SamplingParams(max_tokens=512, temperature=0.0)

    async def process_request(idx, prompt, request_id):
        final_output = None
        async for request_output in engine.generate(prompt, sampling_params, request_id):
            final_output = request_output
        return final_output

    tasks = [(i, prompt, f"req_{i}") for i, prompt in enumerate(long_prompts)]
    results = await asyncio.gather(*(process_request(i, p, rid) for i, p, rid in tasks))
    print(f"Completed {len(results)} requests")

    from vllm.v1.metrics.reader import get_metrics_snapshot, Histogram
    metrics = get_metrics_snapshot()
    total_evictions = 0
    for metric in metrics:
        if metric.name == "vllm:kv_block_lifetime_seconds" and isinstance(metric, Histogram):
            total_evictions = metric.count
            break

    print("\n" + "=" * 70)
    print("RESULTS - 32 CONCURRENT TEST")
    print("=" * 70)
    print(f"Total KV Cache Block Evictions: {total_evictions}")
    print(f"Goldilocks baseline: 131 evictions (24 concurrent)")
    print(f"Success threshold: >1000 evictions")

    if total_evictions > 1000:
        print("🚀 SUCCESS! Higher concurrency worked!")
    elif total_evictions > 131:
        print(f"✓ Improvement! {total_evictions} > 131 (Goldilocks)")
        print(f"  Increase: {total_evictions - 131} more evictions")
    else:
        print("⚠️  No improvement from higher concurrency")

    print("=" * 70)
    return total_evictions

if __name__ == "__main__":
    total_evictions = asyncio.run(run_32concurrent_test())
    exit(0 if total_evictions > 1000 else 1)
