#!/usr/bin/env python3
"""
Week 1 Aggressive Test: 45% GPU Memory

Most aggressive viable configuration:
- 45% GPU (between 35% crash and 50%)
- 24 concurrent requests
- Pushing the limits

Goal: Maximum evictions without crashing
"""

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
import asyncio

async def run_45pct_test():
    """45% GPU - most aggressive viable config."""

    print("=" * 70)
    print("WEEK 1 AGGRESSIVE TEST: 45% GPU Memory")
    print("=" * 70)

    engine_args = AsyncEngineArgs(
        model="Qwen/Qwen2.5-7B-Instruct",
        gpu_memory_utilization=0.45,  # Very tight
        max_num_seqs=24,
        max_model_len=16384,
        enable_prefix_caching=True,
        enforce_eager=True,
        disable_log_stats=False,
        kv_cache_metrics=True,
        trust_remote_code=True
    )

    print("\nConfiguration:")
    print(f"  GPU memory: 45% (AGGRESSIVE)")
    print(f"  Concurrent requests: 24")
    print(f"  Expected KV cache: ~30,000 tokens")
    print(f"  Workload: ~240,000 tokens")
    print(f"  Expected overflow: ~210,000 tokens")

    print("\nInitializing engine...")
    try:
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("✓ Engine initialized successfully!")
    except ValueError as e:
        if "No available memory" in str(e):
            print("✗ 45% too tight! Use 50% instead")
            return 0
        raise

    long_prompts = [
        f"Request {i}: Detailed analysis needed. " + "word " * 10000
        for i in range(24)
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
    print("RESULTS - 45% GPU TEST (AGGRESSIVE)")
    print("=" * 70)
    print(f"Total KV Cache Block Evictions: {total_evictions}")
    print(f"Success threshold: >1000 evictions")

    if total_evictions > 1000:
        print("🔥 SUCCESS! Maximum evictions achieved!")
    elif total_evictions > 131:
        print(f"✓ Better than Goldilocks! {total_evictions} evictions")
    else:
        print("⚠️  Not enough improvement")

    print("=" * 70)
    return total_evictions

if __name__ == "__main__":
    total_evictions = asyncio.run(run_45pct_test())
    exit(0 if total_evictions > 1000 else 1)
