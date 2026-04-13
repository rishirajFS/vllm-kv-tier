#!/usr/bin/env python3
"""
Week 1 Test Method 2.5: Goldilocks Zone (55% GPU)

Finding the sweet spot:
- Not too tight (35% crashed)
- Not too loose (80% scheduler blocked)
- Just right: 55% GPU + aggressive concurrency

Goal: Trigger evictions without crashing initialization.
"""

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
import asyncio

async def run_goldilocks_test():
    """Goldilocks memory pressure: 55% GPU + 24 concurrent requests."""

    print("=" * 70)
    print("WEEK 1 TEST (METHOD 2.5): Goldilocks Zone - 55% GPU")
    print("=" * 70)

    # Goldilocks configuration: Not too tight, not too loose
    engine_args = AsyncEngineArgs(
        model="Qwen/Qwen2.5-7B-Instruct",
        gpu_memory_utilization=0.55,  # Sweet spot
        max_num_seqs=24,               # Very high concurrency
        max_model_len=16384,           # Long context
        enable_prefix_caching=True,
        enforce_eager=True,
        disable_log_stats=False,
        kv_cache_metrics=True,
        trust_remote_code=True
    )

    print("\nConfiguration:")
    print(f"  GPU memory: 55% (Goldilocks zone)")
    print(f"  Max concurrent seqs: 24")
    print(f"  Max model length: 16384")
    print(f"  Prefix caching: Enabled")
    print(f"  KV cache metrics: Enabled")

    print("\nInitializing engine...")
    try:
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("✓ Engine initialized successfully!")
    except ValueError as e:
        if "No available memory" in str(e):
            print("✗ Still too tight! Need more GPU memory.")
            print("  Recommendation: Try 60-65% GPU")
            return 0
        raise

    # More aggressive workload than Method 2
    # 24 requests × 10K tokens each = 240K tokens total
    long_prompts = [
        f"Request {i}: Please provide a comprehensive analysis of this extensive document. " + "word " * 10000
        for i in range(24)
    ]

    print(f"\nGenerating {len(long_prompts)} concurrent requests")
    print(f"Estimated tokens per request: ~10,000")
    print(f"Total estimated tokens: ~240,000")
    print("\nFlooding engine with maximum concurrency...")

    sampling_params = SamplingParams(max_tokens=512, temperature=0.0)

    # Process all requests concurrently to flood the engine
    async def process_request(idx, prompt, request_id):
        final_output = None
        async for request_output in engine.generate(prompt, sampling_params, request_id):
            final_output = request_output
        return final_output

    print(f"\nSubmitting all {len(long_prompts)} requests to the event loop...")
    tasks = [(i, prompt, f"req_{i}") for i, prompt in enumerate(long_prompts)]
    results = await asyncio.gather(*(process_request(i, p, rid) for i, p, rid in tasks))
    print(f"Completed {len(results)} requests")

    # Check stats using v1 metrics reader
    from vllm.v1.metrics.reader import get_metrics_snapshot, Histogram
    metrics = get_metrics_snapshot()
    total_evictions = 0
    for metric in metrics:
        if metric.name == "vllm:kv_block_lifetime_seconds" and isinstance(metric, Histogram):
            total_evictions = metric.count
            break

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"GPU Memory Utilization: 55%")
    print(f"Concurrent Requests: 24")
    print(f"Total Workload: ~240,000 tokens")
    print(f"Total KV Cache Block Evictions: {total_evictions}")
    print(f"Success threshold: >1000 evictions")
    print("=" * 70)

    if total_evictions > 1000:
        print("✅ SUCCESS: Goldilocks zone found!")
        print("   Ready for Week 2 (LongBench)")
        print("   This configuration triggers evictions")
    elif total_evictions > 0:
        print(f"⚠️  PARTIAL: Some evictions ({total_evictions}), but below threshold")
        print("   Try increasing concurrent requests to 32")
        print("   Or decrease GPU memory to 50%")
    else:
        print("❌ FAILURE: Zero evictions")
        print("   Scheduler still blocking even at 55% GPU")
        print("   Recommendation: Custom scheduler or Path B")

    print("=" * 70)

    return total_evictions

if __name__ == "__main__":
    total_evictions = asyncio.run(run_goldilocks_test())
    exit(0 if total_evictions > 1000 else 1)
