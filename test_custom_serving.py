#!/usr/bin/env python3
"""
Week 1 Test Method 2: Custom serving loop (more aggressive).

Uses AsyncLLMEngine directly to flood with concurrent long requests.
Goal: Bypass scheduler admission control by overwhelming capacity.
"""

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
import asyncio

async def run_custom_serving():
    """Flood engine with concurrent long requests."""

    print("=" * 70)
    print("WEEK 1 TEST (METHOD 2): Custom Serving with High Concurrency")
    print("=" * 70)

    # Recalibrated for V100-32GB: 35% was too low for 7B model weights
    engine_args = AsyncEngineArgs(
        model="Qwen/Qwen2.5-7B-Instruct",
        gpu_memory_utilization=0.60,   # 60% (Enough for weights + ~5GB cache)
        max_num_seqs=32,               # High concurrency
        max_model_len=16384,           # Long context
        enable_prefix_caching=True,
        enforce_eager=True,
        disable_log_stats=False
    )

    print("\nConfiguration:")
    print(f"  GPU memory: 60%")
    print(f"  Max concurrent seqs: 32")
    print(f"  Max model length: 16384")
    print(f"  Prefix caching: Enabled")

    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Flood with long requests simultaneously
    # 32 requests × 8K tokens each = 256K tokens total
    # With 60% GPU memory (~5GB cache), 256K tokens (approx 16GB) MUST trigger tiering
    long_prompts = [
        f"Request {i}: Please analyze this extensive document in detail. " + "token " * 8000
        for i in range(32)
    ]

    print(f"\nGenerating {len(long_prompts)} concurrent requests")
    print(f"Estimated tokens per request: ~8,000")
    print(f"Total estimated tokens: ~128,000")
    print("\nFlooding engine with requests...")

    sampling_params = SamplingParams(max_tokens=512, temperature=0.0)

    # Submit all requests concurrently
    tasks = []
    for i, prompt in enumerate(long_prompts):
        request_id = f"req_{i}"
        # Note: async_generate returns an async generator
        tasks.append((i, prompt, request_id))

    # Process all requests concurrently to flood the engine
    async def process_request(idx, prompt, request_id):
        final_output = None
        async for request_output in engine.generate(prompt, sampling_params, request_id):
            final_output = request_output
        return final_output

    print(f"\nSubmitting all {len(tasks)} requests to the event loop...")
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
    print(f"Total KV Cache Block Evictions: {total_evictions}")
    print(f"Success threshold: >1000 evictions")

    if total_evictions > 1000:
        print("✅ SUCCESS: Ready for Week 2 (LongBench)")
        print("   This method bypasses scheduler effectively")
    elif total_evictions > 0:
        print(f"⚠️  PARTIAL: Some evictions ({total_evictions}), but below threshold")
        print("   Try increasing concurrent requests or context length")
    else:
        print("❌ FAILURE: Zero evictions")
        print("   Even aggressive flooding didn't work")
        print("   Recommendation: Proceed with Path B (workshop paper)")

    print("=" * 70)

    return total_evictions

if __name__ == "__main__":
    total_evictions = asyncio.run(run_custom_serving())
    exit(0 if total_evictions > 1000 else 1)
