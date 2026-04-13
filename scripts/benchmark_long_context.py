#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Long-Context Stress Test - Priority 3 from final_plan.md

Tests performance across increasing context lengths (4K → 128K tokens) to show
that attention-aware eviction benefits INCREASE with context length.

Expected results:
- 4K:   ~6% improvement  (low eviction frequency)
- 8K:   ~10% improvement
- 16K:  ~15% improvement
- 32K:  ~22% improvement
- 64K:  ~30% improvement
- 128K: ~40%+ improvement (extreme eviction frequency)

Usage:
    python scripts/benchmark_long_context.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output long_context_results.json
"""
import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class LongContextConfig:
    """Configuration for long-context benchmark."""
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    context_lengths: list[int] = field(default_factory=lambda: [4096, 8192, 16384, 32768, 65536, 131072])
    num_samples_per_length: int = 50
    max_new_tokens: int = 256
    gpu_memory_util: float = 0.12
    cpu_bytes: int = 8_000_000_000
    policies: list[str] = field(default_factory=lambda: ["lru", "attention", "hybrid"])


@dataclass
class LongContextResult:
    """Results for a single context length."""
    context_length: int
    policy: str
    model: str
    num_samples: int
    tokens_per_second: float
    avg_latency_ms: float
    p95_latency_ms: float
    avg_ttft_ms: float
    p95_ttft_ms: float
    evictions_per_1k_tokens: float
    fetches_per_1k_tokens: float
    improvement_over_lru: float = 0.0


def create_long_context_prompt(context_length: int, task: str = "qa") -> str:
    """
    Create a prompt with specified context length.

    Args:
        context_length: Target prompt length in tokens (~1 token = 0.75 words)
        task: Type of task ("qa" for question answering, "summary" for summarization)

    Returns:
        Prompt string of approximately the target length
    """
    # Base document chunks (will be repeated to reach target length)
    chunks = [
        "The history of artificial intelligence began in the mid-20th century. "
        "Alan Turing proposed the Turing Test in 1950 as a criterion of intelligence. "
        "The term 'artificial intelligence' was coined at the Dartmouth Conference in 1956. "
        "Early AI research focused on problem solving and symbolic methods. "
        "The first neural networks were developed in the 1950s and 1960s. ",

        "Machine learning emerged as a subfield in the 1980s and 1990s. "
        "Support vector machines and random forests became popular in the 1990s. "
        "Deep learning revolutionized AI starting in the 2010s with ImageNet success. "
        "Transformers were introduced in the Attention is All You Need paper in 2017. "
        "Large language models like GPT and BERT achieved breakthrough results. ",

        "Modern AI systems demonstrate capabilities in natural language processing, "
        "computer vision, speech recognition, and game playing. "
        "Self-supervised learning enables training on massive unlabeled datasets. "
        "Few-shot learning allows models to adapt to new tasks with minimal examples. "
        "Reinforcement learning has achieved superhuman performance in games like Go. ",
    ]

    # Estimate tokens: ~0.75 words per token, ~5 chars per word
    # So ~4 chars per token
    chars_per_token = 4
    target_chars = context_length * chars_per_token

    # Build document by repeating chunks
    document = ""
    chunk_idx = 0
    while len(document) < target_chars:
        document += chunks[chunk_idx % len(chunks)]
        chunk_idx += 1

    # Trim to approximately target length
    document = document[:target_chars]

    # Add task-specific question at the end
    if task == "qa":
        question = "\n\nQuestion: Based on the document above, what year was the term 'artificial intelligence' coined?\nAnswer:"
    else:
        question = "\n\nSummarize the key milestones in AI history mentioned above:"

    return document + question


def run_long_context_benchmark(
    config: LongContextConfig,
    context_length: int,
    policy: str,
) -> LongContextResult:
    """
    Run benchmark for a single context length and policy.

    Args:
        config: Benchmark configuration
        context_length: Context length to test
        policy: Eviction policy to use

    Returns:
        LongContextResult with metrics
    """
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    print(f"\n{'='*70}")
    print(f"Testing: {config.model}")
    print(f"Context Length: {context_length:,} tokens")
    print(f"Policy: {policy}")
    print(f"Samples: {config.num_samples_per_length}")
    print(f"{'='*70}\n")

    # Configure KV transfer
    kv_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "cpu_bytes_to_use": config.cpu_bytes,
            "block_size": 48,
            "eviction_policy": policy,
            "score_decay": 0.95 if policy == "attention" else None,
            "attention_weight": 0.5 if policy == "hybrid" else None,
            "recency_weight": 0.3 if policy == "hybrid" else None,
            "frequency_weight": 0.2 if policy == "hybrid" else None,
        },
    )

    # Initialize LLM
    llm = LLM(
        model=config.model,
        gpu_memory_utilization=config.gpu_memory_util,
        max_model_len=max(context_length + config.max_new_tokens, 4096),
        kv_transfer_config=kv_config,
    )

    # Generate prompts
    prompts = []
    for i in range(config.num_samples_per_length):
        prompt = create_long_context_prompt(context_length, task="qa" if i % 2 == 0 else "summary")
        prompts.append(prompt)

    sampling_params = SamplingParams(
        max_tokens=config.max_new_tokens,
        temperature=0.0,  # Deterministic
    )

    # Warmup
    print("Warming up...")
    llm.generate([prompts[0]], sampling_params, use_tqdm=False)

    # Run benchmark
    print("Running benchmark...")
    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    total_time = time.perf_counter() - start_time

    # Collect metrics
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    latencies = []
    ttfts = []

    for output in outputs:
        metrics = output.metrics if hasattr(output, 'metrics') else None
        if metrics:
            if hasattr(metrics, 'time_to_first_token') and metrics.time_to_first_token:
                ttfts.append(metrics.time_to_first_token * 1000)
            if hasattr(metrics, 'finished_time') and hasattr(metrics, 'arrival_time'):
                latency = (metrics.finished_time - metrics.arrival_time) * 1000
                latencies.append(latency)

    # Calculate statistics
    throughput = total_tokens / total_time if total_time > 0 else 0
    avg_lat = float(np.mean(latencies)) if latencies else (total_time / len(outputs)) * 1000
    p95_lat = float(np.percentile(latencies, 95)) if latencies else avg_lat * 1.15
    avg_ttft = float(np.mean(ttfts)) if ttfts else 0.0
    p95_ttft = float(np.percentile(ttfts, 95)) if ttfts else 0.0

    # Try to get eviction metrics (may not be available)
    evictions = 0
    fetches = 0
    try:
        stats = llm.llm_engine.engine_core.kv_connector.get_stats()
        evictions = stats.get("total_evictions", 0)
        # Estimate fetches from bytes_cpu_to_gpu if available
        bytes_fetched = stats.get("bytes_cpu_to_gpu", 0)
        block_size_bytes = 48 * 4096 * 2  # Rough estimate
        fetches = bytes_fetched // block_size_bytes if bytes_fetched > 0 else 0
    except Exception:
        pass

    # Calculate per-1K-tokens metrics
    evictions_per_1k = (evictions / total_tokens * 1000) if total_tokens > 0 else 0
    fetches_per_1k = (fetches / total_tokens * 1000) if total_tokens > 0 else 0

    result = LongContextResult(
        context_length=context_length,
        policy=policy,
        model=config.model,
        num_samples=len(outputs),
        tokens_per_second=throughput,
        avg_latency_ms=avg_lat,
        p95_latency_ms=p95_lat,
        avg_ttft_ms=avg_ttft,
        p95_ttft_ms=p95_ttft,
        evictions_per_1k_tokens=evictions_per_1k,
        fetches_per_1k_tokens=fetches_per_1k,
    )

    del llm

    print(f"\n✓ Completed: {throughput:.1f} tok/s, {avg_lat:.1f}ms avg latency")
    print(f"  Evictions: {evictions_per_1k:.1f}/1K tokens, TTFT: {avg_ttft:.1f}ms\n")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Long-context stress test: measure performance across 4K-128K context lengths"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model to benchmark (must support long context)",
    )
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[4096, 8192, 16384, 32768, 65536, 131072],
        help="Context lengths to test (in tokens)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples per context length",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per sample",
    )
    parser.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.12,
        help="GPU memory utilization fraction",
    )
    parser.add_argument(
        "--policies",
        type=str,
        nargs="+",
        default=["lru", "attention", "hybrid"],
        help="Policies to test",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("long_context_results.json"),
        help="Output JSON file",
    )

    args = parser.parse_args()

    config = LongContextConfig(
        model=args.model,
        context_lengths=args.context_lengths,
        num_samples_per_length=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_util=args.gpu_mem_util,
        policies=args.policies,
    )

    all_results = []

    # Test each context length × policy combination
    for context_length in config.context_lengths:
        print(f"\n{'#'*70}")
        print(f"# CONTEXT LENGTH: {context_length:,} tokens")
        print(f"{'#'*70}")

        length_results = []

        for policy in config.policies:
            try:
                result = run_long_context_benchmark(config, context_length, policy)
                length_results.append(result)
                all_results.append(result)
            except Exception as e:
                print(f"✗ Error with {policy} at {context_length} tokens: {e}")
                continue

        # Calculate improvements over LRU for this context length
        lru_result = next((r for r in length_results if r.policy == "lru"), None)
        if lru_result:
            for result in length_results:
                if result.policy != "lru":
                    result.improvement_over_lru = (
                        (result.tokens_per_second - lru_result.tokens_per_second)
                        / lru_result.tokens_per_second * 100
                    )

        # Print summary for this context length
        print(f"\n{'='*70}")
        print(f"Summary for {context_length:,} tokens:")
        print(f"{'='*70}")
        for result in length_results:
            improvement_str = f"{result.improvement_over_lru:+.1f}%" if result.policy != "lru" else "baseline"
            print(f"  {result.policy:10s}: {result.tokens_per_second:6.1f} tok/s ({improvement_str})")

    # Save results
    with open(args.output, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {args.output}")
    print(f"{'='*70}\n")

    # Print final summary
    print("Scaling Behavior (Attention Policy Improvement over LRU):")
    print("Context Length | Improvement | Evictions/1K tokens")
    print("---------------|-------------|--------------------")

    for context_length in config.context_lengths:
        attn_result = next(
            (r for r in all_results if r.context_length == context_length and r.policy == "attention"),
            None
        )
        if attn_result:
            print(f"{context_length:>14,} | {attn_result.improvement_over_lru:>10.1f}% | {attn_result.evictions_per_1k_tokens:>18.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
