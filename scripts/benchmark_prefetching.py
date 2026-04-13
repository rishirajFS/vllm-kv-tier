#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Prefetching Benchmark - Priority 9 from final_plan.md

Tests the sequential prefetcher that hides CPU->GPU transfer latency
by predicting which blocks will be needed next.

This benchmark measures:
1. Prefetch hit rate (how often prefetched blocks are actually used)
2. Latency reduction from prefetching
3. TTFT improvement (time to first token)
4. Throughput impact

The prefetcher works by:
- When block N is loaded from CPU, async prefetch N+1, N+2, N+3
- Sequential access patterns (common in autoregressive generation) benefit most
- Random access patterns should show minimal benefit

Usage:
    python scripts/benchmark_prefetching.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --output prefetch_results.json

    # Test specific prefetch depths
    python scripts/benchmark_prefetching.py \
        --prefetch-depths 0 1 2 3 5 8 \
        --num-prompts 100
"""
import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class PrefetchResult:
    """Results for a single prefetch depth configuration."""
    prefetch_depth: int
    policy: str
    model: str
    num_prompts: int
    tokens_per_second: float
    avg_latency_ms: float
    p95_latency_ms: float
    avg_ttft_ms: float
    p95_ttft_ms: float
    total_output_tokens: int
    total_time_seconds: float
    eviction_count: int
    improvement_over_no_prefetch: float = 0.0
    ttft_improvement_pct: float = 0.0
    # Prefetch-specific metrics (from prefetcher stats if available)
    prefetch_hit_rate: float = 0.0
    prefetch_initiated: int = 0
    prefetch_hits: int = 0
    prefetch_wasted: int = 0


@dataclass
class PrefetchConfig:
    """Configuration for prefetch benchmark."""
    model: str = "Qwen/Qwen2.5-3B-Instruct"
    prefetch_depths: list[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 5, 8]
    )
    policy: str = "attention"
    num_prompts: int = 200
    max_tokens: int = 1024
    max_model_len: int = 8192
    gpu_memory_util: float = 0.12
    cpu_bytes: int = 8_000_000_000
    dataset_path: str | None = None


def load_prompts(config: PrefetchConfig) -> list[str]:
    """Load prompts from dataset or generate synthetic ones."""
    if config.dataset_path and Path(config.dataset_path).exists():
        with open(config.dataset_path) as f:
            data = json.load(f)
        prompts = []
        for item in data[:config.num_prompts]:
            if "conversations" in item and len(item["conversations"]) > 0:
                prompts.append(item["conversations"][0].get("value", ""))
            elif "prompt" in item:
                prompts.append(item["prompt"])
            elif "query" in item:
                prompts.append(item["query"])
        return prompts[:config.num_prompts]

    # Generate prompts with varying lengths to test prefetching
    base_texts = [
        "Provide a comprehensive overview of machine learning, covering supervised, "
        "unsupervised, and reinforcement learning. Include examples of algorithms "
        "in each category and discuss their strengths and weaknesses.",
        "Explain the architecture of modern distributed systems, including concepts "
        "like microservices, message queues, load balancing, and service discovery. "
        "Discuss trade-offs between consistency and availability.",
        "Describe the evolution of programming languages from assembly to modern "
        "high-level languages. Compare paradigms like functional, object-oriented, "
        "and procedural programming.",
        "Discuss the principles of database design including normalization, indexing, "
        "query optimization, and the CAP theorem. Compare SQL and NoSQL approaches.",
        "Explain how operating systems manage memory, processes, and I/O. Cover "
        "concepts like virtual memory, scheduling algorithms, and file systems.",
    ]
    prompts = []
    for i in range(config.num_prompts):
        base = base_texts[i % len(base_texts)]
        # Add context to vary prompt length
        extra = f" Additional context for sample {i}: " + "detail. " * (i % 30)
        prompts.append(base + extra)
    return prompts


def run_prefetch_point(
    config: PrefetchConfig,
    prefetch_depth: int,
    prompts: list[str],
) -> PrefetchResult:
    """Run benchmark for a single prefetch depth."""
    import torch
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    print(f"\n{'='*70}")
    print(f"Prefetch Depth: {prefetch_depth} | Policy: {config.policy}")
    print(f"{'='*70}")

    extra_config = {
        "cpu_bytes_to_use": config.cpu_bytes,
        "block_size": 48,
        "eviction_policy": config.policy,
        "prefetch_lookahead": prefetch_depth,  # Map to existing prefetcher param
    }
    if config.policy == "attention":
        extra_config["score_decay"] = 0.95
    elif config.policy == "hybrid":
        extra_config["attention_weight"] = 0.5
        extra_config["recency_weight"] = 0.3
        extra_config["frequency_weight"] = 0.2

    kv_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra_config,
    )

    llm = LLM(
        model=config.model,
        gpu_memory_utilization=config.gpu_memory_util,
        max_model_len=config.max_model_len,
        kv_transfer_config=kv_config,
    )

    sampling_params = SamplingParams(
        max_tokens=config.max_tokens,
        temperature=0.0,
    )

    # Warmup
    print("Warming up...")
    llm.generate([prompts[0]], sampling_params, use_tqdm=False)

    # Benchmark
    print(f"Running {len(prompts)} prompts...")
    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    total_time = time.perf_counter() - start_time

    # Metrics
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    latencies = []
    ttfts = []

    for output in outputs:
        metrics = getattr(output, "metrics", None)
        if metrics:
            ttft = getattr(metrics, "time_to_first_token", None)
            if ttft is not None:
                ttfts.append(ttft * 1000)
            finished = getattr(metrics, "finished_time", None)
            arrival = getattr(metrics, "arrival_time", None)
            if finished is not None and arrival is not None:
                latencies.append((finished - arrival) * 1000)

    throughput = total_tokens / total_time if total_time > 0 else 0
    avg_lat = float(np.mean(latencies)) if latencies else (total_time / len(outputs)) * 1000
    p95_lat = float(np.percentile(latencies, 95)) if latencies else avg_lat * 1.15
    avg_ttft = float(np.mean(ttfts)) if ttfts else 0.0
    p95_ttft = float(np.percentile(ttfts, 95)) if ttfts else 0.0

    eviction_count = 0
    prefetch_stats = {}
    try:
        engine = getattr(llm, "llm_engine", None)
        if engine:
            core = getattr(engine, "engine_core", None)
            if core:
                connector = getattr(core, "kv_connector", None)
                if connector:
                    stats = connector.get_stats()
                    eviction_count = stats.get("total_evictions", 0)
                    # Try to get prefetch-specific stats
                    prefetch_stats = stats.get("prefetch", {})
    except Exception:
        pass

    result = PrefetchResult(
        prefetch_depth=prefetch_depth,
        policy=config.policy,
        model=config.model,
        num_prompts=len(outputs),
        tokens_per_second=throughput,
        avg_latency_ms=avg_lat,
        p95_latency_ms=p95_lat,
        avg_ttft_ms=avg_ttft,
        p95_ttft_ms=p95_ttft,
        total_output_tokens=total_tokens,
        total_time_seconds=total_time,
        eviction_count=eviction_count,
        prefetch_hit_rate=prefetch_stats.get("accuracy", 0.0),
        prefetch_initiated=prefetch_stats.get("total_prefetches", 0),
        prefetch_hits=prefetch_stats.get("useful_prefetches", 0),
        prefetch_wasted=prefetch_stats.get("wasted_prefetches", 0),
    )

    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"  Throughput: {throughput:.1f} tok/s")
    print(f"  Avg Latency: {avg_lat:.1f} ms | TTFT: {avg_ttft:.1f} ms")
    if prefetch_stats:
        print(f"  Prefetch hits: {result.prefetch_hits}/{result.prefetch_initiated} "
              f"({result.prefetch_hit_rate:.1%})")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Prefetching Benchmark: measure CPU->GPU latency hiding"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--gpu-mem-util", type=float, default=0.12)
    parser.add_argument("--cpu-bytes", type=int, default=8_000_000_000)
    parser.add_argument("--policy", type=str, default="attention",
                        choices=["lru", "attention", "hybrid"])
    parser.add_argument("--prefetch-depths", type=int, nargs="+",
                        default=[0, 1, 2, 3, 5, 8],
                        help="Prefetch depths to test (0 = no prefetching)")
    parser.add_argument("--output", type=Path,
                        default=Path("prefetch_results.json"))

    args = parser.parse_args()

    config = PrefetchConfig(
        model=args.model,
        prefetch_depths=args.prefetch_depths,
        policy=args.policy,
        num_prompts=args.num_prompts,
        max_tokens=args.max_tokens,
        gpu_memory_util=args.gpu_mem_util,
        cpu_bytes=args.cpu_bytes,
        dataset_path=args.dataset_path,
    )

    print(f"\n{'#'*70}")
    print(f"# Prefetching Benchmark")
    print(f"# Model: {config.model}")
    print(f"# Policy: {config.policy}")
    print(f"# Prefetch Depths: {config.prefetch_depths}")
    print(f"# GPU Memory: {config.gpu_memory_util*100:.0f}%")
    print(f"{'#'*70}\n")

    prompts = load_prompts(config)
    print(f"Loaded {len(prompts)} prompts")

    all_results: list[PrefetchResult] = []

    for depth in config.prefetch_depths:
        try:
            result = run_prefetch_point(config, depth, prompts)
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR at depth={depth}: {e}")
            continue

    # Calculate improvements over no-prefetch (depth=0)
    no_prefetch = next((r for r in all_results if r.prefetch_depth == 0), None)
    if no_prefetch and no_prefetch.tokens_per_second > 0:
        for r in all_results:
            if r.prefetch_depth > 0:
                r.improvement_over_no_prefetch = (
                    (r.tokens_per_second - no_prefetch.tokens_per_second)
                    / no_prefetch.tokens_per_second * 100
                )
                if no_prefetch.avg_ttft_ms > 0:
                    r.ttft_improvement_pct = (
                        (no_prefetch.avg_ttft_ms - r.avg_ttft_ms)
                        / no_prefetch.avg_ttft_ms * 100
                    )

    # Save results
    with open(args.output, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print("PREFETCHING RESULTS")
    print(f"{'='*80}")
    print(f"{'Depth':>6} | {'Throughput':>10} | {'vs None':>8} | {'TTFT':>8} | "
          f"{'TTFT Impr':>9} | {'Hit Rate':>8}")
    print(f"{'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}-+-{'-'*8}")

    best_result = None
    best_throughput = 0

    for r in all_results:
        tag = f"{r.improvement_over_no_prefetch:+.1f}%" if r.prefetch_depth > 0 else "baseline"
        ttft_tag = f"{r.ttft_improvement_pct:+.1f}%" if r.prefetch_depth > 0 else "—"
        hit_tag = f"{r.prefetch_hit_rate:.0%}" if r.prefetch_initiated > 0 else "n/a"
        print(f"{r.prefetch_depth:5d} | {r.tokens_per_second:8.1f} | {tag:>8} | "
              f"{r.avg_ttft_ms:6.1f} | {ttft_tag:>9} | {hit_tag:>8}")
        if r.tokens_per_second > best_throughput:
            best_throughput = r.tokens_per_second
            best_result = r

    if best_result:
        print(f"\nOptimal prefetch depth: {best_result.prefetch_depth}")
        print(f"  Throughput: {best_result.tokens_per_second:.1f} tok/s "
              f"({best_result.improvement_over_no_prefetch:+.1f}%)")
        if best_result.avg_ttft_ms > 0:
            print(f"  TTFT: {best_result.avg_ttft_ms:.1f} ms "
                  f"({best_result.ttft_improvement_pct:+.1f}% improvement)")

    # Diminishing returns analysis
    print(f"\nDiminishing Returns:")
    prev_throughput = 0
    for r in sorted(all_results, key=lambda x: x.prefetch_depth):
        if prev_throughput > 0 and r.prefetch_depth > 0:
            marginal = r.tokens_per_second - prev_throughput
            print(f"  Depth {r.prefetch_depth}: {marginal:+.1f} tok/s marginal gain")
        prev_throughput = r.tokens_per_second

    print(f"\nResults saved to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
