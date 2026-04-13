#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Memory Pressure Sweep - Priority 6 from final_plan.md

Tests performance across GPU memory budgets (10%, 12%, 25%, 50%, 75%, 90%)
to show where attention-aware eviction provides the most benefit.

Hypothesis:
- At high GPU memory (75-90%): no evictions needed, all policies equal
- At medium GPU memory (25-50%): occasional evictions, small differences
- At low GPU memory (10-12%): frequent evictions, large policy differences
- Benefits correlate with eviction frequency (>5 evictions/1K tokens)

Usage:
    python scripts/benchmark_memory_pressure_sweep.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dataset-path ~/workspace/vllm/datasets/sharegpt.json \
        --output memory_pressure_sweep.json

    # Quick test with fewer prompts
    python scripts/benchmark_memory_pressure_sweep.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --num-prompts 50 \
        --gpu-levels 0.12 0.25 0.50 0.90
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
class SweepPointResult:
    """Results for a single (gpu_util, policy) combination."""
    gpu_memory_util: float
    policy: str
    model: str
    num_prompts: int
    tokens_per_second: float
    avg_latency_ms: float
    p95_latency_ms: float
    avg_ttft_ms: float
    total_output_tokens: int
    total_time_seconds: float
    eviction_count: int
    evictions_per_1k_tokens: float
    improvement_over_lru: float = 0.0


@dataclass
class SweepConfig:
    """Configuration for memory pressure sweep."""
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    gpu_levels: list[float] = field(
        default_factory=lambda: [0.10, 0.12, 0.25, 0.50, 0.75, 0.90]
    )
    policies: list[str] = field(
        default_factory=lambda: ["lru", "attention", "hybrid"]
    )
    num_prompts: int = 200
    max_tokens: int = 1024
    max_model_len: int = 8192
    cpu_bytes: int = 8_000_000_000
    dataset_path: str | None = None


def load_prompts(config: SweepConfig) -> list[str]:
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

    # Synthetic prompts with varying lengths for realistic pressure
    topics = [
        "Explain the theory of relativity and its implications for modern physics",
        "Describe the process of photosynthesis in plants and its importance",
        "Write a detailed overview of the history of computer science",
        "Discuss the economic implications of artificial intelligence adoption",
        "Explain how neural networks learn through backpropagation",
        "Describe the architecture of modern operating systems",
        "Write about the evolution of programming languages over decades",
        "Discuss the challenges and solutions in distributed computing",
        "Explain the mathematics behind quantum computing",
        "Describe the principles of compiler design and optimization",
    ]
    prompts = []
    for i in range(config.num_prompts):
        topic = topics[i % len(topics)]
        # Add varying context to create different sequence lengths
        context = f"Context {i}: " + "additional background information. " * (i % 20)
        prompts.append(f"{context}\n\n{topic}. Provide a comprehensive answer.")
    return prompts


def run_sweep_point(
    config: SweepConfig,
    gpu_util: float,
    policy: str,
    prompts: list[str],
) -> SweepPointResult:
    """
    Run benchmark for a single (gpu_util, policy) combination.

    Args:
        config: Sweep configuration
        gpu_util: GPU memory utilization fraction
        policy: Eviction policy
        prompts: List of prompts to run

    Returns:
        SweepPointResult with metrics
    """
    import torch
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    print(f"\n{'='*70}")
    print(f"GPU Memory: {gpu_util*100:.0f}% | Policy: {policy}")
    print(f"{'='*70}")

    # Build KV config - only enable tiering for low GPU utilization
    kv_config = None
    if gpu_util < 0.80:
        extra_config = {
            "cpu_bytes_to_use": config.cpu_bytes,
            "block_size": 48,
            "eviction_policy": policy,
        }
        if policy == "attention":
            extra_config["score_decay"] = 0.95
        elif policy == "hybrid":
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
        gpu_memory_utilization=gpu_util,
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

    # Collect metrics
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

    # Try to get eviction count
    eviction_count = 0
    try:
        engine = getattr(llm, "llm_engine", None)
        if engine:
            core = getattr(engine, "engine_core", None)
            if core:
                connector = getattr(core, "kv_connector", None)
                if connector:
                    stats = connector.get_stats()
                    eviction_count = stats.get("total_evictions", 0)
    except Exception:
        pass

    evictions_per_1k = (eviction_count / total_tokens * 1000) if total_tokens > 0 else 0

    result = SweepPointResult(
        gpu_memory_util=gpu_util,
        policy=policy,
        model=config.model,
        num_prompts=len(outputs),
        tokens_per_second=throughput,
        avg_latency_ms=avg_lat,
        p95_latency_ms=p95_lat,
        avg_ttft_ms=avg_ttft,
        total_output_tokens=total_tokens,
        total_time_seconds=total_time,
        eviction_count=eviction_count,
        evictions_per_1k_tokens=evictions_per_1k,
    )

    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"  Throughput: {throughput:.1f} tok/s")
    print(f"  Avg Latency: {avg_lat:.1f} ms")
    print(f"  Evictions: {eviction_count} ({evictions_per_1k:.1f}/1K tokens)")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Memory Pressure Sweep: test performance across GPU memory budgets"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Path to dataset JSON (e.g. ShareGPT)")
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--cpu-bytes", type=int, default=8_000_000_000)
    parser.add_argument("--gpu-levels", type=float, nargs="+",
                        default=[0.10, 0.12, 0.25, 0.50, 0.75, 0.90],
                        help="GPU memory utilization levels to sweep")
    parser.add_argument("--policies", type=str, nargs="+",
                        default=["lru", "attention", "hybrid"])
    parser.add_argument("--output", type=Path,
                        default=Path("memory_pressure_sweep.json"))

    args = parser.parse_args()

    config = SweepConfig(
        model=args.model,
        gpu_levels=args.gpu_levels,
        policies=args.policies,
        num_prompts=args.num_prompts,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        cpu_bytes=args.cpu_bytes,
        dataset_path=args.dataset_path,
    )

    print(f"\n{'#'*70}")
    print(f"# Memory Pressure Sweep")
    print(f"# Model: {config.model}")
    print(f"# GPU Levels: {config.gpu_levels}")
    print(f"# Policies: {config.policies}")
    print(f"# Prompts: {config.num_prompts}")
    print(f"{'#'*70}\n")

    prompts = load_prompts(config)
    print(f"Loaded {len(prompts)} prompts")

    all_results: list[SweepPointResult] = []

    for gpu_util in config.gpu_levels:
        print(f"\n{'#'*70}")
        print(f"# GPU MEMORY LEVEL: {gpu_util*100:.0f}%")
        print(f"{'#'*70}")

        level_results: list[SweepPointResult] = []

        for policy in config.policies:
            try:
                result = run_sweep_point(config, gpu_util, policy, prompts)
                level_results.append(result)
                all_results.append(result)
            except Exception as e:
                print(f"  ERROR: {policy} at {gpu_util*100:.0f}% GPU: {e}")
                continue

        # Calculate improvements over LRU
        lru_result = next((r for r in level_results if r.policy == "lru"), None)
        if lru_result and lru_result.tokens_per_second > 0:
            for result in level_results:
                if result.policy != "lru":
                    result.improvement_over_lru = (
                        (result.tokens_per_second - lru_result.tokens_per_second)
                        / lru_result.tokens_per_second * 100
                    )

        # Print level summary
        print(f"\n--- Summary for {gpu_util*100:.0f}% GPU ---")
        for r in level_results:
            tag = f"{r.improvement_over_lru:+.1f}%" if r.policy != "lru" else "baseline"
            print(f"  {r.policy:10s}: {r.tokens_per_second:6.1f} tok/s ({tag}) "
                  f"[{r.evictions_per_1k_tokens:.1f} evictions/1K]")

    # Save results
    with open(args.output, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)

    # Print final summary table
    print(f"\n{'='*80}")
    print("MEMORY PRESSURE SWEEP RESULTS")
    print(f"{'='*80}")
    print(f"{'GPU %':>6} | {'Policy':>10} | {'Throughput':>10} | {'Improvement':>11} | {'Evict/1K':>8}")
    print(f"{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*11}-+-{'-'*8}")

    for gpu_util in config.gpu_levels:
        for r in all_results:
            if r.gpu_memory_util == gpu_util:
                tag = f"{r.improvement_over_lru:+.1f}%" if r.policy != "lru" else "—"
                print(f"{gpu_util*100:5.0f}% | {r.policy:>10} | {r.tokens_per_second:8.1f} | "
                      f"{tag:>11} | {r.evictions_per_1k_tokens:6.1f}")
        print(f"{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*11}-+-{'-'*8}")

    # Print key insight
    print(f"\nKey Insight: Benefit vs Memory Pressure")
    print(f"{'GPU %':>6} | {'Attention Improvement':>20} | {'Evictions/1K':>12}")
    print(f"{'-'*6}-+-{'-'*20}-+-{'-'*12}")
    for gpu_util in config.gpu_levels:
        attn = next((r for r in all_results
                     if r.gpu_memory_util == gpu_util and r.policy == "attention"), None)
        if attn:
            print(f"{gpu_util*100:5.0f}% | {attn.improvement_over_lru:>19.1f}% | {attn.evictions_per_1k_tokens:>10.1f}")

    print(f"\nResults saved to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
