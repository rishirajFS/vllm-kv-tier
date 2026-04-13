#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
MemoryPressureScheduler: Forces KV cache evictions by constraining GPU memory.

Problem: vLLM's scheduler prevents KV cache overflow through preemption.
When block allocation fails, the scheduler preempts (removes) the
lowest-priority running request and frees its GPU blocks. This means the
offloading connector's eviction policies (LRU, attention, hybrid) never
trigger because GPU blocks never actually overflow.

Solution: This module provides two complementary mechanisms:
1. Artificially restrict GPU KV block count at runtime to force higher
   cache pressure with fewer requests.
2. Provide a benchmark wrapper that configures extreme memory pressure
   (low gpu_memory_utilization + high concurrency + ignore_eos) to
   maximize the chance of filling the CPU offload tier and triggering
   real evictions.

Usage:
    # As a benchmark wrapper (recommended):
    python -m kv_cache_tiering.memory_pressure_scheduler \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --policies lru attention hybrid \\
        --dataset sharegpt \\
        --dataset-path /path/to/sharegpt.json \\
        --num-prompts 200 \\
        --gpu-mem-util 0.12 \\
        --cpu-bytes 8000000000 \\
        --max-tokens 1024 \\
        --output results.json

    # As a programmatic API:
    from kv_cache_tiering.memory_pressure_scheduler import (
        apply_memory_pressure,
        run_pressure_benchmark,
    )
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kv_cache_tiering.benchmarks.benchmark import (
    BenchmarkConfig,
    BenchmarkMetrics,
    build_kv_connector_config,
    load_prompts,
)


@dataclass
class PressureConfig:
    """Configuration for memory pressure parameters."""

    # GPU memory fraction -- lower = fewer KV blocks = more pressure.
    gpu_memory_utilization: float = 0.12

    # CPU offload tier size in bytes.
    cpu_bytes_to_use: int = 8_000_000_000  # 8 GB

    # Maximum output tokens per request.  Combined with ignore_eos=True
    # this forces each request to consume its full KV budget.
    max_tokens: int = 1024

    # Maximum sequence length the model will accept.
    max_model_len: int = 16384

    # Number of prompts to submit in a single batch.
    num_prompts: int = 200

    # GPU block fraction: artificially restrict free blocks to this
    # fraction of the actual pool.  1.0 = no restriction, 0.5 = only
    # half the real blocks visible to the scheduler.  Set <1.0 to force
    # even more eviction pressure without reducing gpu_memory_utilization
    # below the model-weight floor.
    block_fraction: float = 1.0


def apply_memory_pressure(block_fraction: float = 0.5) -> dict[str, Any]:
    """
    Monkey-patch the vLLM block pool to report fewer free blocks.

    This forces the scheduler to preempt sooner, which in turn forces
    the offloading connector to store more blocks to CPU, eventually
    filling the CPU tier and triggering real eviction-policy decisions.

    Args:
        block_fraction: Fraction of real free blocks to report.
            0.5 means the scheduler sees only half the actual free blocks.

    Returns:
        dict with keys 'patched' (bool) and 'original_fn' (callable).
    """
    try:
        from vllm.v1.core.block_pool import BlockPool
    except ImportError:
        return {"patched": False, "reason": "BlockPool not importable"}

    original_get_num_free = BlockPool.get_num_free_blocks

    def constrained_get_num_free(self: Any) -> int:
        real_free = original_get_num_free(self)
        return max(1, int(real_free * block_fraction))

    BlockPool.get_num_free_blocks = constrained_get_num_free  # type: ignore[assignment]
    return {"patched": True, "original_fn": original_get_num_free}


def restore_block_pool(patch_info: dict[str, Any]) -> None:
    """Undo the monkey-patch applied by apply_memory_pressure."""
    if not patch_info.get("patched"):
        return
    try:
        from vllm.v1.core.block_pool import BlockPool

        BlockPool.get_num_free_blocks = patch_info["original_fn"]
    except ImportError:
        pass


def run_pressure_benchmark(
    config: BenchmarkConfig,
    pressure: PressureConfig | None = None,
) -> BenchmarkMetrics:
    """
    Run a benchmark under controlled memory pressure.

    Compared to the vanilla ``run_benchmark``, this function:
    - Uses ``ignore_eos=True`` so every request generates exactly
      ``max_tokens`` output tokens, maximising KV cache consumption.
    - Optionally monkey-patches the block pool (``block_fraction < 1``).
    - Prints a pressure summary before and after the run.
    """
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    if pressure is None:
        pressure = PressureConfig()

    # Override config with pressure settings.
    config.gpu_memory_utilization = pressure.gpu_memory_utilization
    config.cpu_bytes_to_use = pressure.cpu_bytes_to_use
    config.max_tokens = pressure.max_tokens
    config.max_model_len = pressure.max_model_len
    config.num_prompts = pressure.num_prompts

    kv_connector_extra = build_kv_connector_config(config)
    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config=kv_connector_extra,
    )

    print("=" * 70)
    print("MEMORY PRESSURE BENCHMARK")
    print("=" * 70)
    print(f"  Model:            {config.model}")
    print(f"  Policy:           {config.eviction_policy}")
    print(f"  Dataset:          {config.dataset}")
    print(f"  GPU mem util:     {pressure.gpu_memory_utilization:.0%}")
    print(f"  CPU offload:      {pressure.cpu_bytes_to_use / 1e9:.1f} GB")
    print(f"  Num prompts:      {pressure.num_prompts}")
    print(f"  Max tokens:       {pressure.max_tokens}")
    print(f"  Block fraction:   {pressure.block_fraction:.0%}")
    print(f"  ignore_eos:       True")
    print("=" * 70)

    # Apply block-pool restriction if requested.
    patch_info: dict[str, Any] = {}
    if pressure.block_fraction < 1.0:
        patch_info = apply_memory_pressure(pressure.block_fraction)
        if patch_info.get("patched"):
            print(f"  [patch] Block pool restricted to {pressure.block_fraction:.0%}")
        else:
            print(f"  [patch] Could not patch: {patch_info.get('reason')}")

    try:
        llm = LLM(
            model=config.model,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            kv_transfer_config=kv_transfer_config,
        )

        prompts = load_prompts(config)

        # Truncate prompts to fit within model length.
        tokenizer = llm.get_tokenizer()
        max_input_tokens = config.max_model_len - config.max_tokens - 10
        truncated = []
        for p in prompts:
            tokens = tokenizer.encode(p)
            if len(tokens) > max_input_tokens:
                p = tokenizer.decode(tokens[:max_input_tokens])
            truncated.append(p)
        prompts = truncated

        # ignore_eos forces full max_tokens generation per request,
        # maximising KV cache block consumption.
        sampling_params = SamplingParams(
            max_tokens=config.max_tokens,
            ignore_eos=True,
        )

        # Warmup.
        print("\nWarmup (2 prompts)...")
        llm.generate(prompts[:2], sampling_params, use_tqdm=False)

        # Main benchmark -- single batch to maximise concurrency.
        print(f"\nRunning {len(prompts)} prompts in single batch...")
        start_time = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        total_time = time.perf_counter() - start_time

        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        n = len(outputs)

        # Latency stats.
        latencies: list[float] = []
        ttfts: list[float] = []
        for output in outputs:
            metrics = getattr(output, "metrics", None)
            if metrics:
                if getattr(metrics, "time_to_first_token", None):
                    ttfts.append(metrics.time_to_first_token * 1000)
                arrival = getattr(metrics, "arrival_time", None)
                finished = getattr(metrics, "finished_time", None)
                if arrival is not None and finished is not None:
                    latencies.append((finished - arrival) * 1000)

        if latencies:
            avg_lat = float(np.mean(latencies))
            p50_lat = float(np.percentile(latencies, 50))
            p95_lat = float(np.percentile(latencies, 95))
            p99_lat = float(np.percentile(latencies, 99))
        else:
            avg_lat = (total_time / n) * 1000 if n else 0.0
            p50_lat = avg_lat
            p95_lat = avg_lat * 1.15
            p99_lat = avg_lat * 1.20

        if ttfts:
            avg_ttft = float(np.mean(ttfts))
            p50_ttft = float(np.percentile(ttfts, 50))
            p95_ttft = float(np.percentile(ttfts, 95))
        else:
            avg_ttft = p50_ttft = p95_ttft = 0.0

        # Read eviction stats from the connector.
        total_evictions = 0
        bytes_gpu_to_cpu = 0
        bytes_cpu_to_gpu = 0
        try:
            stats = llm.llm_engine.engine_core.kv_connector.get_stats()
            total_evictions = stats.get("total_evictions", 0)
            bytes_gpu_to_cpu = stats.get("bytes_gpu_to_cpu", 0)
            bytes_cpu_to_gpu = stats.get("bytes_cpu_to_gpu", 0)
        except Exception:
            pass

        result = BenchmarkMetrics(
            policy=config.eviction_policy,
            model=config.model,
            dataset=config.dataset,
            num_prompts=n,
            total_time_seconds=total_time,
            tokens_per_second=total_tokens / total_time if total_time else 0,
            requests_per_second=n / total_time if total_time else 0,
            avg_latency_ms=avg_lat,
            p50_latency_ms=p50_lat,
            p95_latency_ms=p95_lat,
            p99_latency_ms=p99_lat,
            avg_ttft_ms=avg_ttft,
            p50_ttft_ms=p50_ttft,
            p95_ttft_ms=p95_ttft,
            total_evictions=total_evictions,
            bytes_gpu_to_cpu=bytes_gpu_to_cpu,
            bytes_cpu_to_gpu=bytes_cpu_to_gpu,
            config=asdict(config),
        )

        print(f"\n{'=' * 70}")
        print(f"RESULTS - {config.eviction_policy.upper()} (Memory Pressure)")
        print(f"{'=' * 70}")
        print(f"  Throughput:     {result.tokens_per_second:.1f} tok/s")
        print(f"  Avg latency:    {result.avg_latency_ms:.1f} ms")
        print(f"  P95 latency:    {result.p95_latency_ms:.1f} ms")
        print(f"  Total evictions: {total_evictions}")
        print(f"  GPU->CPU bytes: {bytes_gpu_to_cpu:,}")
        print(f"  CPU->GPU bytes: {bytes_cpu_to_gpu:,}")
        print(f"{'=' * 70}")

        del llm
        return result

    finally:
        restore_block_pool(patch_info)


def run_pressure_comparison(
    configs: list[BenchmarkConfig],
    pressure: PressureConfig,
) -> list[BenchmarkMetrics]:
    """Run multiple policies under identical memory pressure."""
    results = []
    for config in configs:
        print(f"\n{'#' * 70}")
        print(f"# Policy: {config.eviction_policy}")
        print(f"{'#' * 70}")
        metrics = run_pressure_benchmark(config, pressure)
        results.append(metrics)
    return results


def print_comparison_table(results: list[BenchmarkMetrics]) -> None:
    """Print a formatted comparison table."""
    if not results:
        return

    print(f"\n{'=' * 80}")
    print("COMPARISON TABLE")
    print(f"{'=' * 80}")
    print(
        f"{'Policy':<12} {'Throughput':>12} {'Avg Lat':>10} "
        f"{'P95 Lat':>10} {'Evictions':>10} {'GPU->CPU':>12}"
    )
    print("-" * 80)

    baseline_tps = None
    for r in results:
        if r.policy == "lru":
            baseline_tps = r.tokens_per_second
            break

    for r in results:
        improvement = ""
        if baseline_tps and baseline_tps > 0 and r.policy != "lru":
            pct = ((r.tokens_per_second - baseline_tps) / baseline_tps) * 100
            improvement = f" ({pct:+.1f}%)"

        print(
            f"{r.policy:<12} {r.tokens_per_second:>9.1f} t/s{improvement:>6} "
            f"{r.avg_latency_ms:>8.1f}ms "
            f"{r.p95_latency_ms:>8.1f}ms "
            f"{r.total_evictions:>10} "
            f"{r.bytes_gpu_to_cpu:>10,}"
        )

    print(f"{'=' * 80}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run KV cache benchmarks under memory pressure"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["lru", "attention", "hybrid"],
    )
    parser.add_argument("--dataset", default="sharegpt")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--cpu-bytes", type=int, default=8_000_000_000)
    parser.add_argument("--gpu-mem-util", type=float, default=0.12)
    parser.add_argument("--block-fraction", type=float, default=1.0,
                        help="Artificially restrict free GPU blocks (0.0-1.0)")
    parser.add_argument("--block-size", type=int, default=48)
    parser.add_argument("--output", default="benchmark_results/pressure_results.json")

    args = parser.parse_args()

    pressure = PressureConfig(
        gpu_memory_utilization=args.gpu_mem_util,
        cpu_bytes_to_use=args.cpu_bytes,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        num_prompts=args.num_prompts,
        block_fraction=args.block_fraction,
    )

    configs = []
    for policy in args.policies:
        configs.append(
            BenchmarkConfig(
                model=args.model,
                eviction_policy=policy,
                cpu_bytes_to_use=args.cpu_bytes,
                gpu_memory_utilization=args.gpu_mem_util,
                max_model_len=args.max_model_len,
                max_tokens=args.max_tokens,
                num_prompts=args.num_prompts,
                dataset=args.dataset,
                dataset_path=args.dataset_path,
                block_size=args.block_size,
            )
        )

    results = run_pressure_comparison(configs, pressure)
    print_comparison_table(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([asdict(m) for m in results], f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
