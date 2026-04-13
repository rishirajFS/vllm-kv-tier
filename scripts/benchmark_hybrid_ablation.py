#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Hybrid Policy Ablation - Priority 7 from final_plan.md

Sweeps the attention weight (alpha) from 0.0 to 1.0 in the hybrid policy
to find the optimal balance for each workload type.

Hybrid score = alpha * attention + beta * recency + gamma * frequency
where beta + gamma = (1 - alpha), with beta/(beta+gamma) fixed at 0.6.

Expected optimal alpha per workload:
- ShareGPT (conversational):  alpha ~0.7 (attention-heavy)
- MS-MARCO (RAG/retrieval):   alpha ~0.6 (moderate attention)
- HumanEval (code completion): alpha ~0.3-0.4 (recency matters more)

Usage:
    python scripts/benchmark_hybrid_ablation.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dataset-path ~/workspace/vllm/datasets/sharegpt.json \
        --output hybrid_ablation_sharegpt.json

    # Quick sweep with fewer alpha values
    python scripts/benchmark_hybrid_ablation.py \
        --alpha-values 0.0 0.3 0.5 0.7 1.0 \
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
class AblationResult:
    """Results for a single alpha value."""
    alpha: float
    beta: float
    gamma: float
    model: str
    dataset: str
    num_prompts: int
    tokens_per_second: float
    avg_latency_ms: float
    p95_latency_ms: float
    avg_ttft_ms: float
    total_output_tokens: int
    total_time_seconds: float
    eviction_count: int
    evictions_per_1k_tokens: float
    improvement_over_pure_lru: float = 0.0


@dataclass
class AblationConfig:
    """Configuration for hybrid ablation sweep."""
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    alpha_values: list[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    # Ratio of beta to (beta + gamma) -- controls recency vs frequency balance
    beta_gamma_ratio: float = 0.6
    num_prompts: int = 200
    max_tokens: int = 1024
    max_model_len: int = 8192
    gpu_memory_util: float = 0.12
    cpu_bytes: int = 8_000_000_000
    dataset_path: str | None = None
    dataset_name: str = "sharegpt"


def compute_weights(alpha: float, beta_gamma_ratio: float) -> tuple[float, float, float]:
    """
    Compute (alpha, beta, gamma) from alpha and the beta/gamma ratio.

    Args:
        alpha: Attention weight [0, 1]
        beta_gamma_ratio: beta / (beta + gamma)

    Returns:
        (alpha, beta, gamma) with alpha + beta + gamma = 1.0
    """
    remaining = 1.0 - alpha
    beta = remaining * beta_gamma_ratio
    gamma = remaining * (1.0 - beta_gamma_ratio)
    return alpha, beta, gamma


def load_prompts(config: AblationConfig) -> list[str]:
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

    # Synthetic prompts
    topics = [
        "Explain machine learning algorithms and their applications",
        "Describe distributed systems architecture patterns",
        "Write about modern database design principles",
        "Discuss the evolution of programming paradigms",
        "Explain cryptographic protocols and their security properties",
    ]
    prompts = []
    for i in range(config.num_prompts):
        topic = topics[i % len(topics)]
        context = f"Context {i}: " + "background info. " * (i % 15)
        prompts.append(f"{context}\n\n{topic}. Provide a detailed response.")
    return prompts


def run_ablation_point(
    config: AblationConfig,
    alpha: float,
    prompts: list[str],
) -> AblationResult:
    """Run benchmark for a single alpha value."""
    import torch
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    alpha_val, beta_val, gamma_val = compute_weights(alpha, config.beta_gamma_ratio)

    print(f"\n{'='*70}")
    print(f"Alpha={alpha_val:.2f}  Beta={beta_val:.2f}  Gamma={gamma_val:.2f}")
    print(f"{'='*70}")

    # For alpha=0.0, this is effectively recency+frequency (no attention)
    # For alpha=1.0, this is pure attention (like attention manager)
    extra_config = {
        "cpu_bytes_to_use": config.cpu_bytes,
        "block_size": 48,
        "eviction_policy": "hybrid",
        "attention_weight": alpha_val,
        "recency_weight": beta_val,
        "frequency_weight": gamma_val,
        "score_decay": 0.95,
    }

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

    result = AblationResult(
        alpha=alpha_val,
        beta=beta_val,
        gamma=gamma_val,
        model=config.model,
        dataset=config.dataset_name,
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

    print(f"  Throughput: {throughput:.1f} tok/s | Latency: {avg_lat:.1f} ms")
    return result


def run_pure_lru_baseline(config: AblationConfig, prompts: list[str]) -> float:
    """Run pure LRU baseline and return throughput."""
    import torch
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    print(f"\n{'='*70}")
    print(f"BASELINE: Pure LRU")
    print(f"{'='*70}")

    kv_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "cpu_bytes_to_use": config.cpu_bytes,
            "block_size": 48,
            "eviction_policy": "lru",
        },
    )

    llm = LLM(
        model=config.model,
        gpu_memory_utilization=config.gpu_memory_util,
        max_model_len=config.max_model_len,
        kv_transfer_config=kv_config,
    )

    sampling_params = SamplingParams(max_tokens=config.max_tokens, temperature=0.0)
    llm.generate([prompts[0]], sampling_params, use_tqdm=False)

    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    total_time = time.perf_counter() - start_time

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / total_time if total_time > 0 else 0

    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"  LRU Baseline Throughput: {throughput:.1f} tok/s")
    return throughput


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Policy Ablation: sweep attention weight (alpha) from 0.0 to 1.0"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default="sharegpt",
                        choices=["sharegpt", "msmarco", "humaneval"])
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--gpu-mem-util", type=float, default=0.12)
    parser.add_argument("--cpu-bytes", type=int, default=8_000_000_000)
    parser.add_argument("--alpha-values", type=float, nargs="+",
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument("--beta-gamma-ratio", type=float, default=0.6,
                        help="beta / (beta + gamma) ratio (default: 0.6)")
    parser.add_argument("--output", type=Path,
                        default=Path("hybrid_ablation_results.json"))

    args = parser.parse_args()

    config = AblationConfig(
        model=args.model,
        alpha_values=args.alpha_values,
        beta_gamma_ratio=args.beta_gamma_ratio,
        num_prompts=args.num_prompts,
        max_tokens=args.max_tokens,
        gpu_memory_util=args.gpu_mem_util,
        cpu_bytes=args.cpu_bytes,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
    )

    print(f"\n{'#'*70}")
    print(f"# Hybrid Policy Ablation Study")
    print(f"# Model: {config.model}")
    print(f"# Dataset: {config.dataset_name}")
    print(f"# Alpha values: {config.alpha_values}")
    print(f"# Beta/Gamma ratio: {config.beta_gamma_ratio}")
    print(f"# GPU Memory: {config.gpu_memory_util*100:.0f}%")
    print(f"{'#'*70}\n")

    prompts = load_prompts(config)
    print(f"Loaded {len(prompts)} prompts")

    # Run LRU baseline first
    lru_throughput = run_pure_lru_baseline(config, prompts)

    # Sweep alpha values
    all_results: list[AblationResult] = []

    for alpha in config.alpha_values:
        try:
            result = run_ablation_point(config, alpha, prompts)
            if lru_throughput > 0:
                result.improvement_over_pure_lru = (
                    (result.tokens_per_second - lru_throughput) / lru_throughput * 100
                )
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR at alpha={alpha}: {e}")
            continue

    # Save results
    output_data = {
        "config": {
            "model": config.model,
            "dataset": config.dataset_name,
            "gpu_memory_util": config.gpu_memory_util,
            "beta_gamma_ratio": config.beta_gamma_ratio,
            "lru_baseline_throughput": lru_throughput,
        },
        "results": [asdict(r) for r in all_results],
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print(f"HYBRID ABLATION RESULTS ({config.dataset_name})")
    print(f"{'='*80}")
    print(f"LRU Baseline: {lru_throughput:.1f} tok/s")
    print()
    print(f"{'Alpha':>6} | {'Beta':>6} | {'Gamma':>6} | {'Throughput':>10} | {'vs LRU':>8} | {'Evict/1K':>8}")
    print(f"{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")

    best_result = None
    best_throughput = 0

    for r in all_results:
        tag = f"{r.improvement_over_pure_lru:+.1f}%"
        print(f"{r.alpha:5.2f} | {r.beta:5.2f} | {r.gamma:5.2f} | "
              f"{r.tokens_per_second:8.1f} | {tag:>8} | {r.evictions_per_1k_tokens:6.1f}")
        if r.tokens_per_second > best_throughput:
            best_throughput = r.tokens_per_second
            best_result = r

    if best_result:
        print(f"\nOptimal alpha for {config.dataset_name}: {best_result.alpha:.2f}")
        print(f"  Weights: alpha={best_result.alpha:.2f}, beta={best_result.beta:.2f}, gamma={best_result.gamma:.2f}")
        print(f"  Throughput: {best_result.tokens_per_second:.1f} tok/s ({best_result.improvement_over_pure_lru:+.1f}% vs LRU)")

    print(f"\nResults saved to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
