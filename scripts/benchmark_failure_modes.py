#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Failure Mode Analysis - Priority 8 from final_plan.md

Tests adversarial and edge-case scenarios to identify when attention-aware
eviction does NOT help or actively hurts performance.

Test scenarios:
1. Adversarial access patterns (random, cyclic, zipf)
2. Very short sequences (<512 tokens) -- no eviction needed
3. Very small models (0.5B-1.5B) -- overhead may dominate
4. Single long request vs many short requests
5. Uniform access patterns (all blocks equally important)

Expected results:
- Short sequences: no improvement (no evictions triggered)
- Random access: attention scores less useful, ~LRU performance
- Very small models: overhead of score computation may hurt
- Uniform access: all policies perform similarly

Usage:
    python scripts/benchmark_failure_modes.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --output failure_modes.json

    # Run specific scenarios
    python scripts/benchmark_failure_modes.py \
        --scenarios short_sequences random_access uniform_access
"""
import argparse
import gc
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class FailureModeResult:
    """Results for a single failure mode scenario."""
    scenario: str
    scenario_description: str
    model: str
    policy: str
    num_prompts: int
    tokens_per_second: float
    avg_latency_ms: float
    p95_latency_ms: float
    total_output_tokens: int
    total_time_seconds: float
    eviction_count: int
    evictions_per_1k_tokens: float
    improvement_over_lru: float = 0.0
    verdict: str = ""  # "helps", "neutral", "hurts"


SCENARIOS = {
    "short_sequences": {
        "description": "Very short sequences (<512 tokens) where eviction is unlikely",
        "max_tokens": 64,
        "num_prompts": 200,
        "prompt_style": "short",
    },
    "random_access": {
        "description": "Random, non-repeating prompts with no prefix sharing",
        "max_tokens": 512,
        "num_prompts": 200,
        "prompt_style": "random",
    },
    "uniform_access": {
        "description": "Uniform prompts where all blocks have equal importance",
        "max_tokens": 512,
        "num_prompts": 200,
        "prompt_style": "uniform",
    },
    "single_long_request": {
        "description": "One very long request instead of many short ones",
        "max_tokens": 2048,
        "num_prompts": 5,
        "prompt_style": "long",
    },
    "high_concurrency_short": {
        "description": "Many concurrent short requests (high throughput, low eviction)",
        "max_tokens": 128,
        "num_prompts": 500,
        "prompt_style": "short_varied",
    },
    "adversarial_cyclic": {
        "description": "Cyclic access pattern designed to defeat LRU (also tests attention)",
        "max_tokens": 512,
        "num_prompts": 200,
        "prompt_style": "cyclic",
    },
}


def generate_prompts(scenario_name: str, scenario_config: dict, count: int) -> list[str]:
    """Generate prompts for the given scenario."""
    style = scenario_config["prompt_style"]
    prompts = []

    if style == "short":
        # Very short prompts
        short_questions = [
            "What is 2+2?",
            "Name a color.",
            "Say hello.",
            "What day is today?",
            "Count to five.",
            "Name a fruit.",
            "What is Python?",
            "Define AI.",
            "Say goodbye.",
            "Name a planet.",
        ]
        for i in range(count):
            prompts.append(short_questions[i % len(short_questions)])

    elif style == "random":
        # Random unique prompts with no shared prefixes
        for i in range(count):
            random_words = " ".join(
                random.choices(
                    ["quantum", "neural", "distributed", "compiler", "kernel",
                     "memory", "cache", "pipeline", "tensor", "gradient",
                     "inference", "training", "embedding", "attention", "layer"],
                    k=random.randint(5, 20)
                )
            )
            prompts.append(f"Topic {i}: {random_words}. Explain this concept.")

    elif style == "uniform":
        # Same prompt repeated -- all blocks should have equal attention
        base_prompt = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a test of uniform access patterns in KV cache eviction. "
            "All tokens in this prompt should receive approximately equal "
            "attention weights, making it difficult for attention-based "
            "eviction to outperform LRU."
        )
        for _ in range(count):
            prompts.append(base_prompt + " Summarize the above text.")

    elif style == "long":
        # Very long prompts
        base = (
            "This is a comprehensive document about the history, theory, and "
            "applications of computer science. " * 200
        )
        for i in range(count):
            prompts.append(f"{base}\n\nSection {i}: Provide a detailed analysis.")

    elif style == "short_varied":
        # Many short but varied prompts
        templates = [
            "Define {topic}.",
            "What is {topic}?",
            "Explain {topic} briefly.",
            "List 3 facts about {topic}.",
            "Why is {topic} important?",
        ]
        topics = [
            "machine learning", "databases", "networking", "security",
            "algorithms", "compilers", "operating systems", "graphics",
            "robotics", "cryptography", "optimization", "parallelism",
        ]
        for i in range(count):
            template = templates[i % len(templates)]
            topic = topics[i % len(topics)]
            prompts.append(template.format(topic=topic))

    elif style == "cyclic":
        # Cyclic pattern: A, B, C, D, A, B, C, D, ...
        # This defeats LRU when cache < 4 items but attention may not help either
        cycle_prompts = [
            "Document A: The theory of relativity explains how gravity affects space-time. "
            "Explain Einstein's key insights about the nature of gravity and acceleration.",
            "Document B: Quantum mechanics describes behavior at the subatomic level. "
            "Explain the uncertainty principle and wave-particle duality.",
            "Document C: The standard model of particle physics categorizes all particles. "
            "Explain the role of quarks, leptons, and bosons in the standard model.",
            "Document D: String theory proposes that fundamental particles are vibrating strings. "
            "Explain how string theory attempts to unify quantum mechanics and gravity.",
        ]
        for i in range(count):
            prompts.append(cycle_prompts[i % len(cycle_prompts)])

    return prompts


def run_scenario_point(
    model: str,
    scenario_name: str,
    scenario_config: dict,
    policy: str,
    prompts: list[str],
    gpu_memory_util: float,
    cpu_bytes: int,
    max_model_len: int,
) -> FailureModeResult:
    """Run a single scenario + policy combination."""
    import torch
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    max_tokens = scenario_config["max_tokens"]

    print(f"  {policy:10s}: ", end="", flush=True)

    extra_config = {
        "cpu_bytes_to_use": cpu_bytes,
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
        model=model,
        gpu_memory_utilization=gpu_memory_util,
        max_model_len=max_model_len,
        kv_transfer_config=kv_config,
    )

    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    llm.generate([prompts[0]], sampling_params, use_tqdm=False)

    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    total_time = time.perf_counter() - start_time

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    latencies = []
    for output in outputs:
        metrics = getattr(output, "metrics", None)
        if metrics:
            finished = getattr(metrics, "finished_time", None)
            arrival = getattr(metrics, "arrival_time", None)
            if finished is not None and arrival is not None:
                latencies.append((finished - arrival) * 1000)

    throughput = total_tokens / total_time if total_time > 0 else 0
    avg_lat = float(np.mean(latencies)) if latencies else (total_time / len(outputs)) * 1000
    p95_lat = float(np.percentile(latencies, 95)) if latencies else avg_lat * 1.15

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

    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"{throughput:6.1f} tok/s, {avg_lat:6.1f}ms, {eviction_count} evictions")

    return FailureModeResult(
        scenario=scenario_name,
        scenario_description=scenario_config["description"],
        model=model,
        policy=policy,
        num_prompts=len(outputs),
        tokens_per_second=throughput,
        avg_latency_ms=avg_lat,
        p95_latency_ms=p95_lat,
        total_output_tokens=total_tokens,
        total_time_seconds=total_time,
        eviction_count=eviction_count,
        evictions_per_1k_tokens=evictions_per_1k,
    )


def classify_verdict(improvement: float) -> str:
    """Classify result as helps, neutral, or hurts."""
    if improvement > 3.0:
        return "helps"
    elif improvement < -3.0:
        return "hurts"
    else:
        return "neutral"


def main():
    parser = argparse.ArgumentParser(
        description="Failure Mode Analysis: test when attention-aware eviction does NOT help"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--gpu-mem-util", type=float, default=0.12)
    parser.add_argument("--cpu-bytes", type=int, default=8_000_000_000)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--scenarios", type=str, nargs="+",
                        default=list(SCENARIOS.keys()),
                        choices=list(SCENARIOS.keys()),
                        help="Scenarios to test")
    parser.add_argument("--policies", type=str, nargs="+",
                        default=["lru", "attention", "hybrid"])
    parser.add_argument("--output", type=Path,
                        default=Path("failure_modes.json"))

    args = parser.parse_args()

    print(f"\n{'#'*70}")
    print(f"# Failure Mode Analysis")
    print(f"# Model: {args.model}")
    print(f"# Scenarios: {args.scenarios}")
    print(f"# GPU Memory: {args.gpu_mem_util*100:.0f}%")
    print(f"{'#'*70}\n")

    all_results: list[FailureModeResult] = []

    for scenario_name in args.scenarios:
        scenario_config = SCENARIOS[scenario_name]
        count = scenario_config["num_prompts"]

        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario_name}")
        print(f"  {scenario_config['description']}")
        print(f"  Prompts: {count}, Max tokens: {scenario_config['max_tokens']}")
        print(f"{'='*70}")

        prompts = generate_prompts(scenario_name, scenario_config, count)
        scenario_results: list[FailureModeResult] = []

        for policy in args.policies:
            try:
                result = run_scenario_point(
                    model=args.model,
                    scenario_name=scenario_name,
                    scenario_config=scenario_config,
                    policy=policy,
                    prompts=prompts,
                    gpu_memory_util=args.gpu_mem_util,
                    cpu_bytes=args.cpu_bytes,
                    max_model_len=args.max_model_len,
                )
                scenario_results.append(result)
                all_results.append(result)
            except Exception as e:
                print(f"  ERROR: {policy}: {e}")
                continue

        # Calculate improvements and verdicts
        lru_result = next((r for r in scenario_results if r.policy == "lru"), None)
        if lru_result and lru_result.tokens_per_second > 0:
            for r in scenario_results:
                if r.policy != "lru":
                    r.improvement_over_lru = (
                        (r.tokens_per_second - lru_result.tokens_per_second)
                        / lru_result.tokens_per_second * 100
                    )
                    r.verdict = classify_verdict(r.improvement_over_lru)
                else:
                    r.verdict = "baseline"

    # Save results
    with open(args.output, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print("FAILURE MODE ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Scenario':>25} | {'Policy':>10} | {'Throughput':>10} | {'vs LRU':>8} | {'Verdict':>8}")
    print(f"{'-'*25}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")

    for scenario_name in args.scenarios:
        for r in all_results:
            if r.scenario == scenario_name:
                tag = f"{r.improvement_over_lru:+.1f}%" if r.policy != "lru" else "—"
                print(f"{scenario_name:>25} | {r.policy:>10} | "
                      f"{r.tokens_per_second:8.1f} | {tag:>8} | {r.verdict:>8}")
        print(f"{'-'*25}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")

    # Print insights
    print(f"\nKey Findings:")
    for scenario_name in args.scenarios:
        attn = next((r for r in all_results
                     if r.scenario == scenario_name and r.policy == "attention"), None)
        if attn:
            if attn.verdict == "hurts":
                print(f"  [HURTS]   {scenario_name}: attention is {attn.improvement_over_lru:+.1f}% vs LRU")
            elif attn.verdict == "neutral":
                print(f"  [NEUTRAL] {scenario_name}: attention is {attn.improvement_over_lru:+.1f}% vs LRU")
            else:
                print(f"  [HELPS]   {scenario_name}: attention is {attn.improvement_over_lru:+.1f}% vs LRU")

    print(f"\nResults saved to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
