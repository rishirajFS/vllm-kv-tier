#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Memory Efficiency Benchmark - Test maximum sequence length capacity.

This script measures:
1. Max sequence length before OOM (GPU-only vs Tiered)
2. Memory amplification factor
3. Effective memory utilization

Usage:
    python scripts/benchmark_memory_efficiency.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --output memory_efficiency_qwen3b.json
"""
import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass
class MemoryEfficiencyResult:
    """Results from memory efficiency benchmark."""
    model: str
    gpu_memory_util: float
    cpu_tier_enabled: bool
    cpu_tier_size_gb: float
    max_sequence_length: int
    max_batch_size: int
    time_to_oom_seconds: float
    peak_gpu_memory_mb: float
    memory_amplification: float = 1.0
    notes: str = ""


def test_max_sequence_length(
    model_name: str,
    gpu_memory_util: float,
    cpu_tier_enabled: bool,
    cpu_tier_size: int,
    start_length: int = 2048,
    step_size: int = 2048,
    max_attempts: int = 10,
) -> MemoryEfficiencyResult:
    """
    Binary search to find maximum sequence length before OOM.

    Args:
        model_name: Hugging Face model path
        gpu_memory_util: GPU memory utilization fraction
        cpu_tier_enabled: Whether to enable CPU tiering
        cpu_tier_size: CPU tier size in bytes (if enabled)
        start_length: Starting sequence length
        step_size: Increment step for sequence length
        max_attempts: Maximum number of attempts before giving up

    Returns:
        MemoryEfficiencyResult with max sequence length and metrics
    """
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"GPU Memory: {gpu_memory_util * 100}%")
    print(f"CPU Tier: {'Enabled' if cpu_tier_enabled else 'Disabled'}")
    if cpu_tier_enabled:
        print(f"CPU Tier Size: {cpu_tier_size / (1024**3):.1f} GB")
    print(f"{'='*60}\n")

    # Configuration
    kv_config = None
    if cpu_tier_enabled:
        kv_config = KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "cpu_bytes_to_use": cpu_tier_size,
                "block_size": 48,
                "eviction_policy": "attention",
                "score_decay": 0.95,
            },
        )

    max_working_length = 0
    failed_length = None
    peak_memory = 0.0
    start_time = time.time()

    current_length = start_length

    for attempt in range(max_attempts):
        try:
            print(f"\nAttempt {attempt + 1}/{max_attempts}: Testing {current_length} tokens...")

            # Initialize LLM
            llm = LLM(
                model=model_name,
                gpu_memory_utilization=gpu_memory_util,
                max_model_len=current_length,
                kv_transfer_config=kv_config,
                enforce_eager=True,  # Avoid CUDA graph overhead for testing
            )

            # Generate a simple prompt that will use the full context
            prompt = "Hello " * (current_length // 2)  # Fill half the context
            sampling_params = SamplingParams(
                max_tokens=min(512, current_length // 4),  # Generate some tokens
                temperature=0.0,  # Deterministic for testing
            )

            # Run generation
            outputs = llm.generate([prompt], sampling_params, use_tqdm=False)

            # Check GPU memory
            if torch.cuda.is_available():
                current_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
                peak_memory = max(peak_memory, current_memory)
                print(f"  ✓ Success! Peak GPU memory: {current_memory:.1f} MB")

            # Clean up
            del llm
            torch.cuda.empty_cache()

            # Success - try longer sequence
            max_working_length = current_length
            current_length += step_size

        except torch.cuda.OutOfMemoryError as e:
            print(f"  ✗ OOM at {current_length} tokens")
            failed_length = current_length

            # Clean up
            torch.cuda.empty_cache()

            # If we found a working length, we're done
            if max_working_length > 0:
                break

            # Otherwise, try shorter sequence
            current_length = current_length // 2

        except Exception as e:
            print(f"  ✗ Error: {e}")
            torch.cuda.empty_cache()
            break

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Maximum working sequence length: {max_working_length} tokens")
    print(f"Failed at: {failed_length} tokens" if failed_length else "Did not hit limit")
    print(f"Peak GPU memory: {peak_memory:.1f} MB")
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"{'='*60}\n")

    return MemoryEfficiencyResult(
        model=model_name,
        gpu_memory_util=gpu_memory_util,
        cpu_tier_enabled=cpu_tier_enabled,
        cpu_tier_size_gb=cpu_tier_size / (1024 ** 3),
        max_sequence_length=max_working_length,
        max_batch_size=1,  # Single prompt for max length test
        time_to_oom_seconds=elapsed,
        peak_gpu_memory_mb=peak_memory,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark memory efficiency: max sequence length and amplification"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Model to benchmark",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("memory_efficiency_results.json"),
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--gpu-memory-baseline",
        type=float,
        default=0.9,
        help="GPU memory utilization for baseline (no tiering)",
    )
    parser.add_argument(
        "--gpu-memory-tiered",
        type=float,
        default=0.12,
        help="GPU memory utilization for tiered system",
    )
    parser.add_argument(
        "--cpu-tier-size",
        type=int,
        default=8_000_000_000,
        help="CPU tier size in bytes (default: 8GB)",
    )
    parser.add_argument(
        "--start-length",
        type=int,
        default=2048,
        help="Starting sequence length for binary search",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=2048,
        help="Step size for sequence length increments",
    )

    args = parser.parse_args()

    results = []

    # Test 1: Baseline (GPU-only, high utilization)
    print("\n" + "="*70)
    print("TEST 1: BASELINE (GPU-only, no tiering)")
    print("="*70)

    baseline_result = test_max_sequence_length(
        model_name=args.model,
        gpu_memory_util=args.gpu_memory_baseline,
        cpu_tier_enabled=False,
        cpu_tier_size=0,
        start_length=args.start_length,
        step_size=args.step_size,
    )
    baseline_result.notes = "Baseline: GPU-only, no CPU tiering"
    results.append(baseline_result)

    # Test 2: Tiered system (low GPU + CPU tier)
    print("\n" + "="*70)
    print("TEST 2: TIERED SYSTEM (Low GPU + CPU tier)")
    print("="*70)

    tiered_result = test_max_sequence_length(
        model_name=args.model,
        gpu_memory_util=args.gpu_memory_tiered,
        cpu_tier_enabled=True,
        cpu_tier_size=args.cpu_tier_size,
        start_length=args.start_length,
        step_size=args.step_size,
    )
    tiered_result.notes = "Tiered: 12% GPU + CPU tier with attention-aware eviction"

    # Calculate memory amplification
    if baseline_result.max_sequence_length > 0:
        amplification = tiered_result.max_sequence_length / baseline_result.max_sequence_length
        tiered_result.memory_amplification = amplification
    else:
        tiered_result.memory_amplification = 0.0

    results.append(tiered_result)

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nModel: {args.model}")
    print(f"\nBaseline (GPU-only, {args.gpu_memory_baseline * 100}% VRAM):")
    print(f"  Max Sequence Length: {baseline_result.max_sequence_length:,} tokens")
    print(f"  Peak GPU Memory: {baseline_result.peak_gpu_memory_mb:.1f} MB")

    print(f"\nTiered System ({args.gpu_memory_tiered * 100}% GPU + {args.cpu_tier_size / 1e9:.0f}GB CPU):")
    print(f"  Max Sequence Length: {tiered_result.max_sequence_length:,} tokens")
    print(f"  Peak GPU Memory: {tiered_result.peak_gpu_memory_mb:.1f} MB")
    print(f"  Memory Amplification: {tiered_result.memory_amplification:.2f}×")

    if tiered_result.memory_amplification > 1.0:
        print(f"\n✓ Tiered system enables {tiered_result.memory_amplification:.1f}× longer sequences!")
    else:
        print(f"\n⚠ Warning: Tiered system did not show improvement")

    print(f"\nResults saved to: {args.output}")
    print("="*70 + "\n")

    # Save results to JSON
    with open(args.output, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
