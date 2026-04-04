#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Eviction Data Collection - Priority 4 from final_plan.md

Collects detailed eviction decisions and attention scores for visualization.
Runs inference on representative samples and logs:
- Which blocks were evicted (LRU vs Attention policies)
- Attention scores for each block
- Access patterns over time

This data enables three key visualizations:
1. Attention heatmaps showing LRU vs Attention eviction differences
2. Score distribution histograms (kept vs evicted blocks)
3. Access pattern comparison across workloads

Usage:
    python scripts/collect_eviction_data.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --dataset sharegpt \
        --num-samples 10 \
        --output eviction_data_sharegpt.json
"""
import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class BlockEvictionRecord:
    """Record of a single block's lifecycle."""
    block_id: int
    seq_id: int
    position: int  # Token position in sequence
    attention_score: float
    was_evicted: bool
    eviction_time: float  # Time when evicted (if applicable)
    policy: str  # "lru", "attention", "hybrid"
    num_accesses: int
    last_access_time: float


@dataclass
class RequestEvictionData:
    """Eviction data for a single request."""
    request_id: str
    workload: str
    prompt_length: int
    output_length: int
    total_blocks: int
    evicted_blocks: list[int]  # Block IDs that were evicted
    kept_blocks: list[int]  # Block IDs that stayed on GPU
    block_scores: dict[int, float]  # block_id -> attention score
    block_access_counts: dict[int, int]  # block_id -> access count
    eviction_timeline: list[dict[str, Any]]  # Time-ordered eviction events


@dataclass
class EvictionDataset:
    """Complete eviction dataset for visualization."""
    model: str
    workload: str
    policy: str
    num_samples: int
    requests: list[RequestEvictionData]
    summary_stats: dict[str, float]


def collect_eviction_data(
    model: str,
    prompts: list[str],
    policy: str,
    max_tokens: int = 256,
    gpu_mem_util: float = 0.12,
) -> list[RequestEvictionData]:
    """
    Run inference with instrumentation to collect eviction data.

    NOTE: This is a TEMPLATE implementation. Actual instrumentation requires
    modifying vLLM internals (block manager) to expose eviction decisions.

    Args:
        model: Model name
        prompts: List of prompts to run
        policy: Eviction policy
        max_tokens: Max tokens per request
        gpu_mem_util: GPU memory utilization

    Returns:
        List of RequestEvictionData for each prompt
    """
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    print(f"\nCollecting eviction data for {policy} policy...")

    kv_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "cpu_bytes_to_use": 8_000_000_000,
            "block_size": 48,
            "eviction_policy": policy,
            "score_decay": 0.95 if policy == "attention" else None,
            "attention_weight": 0.5 if policy == "hybrid" else None,
            "recency_weight": 0.3 if policy == "hybrid" else None,
            "frequency_weight": 0.2 if policy == "hybrid" else None,
            # INSTRUMENTATION: Enable logging
            "log_evictions": True,
            "log_attention_scores": True,
        },
    )

    llm = LLM(
        model=model,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=8192,
        kv_transfer_config=kv_config,
    )

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        seed=42,
    )

    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    # TEMPLATE: Extract eviction data from outputs
    # In real implementation, this would access internal block manager logs
    eviction_data = []

    for i, output in enumerate(outputs):
        # PLACEHOLDER: This data would come from instrumented block manager
        # For now, we create synthetic placeholder structure
        request_data = RequestEvictionData(
            request_id=f"req_{i}",
            workload="unknown",  # Set by caller
            prompt_length=len(output.prompt_token_ids),
            output_length=len(output.outputs[0].token_ids),
            total_blocks=0,  # Would be extracted from block manager
            evicted_blocks=[],  # Would be extracted from eviction log
            kept_blocks=[],  # Would be extracted from block manager
            block_scores={},  # Would be extracted from attention scores
            block_access_counts={},  # Would be extracted from access log
            eviction_timeline=[],  # Would be extracted from eviction events
        )

        eviction_data.append(request_data)

    del llm
    return eviction_data


def generate_synthetic_eviction_data(
    num_samples: int,
    workload: str,
    policy: str,
) -> list[RequestEvictionData]:
    """
    Generate synthetic eviction data for testing visualization.

    This simulates what real instrumentation would produce.
    Replace this with actual data collection once block manager is instrumented.

    Args:
        num_samples: Number of samples to generate
        workload: Workload type (sharegpt, msmarco, humaneval)
        policy: Eviction policy

    Returns:
        Synthetic eviction data
    """
    np.random.seed(42)
    data = []

    for i in range(num_samples):
        # Simulate a request with 50-200 blocks
        num_blocks = np.random.randint(50, 200)
        blocks = list(range(num_blocks))

        # Generate attention scores based on workload pattern
        if workload == "sharegpt":
            # Conversational: higher scores at beginning (system prompt) and end (recent)
            scores = np.exp(-0.05 * np.arange(num_blocks))
            scores[-20:] *= 2.0  # Recent context boost
        elif workload == "msmarco":
            # RAG: clustered high scores (document sections)
            scores = np.random.uniform(0.1, 0.3, num_blocks)
            # Add 2-3 high-attention clusters
            for _ in range(3):
                cluster_start = np.random.randint(0, num_blocks - 20)
                scores[cluster_start:cluster_start + 20] = np.random.uniform(0.8, 1.0, 20)
        else:  # humaneval
            # Code: local hotspots around current function
            scores = np.random.uniform(0.1, 0.4, num_blocks)
            # Current function context
            function_pos = int(num_blocks * 0.7)
            scores[max(0, function_pos - 10):function_pos + 10] = np.random.uniform(0.7, 1.0, 20)

        # Normalize scores
        scores = scores / scores.max()

        # Simulate evictions (assume 30% of blocks evicted under 12% GPU)
        num_evicted = int(num_blocks * 0.3)

        if policy == "lru":
            # LRU: evict oldest blocks (low indices)
            evicted = blocks[:num_evicted]
        elif policy == "attention":
            # Attention: evict lowest-scoring blocks
            evicted = sorted(blocks, key=lambda b: scores[b])[:num_evicted]
        else:  # hybrid
            # Hybrid: mix of low score and old
            score_evict = sorted(blocks, key=lambda b: scores[b])[:num_evicted // 2]
            age_evict = blocks[:num_evicted - len(score_evict)]
            evicted = score_evict + age_evict

        kept = [b for b in blocks if b not in evicted]

        # Generate access counts (higher for recent blocks)
        access_counts = {b: max(1, int(10 * scores[b])) for b in blocks}

        # Generate eviction timeline
        timeline = []
        for t, block_id in enumerate(evicted):
            timeline.append({
                "time": t * 0.1,
                "block_id": block_id,
                "score": float(scores[block_id]),
                "reason": "low_score" if policy == "attention" else "age",
            })

        data.append(RequestEvictionData(
            request_id=f"{workload}_{i}",
            workload=workload,
            prompt_length=num_blocks * 48,  # Assume 48 tokens/block
            output_length=256,
            total_blocks=num_blocks,
            evicted_blocks=evicted,
            kept_blocks=kept,
            block_scores={b: float(scores[b]) for b in blocks},
            block_access_counts=access_counts,
            eviction_timeline=timeline,
        ))

    return data


def compute_summary_stats(requests: list[RequestEvictionData]) -> dict[str, float]:
    """Compute summary statistics across all requests."""
    all_evicted_scores = []
    all_kept_scores = []

    for req in requests:
        evicted_scores = [req.block_scores[b] for b in req.evicted_blocks if b in req.block_scores]
        kept_scores = [req.block_scores[b] for b in req.kept_blocks if b in req.block_scores]

        all_evicted_scores.extend(evicted_scores)
        all_kept_scores.extend(kept_scores)

    return {
        "mean_evicted_score": float(np.mean(all_evicted_scores)) if all_evicted_scores else 0.0,
        "mean_kept_score": float(np.mean(all_kept_scores)) if all_kept_scores else 0.0,
        "std_evicted_score": float(np.std(all_evicted_scores)) if all_evicted_scores else 0.0,
        "std_kept_score": float(np.std(all_kept_scores)) if all_kept_scores else 0.0,
        "score_separation": float(np.mean(all_kept_scores) - np.mean(all_evicted_scores)) if (all_kept_scores and all_evicted_scores) else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Collect eviction data for visualization"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Model to test",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sharegpt",
        choices=["sharegpt", "msmarco", "humaneval"],
        help="Dataset/workload type",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to collect",
    )
    parser.add_argument(
        "--policies",
        type=str,
        nargs="+",
        default=["lru", "attention"],
        help="Policies to collect data for",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eviction_data.json"),
        help="Output JSON file",
    )
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Use synthetic data (for testing visualization without instrumentation)",
    )

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Eviction Data Collection")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.num_samples}")
    print(f"Policies: {', '.join(args.policies)}")
    print(f"Synthetic: {args.use_synthetic}")
    print(f"{'='*70}\n")

    all_datasets = []

    for policy in args.policies:
        print(f"\nCollecting data for {policy} policy...")

        if args.use_synthetic:
            requests = generate_synthetic_eviction_data(
                args.num_samples,
                args.dataset,
                policy,
            )
        else:
            # REAL IMPLEMENTATION: Would load prompts and run instrumented inference
            print("\n⚠️  WARNING: Real data collection requires vLLM instrumentation.")
            print("This requires modifying:")
            print("  - vllm/v1/core/block_manager.py: Log eviction decisions")
            print("  - vllm/v1/worker/gpu_model_runner.py: Log attention scores")
            print("\nFalling back to synthetic data for now...\n")

            requests = generate_synthetic_eviction_data(
                args.num_samples,
                args.dataset,
                policy,
            )

        summary_stats = compute_summary_stats(requests)

        dataset = EvictionDataset(
            model=args.model,
            workload=args.dataset,
            policy=policy,
            num_samples=len(requests),
            requests=requests,
            summary_stats=summary_stats,
        )

        all_datasets.append(dataset)

        print(f"\n{policy.upper()} Policy Statistics:")
        print(f"  Mean score (evicted blocks): {summary_stats['mean_evicted_score']:.4f}")
        print(f"  Mean score (kept blocks):    {summary_stats['mean_kept_score']:.4f}")
        print(f"  Score separation:            {summary_stats['score_separation']:.4f}")

    # Save to JSON
    output_data = [asdict(ds) for ds in all_datasets]

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Data saved to: {args.output}")
    print(f"{'='*70}\n")

    print("Next steps:")
    print("1. Visualize: python scripts/visualize_attention_patterns.py --data", args.output)
    print("2. For real data: Instrument block_manager.py and gpu_model_runner.py")
    print("   - Add eviction logging in evict() method")
    print("   - Add attention score logging in model runner")
    print("   - Export via RequestOutput.metrics")

    return 0


if __name__ == "__main__":
    sys.exit(main())
