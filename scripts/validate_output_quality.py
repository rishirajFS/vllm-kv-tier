#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Quality Validation - Priority 5 from final_plan.md

Proves that KV cache eviction doesn't hurt generation quality by comparing
outputs from GPU-only baseline vs tiered systems.

Expected results:
- ROUGE-L: >0.98 (nearly identical)
- BERTScore: >0.95 (semantically similar)
- Exact match: 85-95% (small variations due to numerical precision)

If quality drops below 0.90, indicates a bug in the implementation.

Usage:
    python scripts/validate_output_quality.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --num-samples 100 \
        --output quality_validation.json
"""
import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class QualityMetrics:
    """Quality comparison metrics."""
    policy: str
    num_samples: int
    rouge_l_precision: float
    rouge_l_recall: float
    rouge_l_f1: float
    bertscore_precision: float
    bertscore_recall: float
    bertscore_f1: float
    exact_match_rate: float
    avg_length_diff: float
    avg_edit_distance: int


def compute_rouge_l(reference: str, hypothesis: str) -> dict[str, float]:
    """
    Compute ROUGE-L score (Longest Common Subsequence).

    Args:
        reference: Reference text
        hypothesis: Hypothesis text to compare

    Returns:
        Dict with precision, recall, and F1
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return {
            "precision": scores['rougeL'].precision,
            "recall": scores['rougeL'].recall,
            "f1": scores['rougeL'].fmeasure,
        }
    except ImportError:
        print("Warning: rouge_score not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rouge-score"])
        return compute_rouge_l(reference, hypothesis)


def compute_bertscore(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """
    Compute BERTScore (semantic similarity).

    Args:
        references: List of reference texts
        hypotheses: List of hypothesis texts

    Returns:
        Dict with avg precision, recall, and F1
    """
    try:
        from bert_score import score
        P, R, F1 = score(hypotheses, references, lang="en", verbose=False)
        return {
            "precision": float(P.mean()),
            "recall": float(R.mean()),
            "f1": float(F1.mean()),
        }
    except ImportError:
        print("Warning: bert_score not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "bert-score"])
        return compute_bertscore(references, hypotheses)


def compute_edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance."""
    if len(s1) < len(s2):
        return compute_edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def generate_baseline_outputs(
    model: str,
    prompts: list[str],
    max_tokens: int = 256,
) -> list[str]:
    """
    Generate outputs using GPU-only baseline (no tiering).

    Args:
        model: Model name
        prompts: List of prompts
        max_tokens: Maximum tokens to generate

    Returns:
        List of generated texts
    """
    from vllm import LLM, SamplingParams

    print(f"\nGenerating baseline outputs (GPU-only, no tiering)...")

    llm = LLM(
        model=model,
        gpu_memory_utilization=0.9,  # High utilization, no tiering needed
        max_model_len=8192,
    )

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,  # Deterministic
        seed=42,  # Fixed seed for reproducibility
    )

    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    texts = [o.outputs[0].text for o in outputs]

    del llm
    return texts


def generate_policy_outputs(
    model: str,
    prompts: list[str],
    policy: str,
    max_tokens: int = 256,
    gpu_mem_util: float = 0.12,
) -> list[str]:
    """
    Generate outputs using specified eviction policy.

    Args:
        model: Model name
        prompts: List of prompts
        policy: Eviction policy ("lru", "attention", "hybrid")
        max_tokens: Maximum tokens to generate
        gpu_mem_util: GPU memory utilization

    Returns:
        List of generated texts
    """
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    print(f"\nGenerating outputs with {policy} policy...")

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
        temperature=0.0,  # Deterministic
        seed=42,  # Same seed as baseline
    )

    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    texts = [o.outputs[0].text for o in outputs]

    del llm
    return texts


def compare_outputs(
    baseline_outputs: list[str],
    policy_outputs: list[str],
    policy_name: str,
) -> QualityMetrics:
    """
    Compare baseline outputs against policy outputs.

    Args:
        baseline_outputs: Outputs from GPU-only baseline
        policy_outputs: Outputs from policy
        policy_name: Name of the policy

    Returns:
        QualityMetrics with comparison results
    """
    print(f"\nComputing quality metrics for {policy_name}...")

    # Compute ROUGE-L for each pair
    rouge_scores = [compute_rouge_l(ref, hyp) for ref, hyp in zip(baseline_outputs, policy_outputs)]
    avg_rouge_p = np.mean([s["precision"] for s in rouge_scores])
    avg_rouge_r = np.mean([s["recall"] for s in rouge_scores])
    avg_rouge_f1 = np.mean([s["f1"] for s in rouge_scores])

    # Compute BERTScore for all pairs
    bertscore = compute_bertscore(baseline_outputs, policy_outputs)

    # Compute exact match rate
    exact_matches = sum(1 for ref, hyp in zip(baseline_outputs, policy_outputs) if ref == hyp)
    exact_match_rate = exact_matches / len(baseline_outputs)

    # Compute average length difference
    length_diffs = [abs(len(ref) - len(hyp)) for ref, hyp in zip(baseline_outputs, policy_outputs)]
    avg_length_diff = np.mean(length_diffs)

    # Compute average edit distance (sample 50 to avoid expensive computation)
    sample_size = min(50, len(baseline_outputs))
    indices = np.random.choice(len(baseline_outputs), sample_size, replace=False)
    edit_distances = [
        compute_edit_distance(baseline_outputs[i], policy_outputs[i])
        for i in indices
    ]
    avg_edit_distance = int(np.mean(edit_distances))

    metrics = QualityMetrics(
        policy=policy_name,
        num_samples=len(baseline_outputs),
        rouge_l_precision=float(avg_rouge_p),
        rouge_l_recall=float(avg_rouge_r),
        rouge_l_f1=float(avg_rouge_f1),
        bertscore_precision=float(bertscore["precision"]),
        bertscore_recall=float(bertscore["recall"]),
        bertscore_f1=float(bertscore["f1"]),
        exact_match_rate=float(exact_match_rate),
        avg_length_diff=float(avg_length_diff),
        avg_edit_distance=avg_edit_distance,
    )

    # Print summary
    print(f"\nQuality Metrics for {policy_name}:")
    print(f"  ROUGE-L F1:       {metrics.rouge_l_f1:.4f}")
    print(f"  BERTScore F1:     {metrics.bertscore_f1:.4f}")
    print(f"  Exact Match Rate: {metrics.exact_match_rate:.2%}")
    print(f"  Avg Length Diff:  {metrics.avg_length_diff:.1f} chars")
    print(f"  Avg Edit Distance: {metrics.avg_edit_distance}")

    # Check for potential bugs
    if metrics.rouge_l_f1 < 0.90:
        print(f"\n⚠️  WARNING: ROUGE-L F1 < 0.90 indicates potential bug!")
        print(f"  Quality degradation detected. Check:")
        print(f"  - Fetching wrong blocks from CPU")
        print(f"  - Corrupting shared blocks")
        print(f"  - Race condition in async transfers")
    elif metrics.rouge_l_f1 > 0.98:
        print(f"\n✓ Excellent quality preservation (ROUGE-L > 0.98)")

    return metrics


def load_prompts(dataset: str = "sharegpt", num_samples: int = 100) -> list[str]:
    """
    Load prompts from dataset.

    Args:
        dataset: Dataset name ("sharegpt", "msmarco", "humaneval")
        num_samples: Number of samples to load

    Returns:
        List of prompts
    """
    import os

    dataset_dir = os.path.expanduser("~/workspace/vllm/datasets")
    dataset_path = Path(dataset_dir) / f"{dataset}.json"

    if not dataset_path.exists():
        print(f"Warning: {dataset_path} not found. Using synthetic prompts.")
        return [f"Explain the concept of {topic} in detail."
                for topic in ["machine learning", "quantum computing", "blockchain",
                            "neural networks", "distributed systems"] * (num_samples // 5)]

    with open(dataset_path) as f:
        data = json.load(f)

    # Extract prompts based on dataset format
    prompts = []
    for item in data[:num_samples]:
        if dataset == "sharegpt":
            # ShareGPT format: conversations list
            if "conversations" in item and len(item["conversations"]) > 0:
                prompts.append(item["conversations"][0].get("value", ""))
        elif dataset == "msmarco":
            # MS-MARCO format: query or prompt field
            prompts.append(item.get("prompt", item.get("query", "")))
        elif dataset == "humaneval":
            # HumanEval format: prompt field
            prompts.append(item.get("prompt", ""))

    return prompts[:num_samples]


def main():
    parser = argparse.ArgumentParser(
        description="Quality validation: compare baseline vs tiered outputs"
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
        help="Dataset to use for prompts",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to test",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per sample",
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
        default=Path("quality_validation.json"),
        help="Output JSON file",
    )

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Quality Validation")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.num_samples}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Policies: {', '.join(args.policies)}")
    print(f"{'='*70}\n")

    # Load prompts
    prompts = load_prompts(args.dataset, args.num_samples)
    print(f"Loaded {len(prompts)} prompts from {args.dataset}")

    # Generate baseline outputs
    baseline_outputs = generate_baseline_outputs(args.model, prompts, args.max_tokens)

    # Test each policy
    all_metrics = []

    for policy in args.policies:
        policy_outputs = generate_policy_outputs(
            args.model,
            prompts,
            policy,
            args.max_tokens,
            gpu_mem_util=0.12,
        )

        metrics = compare_outputs(baseline_outputs, policy_outputs, policy)
        all_metrics.append(metrics)

    # Save results
    with open(args.output, "w") as f:
        json.dump([asdict(m) for m in all_metrics], f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {args.output}")
    print(f"{'='*70}\n")

    # Print summary table
    print("Quality Comparison Summary:")
    print("Policy     | ROUGE-L F1 | BERTScore F1 | Exact Match | Status")
    print("-----------|------------|--------------|-------------|--------")

    for metrics in all_metrics:
        status = "✓ PASS" if metrics.rouge_l_f1 > 0.95 else "⚠ CHECK" if metrics.rouge_l_f1 > 0.90 else "✗ FAIL"
        print(f"{metrics.policy:10s} | {metrics.rouge_l_f1:10.4f} | {metrics.bertscore_f1:12.4f} | "
              f"{metrics.exact_match_rate:10.1%} | {status}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
