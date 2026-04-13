#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Quality validation: Compare outputs with/without eviction.

This script generates outputs using different eviction policies and
compares them using ROUGE-L and BERTScore to prove eviction doesn't
degrade model quality.

Usage:
    python scripts/run_quality_validation.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --dataset ~/vllm/datasets/quality_subset_sharegpt.json \
        --output ~/vllm/benchmark_results/quality_validation.json
"""
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bertscore


def load_prompts(dataset_path: str, max_samples: int = 100) -> List[str]:
    """Load prompts from dataset file."""
    with open(dataset_path) as f:
        data = json.load(f)

    prompts = [item['prompt'] for item in data[:max_samples]]
    print(f"Loaded {len(prompts)} prompts from {dataset_path}")
    return prompts


def generate_baseline_outputs(model_name: str, prompts: List[str]) -> List[str]:
    """
    Generate baseline outputs without eviction (GPU-only).

    This uses high GPU memory utilization to avoid any eviction.
    """
    from vllm import LLM, SamplingParams

    print("\n" + "="*70)
    print("Generating Baseline Outputs (No Eviction)")
    print("="*70 + "\n")

    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.90,  # High GPU = no eviction
        max_model_len=2048,
        enforce_eager=True
    )

    sampling_params = SamplingParams(
        max_tokens=256,
        temperature=0.0  # Deterministic for comparison
    )

    print(f"Running {len(prompts)} prompts...")
    start = time.time()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    elapsed = time.time() - start

    output_texts = [o.outputs[0].text for o in outputs]

    print(f"✅ Completed in {elapsed:.1f}s")
    print(f"   Avg tokens: {sum(len(o.outputs[0].token_ids) for o in outputs) / len(outputs):.0f}")

    del llm
    return output_texts


def generate_policy_outputs(
    model_name: str,
    prompts: List[str],
    policy: str
) -> Dict:
    """
    Generate outputs with eviction enabled using specified policy.
    """
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    print("\n" + "="*70)
    print(f"Generating Outputs with {policy.upper()} Eviction")
    print("="*70 + "\n")

    # Configure KV transfer with eviction
    kv_config = KVTransferConfig(
        kv_connector='OffloadingConnector',
        kv_role='kv_both',
        kv_connector_extra_config={
            'cpu_bytes_to_use': 8_000_000_000,
            'block_size': 48,
            'eviction_policy': policy,
            'log_evictions': True,
        }
    )

    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.12,  # Low GPU = force evictions
        max_model_len=2048,
        kv_transfer_config=kv_config,
        enforce_eager=True
    )

    sampling_params = SamplingParams(
        max_tokens=256,
        temperature=0.0  # Deterministic
    )

    print(f"Running {len(prompts)} prompts...")
    start = time.time()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    elapsed = time.time() - start

    output_texts = [o.outputs[0].text for o in outputs]

    # Get eviction stats
    stats = {
        'total_evictions': 0,
        'bytes_gpu_to_cpu': 0,
        'bytes_cpu_to_gpu': 0
    }

    try:
        connector_stats = llm.llm_engine.engine_core.kv_connector.get_stats()
        stats['total_evictions'] = connector_stats.get('total_evictions', 0)
        stats['bytes_gpu_to_cpu'] = connector_stats.get('bytes_gpu_to_cpu', 0)
        stats['bytes_cpu_to_gpu'] = connector_stats.get('bytes_cpu_to_gpu', 0)
    except:
        pass

    print(f"✅ Completed in {elapsed:.1f}s")
    print(f"   Evictions: {stats['total_evictions']}")
    print(f"   Avg tokens: {sum(len(o.outputs[0].token_ids) for o in outputs) / len(outputs):.0f}")

    del llm

    return {
        'policy': policy,
        'outputs': output_texts,
        'time_seconds': elapsed,
        **stats
    }


def compute_rouge_scores(
    baseline: List[str],
    comparison: List[str]
) -> Dict:
    """Compute ROUGE-L scores between baseline and comparison outputs."""
    print("\nComputing ROUGE-L scores...")

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    scores = []
    for ref, hyp in zip(baseline, comparison):
        score = scorer.score(ref, hyp)
        scores.append(score['rougeL'].fmeasure)

    mean_score = np.mean(scores)
    median_score = np.median(scores)
    min_score = np.min(scores)

    print(f"   Mean ROUGE-L: {mean_score:.4f}")
    print(f"   Median: {median_score:.4f}")
    print(f"   Min: {min_score:.4f}")

    return {
        'mean': float(mean_score),
        'median': float(median_score),
        'min': float(min_score),
        'scores': [float(s) for s in scores]
    }


def compute_bert_scores(
    baseline: List[str],
    comparison: List[str]
) -> Dict:
    """Compute BERTScore between baseline and comparison outputs."""
    print("\nComputing BERTScore...")

    # BERTScore computes precision, recall, F1
    P, R, F1 = bertscore(
        comparison,  # candidates
        baseline,     # references
        lang='en',
        verbose=False
    )

    # Use F1 as primary metric
    mean_f1 = float(F1.mean())
    median_f1 = float(F1.median())
    min_f1 = float(F1.min())

    print(f"   Mean BERTScore F1: {mean_f1:.4f}")
    print(f"   Median: {median_f1:.4f}")
    print(f"   Min: {min_f1:.4f}")

    return {
        'mean': mean_f1,
        'median': median_f1,
        'min': min_f1,
        'f1_scores': F1.tolist(),
        'precision_scores': P.tolist(),
        'recall_scores': R.tolist()
    }


def main():
    parser = argparse.ArgumentParser(
        description='Quality validation for KV cache eviction'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen2.5-3B-Instruct',
        help='Model to test'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to quality subset dataset'
    )
    parser.add_argument(
        '--policies',
        nargs='+',
        default=['lru', 'attention', 'hybrid'],
        help='Eviction policies to test'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=100,
        help='Maximum samples to test'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output file for results'
    )

    args = parser.parse_args()

    print("="*70)
    print("Quality Validation - KV Cache Eviction")
    print("="*70)
    print(f"\nModel: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Policies: {', '.join(args.policies)}")
    print(f"Max samples: {args.max_samples}\n")

    # Load prompts
    prompts = load_prompts(args.dataset, args.max_samples)

    # Generate baseline outputs (no eviction)
    baseline_outputs = generate_baseline_outputs(args.model, prompts)

    # Test each policy
    results = []

    for policy in args.policies:
        # Generate outputs with eviction
        policy_result = generate_policy_outputs(args.model, prompts, policy)

        # Compute quality metrics
        rouge_scores = compute_rouge_scores(
            baseline_outputs,
            policy_result['outputs']
        )

        bert_scores = compute_bert_scores(
            baseline_outputs,
            policy_result['outputs']
        )

        # Combine results
        result = {
            'policy': policy,
            'model': args.model,
            'num_samples': len(prompts),
            'total_evictions': policy_result['total_evictions'],
            'bytes_gpu_to_cpu': policy_result['bytes_gpu_to_cpu'],
            'bytes_cpu_to_gpu': policy_result['bytes_cpu_to_gpu'],
            'generation_time_seconds': policy_result['time_seconds'],
            'rouge_l': rouge_scores,
            'bertscore': bert_scores
        }

        results.append(result)

        # Check quality thresholds
        rouge_threshold = 0.95
        bert_threshold = 0.98

        rouge_pass = rouge_scores['mean'] >= rouge_threshold
        bert_pass = bert_scores['mean'] >= bert_threshold

        print(f"\n{'='*70}")
        print(f"Quality Check - {policy.upper()}")
        print(f"{'='*70}")
        print(f"ROUGE-L: {rouge_scores['mean']:.4f} {'✅ PASS' if rouge_pass else '❌ FAIL'} (threshold: {rouge_threshold})")
        print(f"BERTScore: {bert_scores['mean']:.4f} {'✅ PASS' if bert_pass else '❌ FAIL'} (threshold: {bert_threshold})")
        print()

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Results saved to: {output_path}")

    # Final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70 + "\n")

    print("Quality Metrics (Mean Scores):\n")
    print(f"{'Policy':<12} {'ROUGE-L':<10} {'BERTScore':<10} {'Evictions':<10} {'Status'}")
    print("-" * 60)

    all_pass = True
    for r in results:
        rouge = r['rouge_l']['mean']
        bert = r['bertscore']['mean']
        evictions = r['total_evictions']

        rouge_ok = rouge >= 0.95
        bert_ok = bert >= 0.98
        status = '✅ PASS' if (rouge_ok and bert_ok) else '❌ FAIL'

        if not (rouge_ok and bert_ok):
            all_pass = False

        print(f"{r['policy']:<12} {rouge:<10.4f} {bert:<10.4f} {evictions:<10} {status}")

    print()

    if all_pass:
        print("✅ All policies passed quality validation!")
        print("   Eviction does NOT degrade model output quality.")
    else:
        print("⚠️  Some policies failed quality thresholds.")
        print("   This may indicate an issue with eviction logic.")

    return 0 if all_pass else 1


if __name__ == '__main__':
    exit(main())
