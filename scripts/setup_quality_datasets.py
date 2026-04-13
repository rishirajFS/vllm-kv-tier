#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Download and prepare datasets for quality validation.

This script sets up MMLU and TriviaQA datasets to validate that
KV cache eviction doesn't degrade model output quality.

Usage:
    python scripts/setup_quality_datasets.py --output ~/vllm/datasets
"""
import argparse
import json
from pathlib import Path
from datasets import load_dataset


def setup_mmlu(output_dir, max_samples=1000):
    """
    Download and prepare MMLU dataset.

    MMLU (Massive Multitask Language Understanding) is a multiple-choice
    benchmark covering 57 subjects. We'll use it to verify eviction doesn't
    hurt model accuracy.

    Args:
        output_dir: Directory to save prepared dataset
        max_samples: Maximum samples to include (default 1000)
    """
    print("\n" + "="*70)
    print("Setting up MMLU (Quality Validation)")
    print("="*70 + "\n")

    try:
        # MMLU has multiple subjects - we'll sample across all
        print("Downloading MMLU dataset from cais/mmlu...")
        dataset = load_dataset('cais/mmlu', 'all', split='test')
        print(f"✅ Downloaded {len(dataset)} samples")

    except Exception as e:
        print(f"❌ Error downloading MMLU: {e}")
        return None

    # Convert to benchmark format
    prompts = []
    subjects = set()

    for i, item in enumerate(dataset):
        if i >= max_samples:
            break

        # MMLU format:
        # {
        #   "question": "What is the capital of France?",
        #   "subject": "geography",
        #   "choices": ["London", "Berlin", "Paris", "Madrid"],
        #   "answer": 2  # Index of correct answer
        # }

        question = item['question']
        choices = item['choices']
        correct_idx = item['answer']
        subject = item.get('subject', 'unknown')

        subjects.add(subject)

        # Format as multiple choice
        choices_text = "\n".join([f"{chr(65+i)}. {choice}"
                                  for i, choice in enumerate(choices)])

        prompt = f"Question: {question}\n\nChoices:\n{choices_text}\n\nAnswer:"
        correct_answer = chr(65 + correct_idx)

        prompts.append({
            "prompt": prompt,
            "correct_answer": correct_answer,
            "correct_index": correct_idx,
            "subject": subject,
            "choices": choices
        })

    # Statistics
    print(f"\n📊 Conversion Statistics:")
    print(f"   Samples converted: {len(prompts)}")
    print(f"   Subjects covered: {len(subjects)}")
    print(f"   Example subjects: {', '.join(list(subjects)[:5])}")

    # Save
    output_file = Path(output_dir) / "mmlu.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(prompts, f, indent=2)

    print(f"   Saved to: {output_file}")

    return {
        'dataset': 'mmlu',
        'file': str(output_file),
        'samples': len(prompts),
        'subjects': len(subjects)
    }


def setup_triviaqa(output_dir, max_samples=500):
    """
    Download and prepare TriviaQA dataset.

    TriviaQA is a reading comprehension dataset with question-answer pairs
    and evidence documents. We'll use it to validate RAG performance.

    Args:
        output_dir: Directory to save prepared dataset
        max_samples: Maximum samples to include (default 500)
    """
    print("\n" + "="*70)
    print("Setting up TriviaQA (RAG Robustness)")
    print("="*70 + "\n")

    try:
        # TriviaQA has multiple configs - we'll use 'unfiltered'
        print("Downloading TriviaQA dataset from trivia_qa...")
        dataset = load_dataset('trivia_qa', 'unfiltered', split='validation')
        print(f"✅ Downloaded {len(dataset)} samples")

    except Exception as e:
        print(f"❌ Error downloading TriviaQA: {e}")
        return None

    # Convert to benchmark format
    prompts = []
    total_context_length = 0

    for i, item in enumerate(dataset):
        if i >= max_samples:
            break

        # TriviaQA format:
        # {
        #   "question": "What is the capital of France?",
        #   "answer": {"value": "Paris", "aliases": ["Paris", "paris"]},
        #   "search_results": {"search_context": ["...", "..."], ...}
        # }

        question = item['question']
        answer = item['answer']['value']

        # Get evidence context (use first search result)
        evidence = ""
        if item['search_results'] and item['search_results']['search_context']:
            evidence = item['search_results']['search_context'][0]
            # Truncate very long evidence
            evidence = evidence[:2000]

        # Build RAG-style prompt
        if evidence:
            prompt = f"Context: {evidence}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"Question: {question}\n\nAnswer:"

        context_length = len(prompt.split())
        total_context_length += context_length

        prompts.append({
            "prompt": prompt,
            "expected_answer": answer,
            "question": question,
            "has_evidence": bool(evidence),
            "context_length": context_length
        })

    # Statistics
    avg_length = total_context_length / len(prompts) if prompts else 0
    has_evidence_count = sum(1 for p in prompts if p['has_evidence'])

    print(f"\n📊 Conversion Statistics:")
    print(f"   Samples converted: {len(prompts)}")
    print(f"   Samples with evidence: {has_evidence_count}")
    print(f"   Avg context length: {avg_length:.0f} tokens")

    # Save
    output_file = Path(output_dir) / "triviaqa.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(prompts, f, indent=2)

    print(f"   Saved to: {output_file}")

    return {
        'dataset': 'triviaqa',
        'file': str(output_file),
        'samples': len(prompts),
        'avg_length': avg_length
    }


def setup_quality_subset(output_dir, source_dataset='sharegpt', max_samples=100):
    """
    Create a subset of an existing dataset for quality comparison.

    We'll use this to compare outputs with/without eviction and compute
    ROUGE-L and BERTScore metrics.

    Args:
        output_dir: Directory to save subset
        source_dataset: Which dataset to sample from
        max_samples: Number of samples to include
    """
    print("\n" + "="*70)
    print(f"Creating Quality Subset from {source_dataset}")
    print("="*70 + "\n")

    # For quality validation, we just need prompts (not full conversations)
    # We'll generate outputs with different policies and compare

    if source_dataset == 'sharegpt':
        try:
            dataset = load_dataset('anon8231489123/ShareGPT_Vicuna_unfiltered', split='train')
            print(f"✅ Loaded {len(dataset)} ShareGPT conversations")

            prompts = []
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break

                # Get first user message as prompt
                conversations = item.get('conversations', [])
                if conversations and conversations[0].get('from') == 'human':
                    prompt_text = conversations[0].get('value', '')
                    if prompt_text:
                        prompts.append({
                            "prompt": prompt_text,
                            "source": "sharegpt",
                            "conversation_id": i
                        })

            print(f"   Extracted {len(prompts)} prompts")

        except Exception as e:
            print(f"❌ Error loading ShareGPT: {e}")
            return None

    else:
        print(f"❌ Unknown source dataset: {source_dataset}")
        return None

    # Save
    output_file = Path(output_dir) / f"quality_subset_{source_dataset}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(prompts, f, indent=2)

    print(f"   Saved to: {output_file}")

    return {
        'dataset': f'quality_subset_{source_dataset}',
        'file': str(output_file),
        'samples': len(prompts)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Setup quality validation datasets (MMLU, TriviaQA)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='~/vllm/datasets',
        help='Output directory for datasets'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['mmlu', 'triviaqa', 'quality_subset', 'all'],
        default='all',
        help='Which datasets to setup (default: all)'
    )
    parser.add_argument(
        '--mmlu-samples',
        type=int,
        default=1000,
        help='Max MMLU samples (default: 1000)'
    )
    parser.add_argument(
        '--triviaqa-samples',
        type=int,
        default=500,
        help='Max TriviaQA samples (default: 500)'
    )
    parser.add_argument(
        '--quality-samples',
        type=int,
        default=100,
        help='Max quality subset samples (default: 100)'
    )

    args = parser.parse_args()

    # Expand home directory
    output_dir = Path(args.output).expanduser()

    print("="*70)
    print("Quality Validation Dataset Setup")
    print("="*70)
    print(f"\nOutput directory: {output_dir}\n")

    # Determine which datasets to setup
    datasets_to_setup = args.datasets
    if 'all' in datasets_to_setup:
        datasets_to_setup = ['mmlu', 'triviaqa', 'quality_subset']

    results = []

    # Setup each dataset
    if 'mmlu' in datasets_to_setup:
        result = setup_mmlu(output_dir, args.mmlu_samples)
        if result:
            results.append(result)

    if 'triviaqa' in datasets_to_setup:
        result = setup_triviaqa(output_dir, args.triviaqa_samples)
        if result:
            results.append(result)

    if 'quality_subset' in datasets_to_setup:
        result = setup_quality_subset(output_dir, 'sharegpt', args.quality_samples)
        if result:
            results.append(result)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70 + "\n")

    if results:
        print(f"✅ Successfully prepared {len(results)} dataset(s):\n")
        for r in results:
            dataset = r['dataset']
            samples = r['samples']
            print(f"   {dataset:20s} → {samples:4d} samples")

        print(f"\n📁 Files saved to: {output_dir}")
        print("\nNext steps:")
        print("  1. Run quality validation:")
        print("     sbatch scripts/slurm_quality_validation.sh")
        print()
        print("  2. Or run MMLU accuracy test:")
        print("     python kv_cache_tiering/benchmarks/benchmark.py \\")
        print(f"       --dataset mmlu \\")
        print(f"       --dataset-path {output_dir}/mmlu.json")
    else:
        print("❌ No datasets were successfully prepared")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
