#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Download and convert LongBench datasets for KV cache tiering benchmarks.

This script downloads the recommended subset of LongBench tasks and converts
them to the format expected by our benchmark harness.

Usage:
    python scripts/setup_longbench.py --output ~/vllm/datasets
"""
import argparse
import json
from pathlib import Path
from datasets import load_dataset


# Recommended tasks covering different context lengths and categories
RECOMMENDED_TASKS = {
    'qasper': {
        'category': 'Single-doc QA',
        'avg_length': 3619,
        'description': 'Scientific paper question answering'
    },
    'narrative_qa': {
        'category': 'Single-doc QA',
        'avg_length': 18409,
        'description': 'Book understanding and comprehension'
    },
    'hotpotqa': {
        'category': 'Multi-doc QA',
        'avg_length': 9151,
        'description': 'Multi-hop reasoning across documents'
    },
    'multi_news': {
        'category': 'Summarization',
        'avg_length': 2113,
        'description': 'Multi-document news summarization'
    },
}


def download_and_convert_task(task_name, task_info, output_dir, max_samples=200):
    """
    Download LongBench task and convert to benchmark format.

    Args:
        task_name: Name of the LongBench task
        task_info: Task metadata (category, avg_length, description)
        output_dir: Directory to save converted dataset
        max_samples: Maximum number of samples to include
    """
    print(f"\n{'='*70}")
    print(f"Processing: {task_name}")
    print(f"Category: {task_info['category']}")
    print(f"Avg Length: {task_info['avg_length']:,} tokens")
    print(f"{'='*70}\n")

    try:
        # Download from Hugging Face
        print(f"Downloading {task_name} from THUDM/LongBench...")
        dataset = load_dataset('THUDM/LongBench', task_name, split='test')
        print(f"✅ Downloaded {len(dataset)} samples")

    except Exception as e:
        print(f"❌ Error downloading {task_name}: {e}")
        return None

    # Convert to benchmark format
    prompts = []
    total_length = 0

    for i, item in enumerate(dataset):
        if i >= max_samples:
            break

        # LongBench format:
        # {
        #   "input": "full prompt with context + question",
        #   "context": "just the context",
        #   "answers": ["answer1", "answer2", ...],
        #   "length": token_count,
        #   "all_classes": null
        # }

        prompt_text = item['input']
        expected_output = item['answers'][0] if item['answers'] else ""
        context_length = item.get('length', len(prompt_text.split()))

        prompts.append({
            "prompt": prompt_text,
            "expected_output": expected_output,
            "context_length": context_length,
            "task": task_name,
            "category": task_info['category']
        })

        total_length += context_length

    # Calculate statistics
    avg_length = total_length / len(prompts) if prompts else 0

    print(f"\n📊 Conversion Statistics:")
    print(f"   Samples converted: {len(prompts)}")
    print(f"   Avg context length: {avg_length:.0f} tokens")
    print(f"   Min length: {min(p['context_length'] for p in prompts):,} tokens")
    print(f"   Max length: {max(p['context_length'] for p in prompts):,} tokens")

    # Save to file
    output_file = Path(output_dir) / f"longbench_{task_name}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(prompts, f, indent=2)

    print(f"   Saved to: {output_file}")

    return {
        'task': task_name,
        'file': str(output_file),
        'samples': len(prompts),
        'avg_length': avg_length
    }


def main():
    parser = argparse.ArgumentParser(
        description='Download and convert LongBench datasets'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='~/vllm/datasets',
        help='Output directory for converted datasets'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=200,
        help='Maximum samples per task (default: 200)'
    )
    parser.add_argument(
        '--tasks',
        nargs='+',
        choices=list(RECOMMENDED_TASKS.keys()) + ['all'],
        default='all',
        help='Tasks to download (default: all recommended)'
    )

    args = parser.parse_args()

    # Expand home directory
    output_dir = Path(args.output).expanduser()

    # Determine which tasks to download
    if args.tasks == 'all' or 'all' in args.tasks:
        tasks_to_download = RECOMMENDED_TASKS
    else:
        tasks_to_download = {k: v for k, v in RECOMMENDED_TASKS.items() if k in args.tasks}

    print("="*70)
    print("LongBench Dataset Setup")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Max samples per task: {args.max_samples}")
    print(f"Tasks to download: {', '.join(tasks_to_download.keys())}\n")

    # Download and convert each task
    results = []
    for task_name, task_info in tasks_to_download.items():
        result = download_and_convert_task(
            task_name,
            task_info,
            output_dir,
            args.max_samples
        )
        if result:
            results.append(result)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70 + "\n")

    if results:
        print(f"✅ Successfully converted {len(results)} tasks:\n")
        for r in results:
            print(f"   {r['task']:15s} → {r['samples']:3d} samples, "
                  f"avg {r['avg_length']:5.0f} tokens")

        print(f"\n📁 Files saved to: {output_dir}")
        print("\nNext steps:")
        print("  1. Run benchmarks with these datasets:")
        print("     cd kv_cache_tiering/benchmarks")
        print("     python benchmark.py --dataset longbench_qasper \\")
        print(f"       --dataset-path {output_dir}/longbench_qasper.json \\")
        print("       --model Qwen/Qwen2.5-3B-Instruct")
        print()
        print("  2. Or run full LongBench suite:")
        print("     bash scripts/run_longbench_suite.sh")
    else:
        print("❌ No tasks were successfully converted")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
