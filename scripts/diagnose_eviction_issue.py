#!/usr/bin/env python3
"""
Diagnostic script to understand why evictions are not happening.

This script analyzes the benchmark configuration and calculates whether
evictions SHOULD be happening based on memory constraints.
"""
import json
import sys
from pathlib import Path


def analyze_memory_capacity(model_size_gb, gpu_mem_util, max_model_len, num_prompts, max_tokens):
    """
    Calculate if evictions should occur given the configuration.

    Returns:
        dict with analysis results
    """
    # V100 has 32GB total
    v100_total_gb = 32

    # Available GPU memory
    gpu_available_gb = v100_total_gb * gpu_mem_util

    # Model takes up space
    kv_cache_available_gb = gpu_available_gb - model_size_gb

    # KV cache size per token (rough estimate)
    # For transformers: 2 × num_layers × hidden_dim × num_heads
    # Typical: ~256 bytes per token for Qwen models
    bytes_per_token = 256

    # Total tokens needed for concurrent batch
    # Each prompt can have up to max_model_len tokens (input + output)
    tokens_per_request = max_model_len  # Worst case
    total_tokens_needed = num_prompts * tokens_per_request

    # KV cache needed (bytes)
    kv_cache_needed_bytes = total_tokens_needed * bytes_per_token
    kv_cache_needed_gb = kv_cache_needed_bytes / (1024**3)

    # Can it fit?
    fits_in_gpu = kv_cache_needed_gb <= kv_cache_available_gb

    # If sequential processing (1 at a time), calculate per-request
    tokens_per_sequential = tokens_per_request
    kv_cache_sequential_gb = tokens_per_sequential * bytes_per_token / (1024**3)
    fits_sequential = kv_cache_sequential_gb <= kv_cache_available_gb

    return {
        "gpu_total_gb": v100_total_gb,
        "gpu_allocated_gb": gpu_available_gb,
        "model_size_gb": model_size_gb,
        "kv_cache_available_gb": kv_cache_available_gb,
        "kv_cache_needed_concurrent_gb": kv_cache_needed_gb,
        "kv_cache_needed_sequential_gb": kv_cache_sequential_gb,
        "fits_in_gpu_concurrent": fits_in_gpu,
        "fits_in_gpu_sequential": fits_sequential,
        "evictions_expected_concurrent": not fits_in_gpu,
        "evictions_expected_sequential": not fits_sequential,
        "num_prompts": num_prompts,
        "max_model_len": max_model_len,
        "max_tokens": max_tokens,
        "gpu_memory_utilization": gpu_mem_util,
    }


def analyze_result_file(filepath):
    """Analyze a single result JSON file."""
    with open(filepath) as f:
        data = json.load(f)

    # Handle both single result and list of results
    results = data if isinstance(data, list) else [data]

    print(f"\n{'='*70}")
    print(f"File: {filepath.name}")
    print(f"{'='*70}\n")

    for result in results:
        policy = result.get('policy', 'unknown')
        model = result.get('model', result.get('config', {}).get('model', 'unknown'))

        # Get config
        config = result.get('config', {})
        gpu_mem_util = config.get('gpu_memory_utilization', result.get('gpu_memory_util', 0))
        max_model_len = config.get('max_model_len', 8192)
        num_prompts = result.get('num_prompts', config.get('num_prompts', 0))
        max_tokens = config.get('max_tokens', 256)

        # Get eviction count
        evictions = result.get('total_evictions', result.get('eviction_count', 0))

        # Estimate model size
        if '1.5B' in model or '1B' in model:
            model_size_gb = 3.0
        elif '3B' in model:
            model_size_gb = 6.0
        elif '7B' in model:
            model_size_gb = 14.0
        else:
            model_size_gb = 6.0  # Default guess

        analysis = analyze_memory_capacity(
            model_size_gb, gpu_mem_util, max_model_len, num_prompts, max_tokens
        )

        print(f"Policy: {policy}")
        print(f"Model: {model}")
        print(f"\nConfiguration:")
        print(f"  GPU Memory Util: {gpu_mem_util*100:.0f}%")
        print(f"  Max Model Len: {max_model_len:,} tokens")
        print(f"  Num Prompts: {num_prompts}")
        print(f"  Max Tokens (output): {max_tokens}")

        print(f"\nMemory Analysis:")
        print(f"  GPU Allocated: {analysis['gpu_allocated_gb']:.2f} GB")
        print(f"  Model Size: {analysis['model_size_gb']:.2f} GB")
        print(f"  KV Cache Available: {analysis['kv_cache_available_gb']:.2f} GB")

        print(f"\nKV Cache Requirements:")
        print(f"  Concurrent (all {num_prompts} prompts): {analysis['kv_cache_needed_concurrent_gb']:.2f} GB")
        print(f"  Sequential (1 prompt): {analysis['kv_cache_needed_sequential_gb']:.4f} GB")

        print(f"\nCapacity Check:")
        print(f"  Fits if CONCURRENT? {analysis['fits_in_gpu_concurrent']}")
        print(f"  Fits if SEQUENTIAL? {analysis['fits_in_gpu_sequential']}")

        print(f"\nExpected Evictions:")
        print(f"  If CONCURRENT processing: {analysis['evictions_expected_concurrent']}")
        print(f"  If SEQUENTIAL processing: {analysis['evictions_expected_sequential']}")

        print(f"\nActual Results:")
        print(f"  Evictions: {evictions}")
        print(f"  Status: {'✅ AS EXPECTED' if evictions == 0 and analysis['fits_in_gpu_sequential'] else '❌ UNEXPECTED'}")

        if evictions == 0 and not analysis['fits_in_gpu_concurrent']:
            print(f"\n⚠️  DIAGNOSIS: No evictions despite tight memory!")
            print(f"   Likely causes:")
            print(f"   1. vLLM is processing requests SEQUENTIALLY (one at a time)")
            print(f"   2. Batch size is too small to saturate GPU memory")
            print(f"   3. OffloadingConnector not initialized properly")
            print(f"   4. V1 engine uses different memory management")

        print()


def main():
    results_dir = Path("/Users/rishi/Downloads/LLMsys_Project/vllm/benchmark_results")

    print("\n" + "="*70)
    print("EVICTION DIAGNOSTIC ANALYSIS")
    print("="*70)

    # Analyze a few key result files
    key_files = [
        "results_qwen1.5b_sharegpt_20260404_093906.json",
        "results_qwen3b_sharegpt_20260404_105824.json",
        "results_qwen7b_sharegpt_20260405_015013.json",
        "memory_sweep_20260405_015013.json",
    ]

    for filename in key_files:
        filepath = results_dir / filename
        if filepath.exists():
            analyze_result_file(filepath)
        else:
            print(f"\nSkipping {filename} (not found)")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print()
    print("Key Findings:")
    print("1. With SEQUENTIAL processing (1 prompt at a time), KV cache fits in GPU")
    print("   even at low memory utilization (12%, 25%, etc.)")
    print()
    print("2. Only CONCURRENT processing would saturate GPU and trigger evictions")
    print()
    print("3. Possible Issues:")
    print("   - vLLM V1 engine may be processing requests sequentially")
    print("   - Batch size may be too small")
    print("   - max_model_len may be too large (reduces concurrent capacity)")
    print()
    print("Recommended Fixes:")
    print("1. Reduce max_model_len to 1024-2048 (forces more concurrent requests)")
    print("2. Increase num_prompts to 500-1000")
    print("3. Set gpu_memory_utilization to 0.04-0.08 (extreme pressure)")
    print("4. Verify vLLM is actually batching requests concurrently")
    print()


if __name__ == "__main__":
    main()
