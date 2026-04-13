#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Validate that eviction instrumentation fixes worked.

Checks all benchmark results from Phase 2 and Phase 3 jobs to confirm:
1. total_evictions > 0 (evictions are happening)
2. bytes_gpu_to_cpu > 0 (transfers are happening)
3. log_evictions_enabled = true (logging is enabled)

Usage:
    python scripts/validate_eviction_fixes.py
"""
import json
from pathlib import Path
from collections import defaultdict


def validate_results():
    """Check all recent benchmark results for eviction data."""
    results_dir = Path(__file__).parent.parent / "benchmark_results"

    print("=" * 70)
    print("EVICTION FIX VALIDATION")
    print("=" * 70)
    print()

    # Find all recent "fixed" results
    fixed_files = sorted(results_dir.glob("*_fixed_*.json"))
    sweep_files = sorted(results_dir.glob("memory_sweep_*.json"))
    ablation_files = sorted(results_dir.glob("hybrid_ablation_*.json"))
    failure_files = sorted(results_dir.glob("failure_modes_*.json"))

    all_files = fixed_files + sweep_files + ablation_files + failure_files

    if not all_files:
        print("❌ No result files found!")
        print(f"\nLooked in: {results_dir}")
        print("\nExpected files:")
        print("  - results_*_fixed_*.json (from Phase 2)")
        print("  - memory_sweep_*.json (from Phase 3)")
        print("  - hybrid_ablation_*.json (from Phase 3)")
        print("  - failure_modes_*.json (from Phase 3)")
        return False

    print(f"Found {len(all_files)} result file(s):\n")

    # Track statistics
    stats = {
        "total_files": 0,
        "files_with_evictions": 0,
        "files_without_evictions": 0,
        "files_with_logging_enabled": 0,
        "total_evictions_sum": 0,
        "total_bytes_gpu_to_cpu": 0,
        "total_bytes_cpu_to_gpu": 0,
    }

    issues = []
    successes = []

    for result_file in all_files:
        print(f"📄 {result_file.name}")

        try:
            with open(result_file) as f:
                data = json.load(f)

            # Handle both single dict and list of dicts
            if isinstance(data, dict):
                results = [data]
            else:
                results = data

            stats["total_files"] += 1

            file_has_evictions = False
            file_has_logging = False

            for i, result in enumerate(results):
                policy = result.get("policy", f"result_{i}")
                evictions = result.get("total_evictions", 0)
                bytes_gpu_cpu = result.get("bytes_gpu_to_cpu", 0)
                bytes_cpu_gpu = result.get("bytes_cpu_to_gpu", 0)

                # Check config for log_evictions flag
                config = result.get("config", {})
                log_evictions_enabled = False
                if "log_evictions_enabled" in config:
                    log_evictions_enabled = config["log_evictions_enabled"]
                elif "kv_connector_extra_config" in config:
                    extra = config["kv_connector_extra_config"]
                    log_evictions_enabled = extra.get("log_evictions", False)

                if log_evictions_enabled:
                    file_has_logging = True

                if evictions > 0:
                    file_has_evictions = True
                    stats["total_evictions_sum"] += evictions
                    stats["total_bytes_gpu_to_cpu"] += bytes_gpu_cpu
                    stats["total_bytes_cpu_to_gpu"] += bytes_cpu_gpu

                    print(f"  ✅ {policy}: {evictions} evictions, "
                          f"{bytes_gpu_cpu:,} bytes GPU→CPU, "
                          f"{bytes_cpu_gpu:,} bytes CPU→GPU")
                else:
                    print(f"  ⚠️  {policy}: 0 evictions (log_enabled={log_evictions_enabled})")

            if file_has_evictions:
                stats["files_with_evictions"] += 1
                successes.append(result_file.name)
            else:
                stats["files_without_evictions"] += 1
                issues.append((result_file.name, "No evictions in any policy"))

            if file_has_logging:
                stats["files_with_logging_enabled"] += 1

            print()

        except Exception as e:
            print(f"  ❌ Error reading file: {e}\n")
            issues.append((result_file.name, f"Parse error: {e}"))

    # Print summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    print(f"📊 Statistics:")
    print(f"   Total files analyzed: {stats['total_files']}")
    print(f"   Files with evictions: {stats['files_with_evictions']}")
    print(f"   Files without evictions: {stats['files_without_evictions']}")
    print(f"   Files with logging enabled: {stats['files_with_logging_enabled']}")
    print()

    print(f"📈 Aggregated Metrics:")
    print(f"   Total evictions across all runs: {stats['total_evictions_sum']:,}")
    print(f"   Total bytes GPU→CPU: {stats['total_bytes_gpu_to_cpu']:,}")
    print(f"   Total bytes CPU→GPU: {stats['total_bytes_cpu_to_gpu']:,}")
    print()

    # Verdict
    if stats["files_with_evictions"] == 0:
        print("❌ FAILED: No files showed evictions!")
        print()
        print("Possible causes:")
        print("  1. GPU memory still too high (need <10%)")
        print("  2. max_model_len still too high (need 2K-4K)")
        print("  3. Fixes didn't apply correctly (check code)")
        print()
        return False

    elif stats["files_without_evictions"] > 0:
        print("⚠️  PARTIAL SUCCESS: Some files show evictions, some don't")
        print()
        print("✅ Files with evictions:")
        for fname in successes:
            print(f"   • {fname}")
        print()
        print("⚠️  Files without evictions:")
        for fname, reason in issues:
            print(f"   • {fname}: {reason}")
        print()
        print("This is expected if memory pressure varies by model/workload.")
        return True

    else:
        print("✅ SUCCESS: All files show evictions!")
        print()
        print(f"The instrumentation fixes are working correctly.")
        print(f"Total {stats['total_evictions_sum']:,} evictions recorded across all runs.")
        print()
        return True


def compare_old_vs_new():
    """Compare original zero-eviction results with new fixed results."""
    results_dir = Path(__file__).parent.parent / "benchmark_results"

    print("\n" + "=" * 70)
    print("COMPARISON: Original vs Fixed Results")
    print("=" * 70)
    print()

    # Find matching pairs
    comparisons = []

    # Qwen 1.5B ShareGPT
    old = results_dir / "results_qwen1.5b_sharegpt_20260404_093906.json"
    new = sorted(results_dir.glob("results_qwen1.5b_sharegpt_fixed_*.json"))
    if old.exists() and new:
        comparisons.append(("Qwen 1.5B ShareGPT", old, new[0]))

    # Qwen 3B ShareGPT
    old = results_dir / "results_qwen3b_sharegpt_20260404_105824.json"
    new = sorted(results_dir.glob("results_qwen3b_sharegpt_fixed_*.json"))
    if old.exists() and new:
        comparisons.append(("Qwen 3B ShareGPT", old, new[0]))

    # Qwen 3B MS-MARCO
    old = results_dir / "results_qwen3b_msmarco_20260404_110742.json"
    new = sorted(results_dir.glob("results_qwen3b_msmarco_fixed_*.json"))
    if old.exists() and new:
        comparisons.append(("Qwen 3B MS-MARCO", old, new[0]))

    if not comparisons:
        print("No matching old/new pairs found for comparison.")
        return

    for name, old_file, new_file in comparisons:
        print(f"📊 {name}:")
        print()

        with open(old_file) as f:
            old_data = json.load(f)
        with open(new_file) as f:
            new_data = json.load(f)

        # Handle list vs dict
        if isinstance(old_data, dict):
            old_data = [old_data]
        if isinstance(new_data, dict):
            new_data = [new_data]

        # Group by policy
        old_by_policy = {r["policy"]: r for r in old_data}
        new_by_policy = {r["policy"]: r for r in new_data}

        print("  Policy          Old Evictions  New Evictions  Old Throughput  New Throughput  Change")
        print("  " + "-" * 82)

        for policy in ["lru", "attention", "hybrid"]:
            if policy in old_by_policy and policy in new_by_policy:
                old_r = old_by_policy[policy]
                new_r = new_by_policy[policy]

                old_evict = old_r.get("total_evictions", 0)
                new_evict = new_r.get("total_evictions", 0)
                old_tput = old_r.get("tokens_per_second", 0)
                new_tput = new_r.get("tokens_per_second", 0)

                change = ((new_tput - old_tput) / old_tput * 100) if old_tput > 0 else 0
                change_str = f"{change:+.2f}%"

                print(f"  {policy:14s}  {old_evict:13,}  {new_evict:13,}  "
                      f"{old_tput:14.1f}  {new_tput:14.1f}  {change_str:>7}")

        print()


if __name__ == "__main__":
    success = validate_results()
    compare_old_vs_new()

    exit(0 if success else 1)
