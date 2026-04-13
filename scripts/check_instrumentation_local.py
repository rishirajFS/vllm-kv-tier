#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Check eviction instrumentation code structure WITHOUT GPU.

This script verifies the instrumentation code is properly set up
without actually running inference (no GPU needed).

Can run locally on Mac.
"""
import sys
from pathlib import Path

def check_instrumentation():
    """Check if eviction instrumentation is properly set up."""
    print("="*70)
    print("Checking Eviction Instrumentation (No GPU Required)")
    print("="*70)
    print()

    vllm_root = Path(__file__).parent.parent
    issues = []
    successes = []

    # Check 1: Managers have eviction logging
    print("1. Checking if managers have eviction logging...")

    manager_files = [
        "vllm/v1/kv_offload/attention_manager.py",
        "vllm/v1/kv_offload/lru_manager.py",
        "vllm/v1/kv_offload/hybrid_manager.py",
    ]

    for manager_file in manager_files:
        file_path = vllm_root / manager_file
        if not file_path.exists():
            issues.append(f"   ❌ {manager_file} not found")
            continue

        with open(file_path) as f:
            content = f.read()

        has_log_evictions = "log_evictions" in content
        has_eviction_record = "EvictionRecord" in content
        has_get_eviction_log = "def get_eviction_log" in content

        if has_log_evictions and has_eviction_record and has_get_eviction_log:
            successes.append(f"   ✅ {manager_file.split('/')[-1]} has eviction logging")
        else:
            missing = []
            if not has_log_evictions: missing.append("log_evictions param")
            if not has_eviction_record: missing.append("EvictionRecord")
            if not has_get_eviction_log: missing.append("get_eviction_log()")
            issues.append(f"   ❌ {manager_file.split('/')[-1]} missing: {', '.join(missing)}")

    # Check 2: Instrumentation module exists
    print("\n2. Checking instrumentation module...")

    instrumentation_file = vllm_root / "vllm/v1/kv_offload/instrumentation.py"
    if instrumentation_file.exists():
        with open(instrumentation_file) as f:
            content = f.read()

        if "class OffloadingMetrics" in content and "total_evictions" in content:
            successes.append("   ✅ instrumentation.py has OffloadingMetrics with total_evictions")
        else:
            issues.append("   ❌ instrumentation.py missing OffloadingMetrics or total_evictions")
    else:
        issues.append("   ❌ instrumentation.py not found")

    # Check 3: Connector passes eviction data
    print("\n3. Checking connector integration...")

    connector_file = vllm_root / "vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py"
    if connector_file.exists():
        with open(connector_file) as f:
            content = f.read()

        has_eviction_log_field = "eviction_log: list[dict]" in content
        has_get_eviction_log_method = "def get_eviction_log" in content
        calls_manager_get_log = "manager.get_eviction_log()" in content

        if has_eviction_log_field and has_get_eviction_log_method and calls_manager_get_log:
            successes.append("   ✅ offloading_connector.py has eviction_log pipeline")
        else:
            missing = []
            if not has_eviction_log_field: missing.append("eviction_log field")
            if not has_get_eviction_log_method: missing.append("get_eviction_log() method")
            if not calls_manager_get_log: missing.append("calls manager.get_eviction_log()")
            issues.append(f"   ❌ offloading_connector.py missing: {', '.join(missing)}")
    else:
        issues.append("   ❌ offloading_connector.py not found")

    # Check 4: Stats tracking
    print("\n4. Checking stats integration...")

    # Check if manager increments eviction counter
    manager_file = vllm_root / "vllm/v1/kv_offload/attention_manager.py"
    manager_tracks_evictions = False
    manager_exposes_evictions = False

    if manager_file.exists():
        with open(manager_file) as f:
            content = f.read()

        # Check for counter field
        has_counter = "_total_evictions" in content
        # Check for increment
        increments_counter = "_total_evictions += 1" in content
        # Check if exposed in get_stats
        exposes_in_stats = '"total_evictions": self._total_evictions' in content

        if has_counter and increments_counter:
            manager_tracks_evictions = True
            successes.append("   ✅ Manager tracks and increments total_evictions")
        else:
            issues.append("   ❌ Manager doesn't track eviction counter")

        if exposes_in_stats:
            manager_exposes_evictions = True
            successes.append("   ✅ Manager exposes total_evictions in get_stats()")
        else:
            issues.append("   ❌ Manager doesn't expose total_evictions")

    # Check if connector aggregates manager stats
    if connector_file.exists():
        with open(connector_file) as f:
            content = f.read()

        has_get_stats = "def get_stats(self) -> dict:" in content
        aggregates_manager = "manager.get_stats()" in content and "stats.update(manager_stats)" in content

        if has_get_stats and aggregates_manager:
            successes.append("   ✅ Connector aggregates manager stats via get_stats()")
        else:
            issues.append("   ❌ Connector doesn't properly aggregate manager stats")

    # Check 5: CPU backend config
    print("\n5. Checking CPU backend config...")

    cpu_file = vllm_root / "vllm/v1/kv_offload/cpu.py"
    if cpu_file.exists():
        with open(cpu_file) as f:
            content = f.read()

        if "log_evictions" in content:
            successes.append("   ✅ cpu.py reads log_evictions from config")
        else:
            issues.append("   ❌ cpu.py doesn't read log_evictions config")
    else:
        issues.append("   ❌ cpu.py not found")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print()

    if successes:
        print("✅ Working:")
        for s in successes:
            print(s)

    print()

    if issues:
        print("⚠️  Issues Found:")
        for i in issues:
            print(i)
        print()
        print("🔧 Possible Root Cause:")
        print("   The eviction_log is being tracked in managers, but")
        print("   total_evictions stat is not being incremented in the connector.")
        print()
        print("   This explains why you see 0 evictions even if they're happening!")
    else:
        print("✅ All instrumentation checks passed!")
        print()
        print("   The code structure looks correct. If you're still seeing")
        print("   zero evictions, the issue is likely:")
        print("   1. GPU memory is too high (no eviction needed)")
        print("   2. max_model_len is too high (no eviction needed)")
        print("   3. Batch size is too small (no memory pressure)")

    print()
    print("="*70)
    print("NEXT STEP")
    print("="*70)
    print()

    if issues:
        print("Fix the instrumentation issues above, then run on GPU:")
        print("  ssh bridges2.psc.edu")
        print("  cd ~/vllm")
        print("  python scripts/test_eviction_trigger.py")
    else:
        print("Instrumentation looks good! Run on GPU to test:")
        print("  ssh bridges2.psc.edu")
        print("  cd ~/vllm")
        print("  python scripts/test_eviction_trigger.py")
        print()
        print("Or submit SLURM job:")
        print("  sbatch scripts/slurm_eviction_test.sh")

    print()

    return len(issues) == 0


if __name__ == "__main__":
    success = check_instrumentation()
    sys.exit(0 if success else 1)
