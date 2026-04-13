#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Quick test to verify evictions are actually triggering.

This script runs a minimal benchmark with EXTREME memory pressure
to force evictions and verify instrumentation is working.

Usage:
    python scripts/test_eviction_trigger.py

Expected output:
    [EVICTION] messages in stdout
    total_evictions > 0 in final stats
"""
import sys
import time

def test_eviction_trigger():
    """Test if evictions trigger with extreme memory pressure."""
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    print("="*70)
    print("EVICTION TRIGGER TEST")
    print("="*70)
    print()

    # Test configurations - progressively tighter memory
    test_configs = [
        {"gpu": 0.10, "max_len": 4096, "desc": "10% GPU, 4K max"},
        {"gpu": 0.08, "max_len": 4096, "desc": "8% GPU, 4K max"},
        {"gpu": 0.06, "max_len": 2048, "desc": "6% GPU, 2K max"},
        {"gpu": 0.04, "max_len": 2048, "desc": "4% GPU, 2K max (extreme)"},
    ]

    model = "Qwen/Qwen2.5-1.5B-Instruct"

    # Long prompt to force memory usage
    long_prompt = "Explain the history of computer science. " * 200  # ~3K tokens

    for config in test_configs:
        print(f"\n{'='*70}")
        print(f"Test: {config['desc']}")
        print(f"{'='*70}\n")

        kv_config = KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "cpu_bytes_to_use": 8_000_000_000,
                "block_size": 48,
                "eviction_policy": "lru",
                "log_evictions": True,  # Enable logging
            },
        )

        try:
            llm = LLM(
                model=model,
                gpu_memory_utilization=config["gpu"],
                max_model_len=config["max_len"],
                kv_transfer_config=kv_config,
                enforce_eager=True,  # Avoid CUDA graphs for testing
            )

            sampling_params = SamplingParams(
                max_tokens=512,
                temperature=0.0,
            )

            print(f"Running 5 prompts with {config['desc']}...")
            start = time.time()
            outputs = llm.generate([long_prompt] * 5, sampling_params, use_tqdm=False)
            elapsed = time.time() - start

            # Try to get stats
            evictions = 0
            bytes_transferred = 0
            try:
                engine = getattr(llm, "llm_engine", None)
                if engine:
                    core = getattr(engine, "engine_core", None)
                    if core:
                        connector = getattr(core, "kv_connector", None)
                        if connector:
                            stats = connector.get_stats()
                            evictions = stats.get("total_evictions", 0)
                            bytes_transferred = stats.get("bytes_gpu_to_cpu", 0)

                            print(f"\n📊 STATS:")
                            print(f"  Total evictions: {evictions}")
                            print(f"  Bytes GPU→CPU: {bytes_transferred:,}")
                            print(f"  Bytes CPU→GPU: {stats.get('bytes_cpu_to_gpu', 0):,}")

                            # Check eviction log
                            eviction_log = connector.get_eviction_log()
                            if eviction_log:
                                print(f"  Eviction log entries: {len(eviction_log)}")
                            else:
                                print(f"  Eviction log: None")
            except Exception as e:
                print(f"  ⚠️  Could not get stats: {e}")

            total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            throughput = total_tokens / elapsed if elapsed > 0 else 0

            print(f"\n⏱️  PERFORMANCE:")
            print(f"  Throughput: {throughput:.1f} tok/s")
            print(f"  Time: {elapsed:.1f}s")

            # Verdict
            if evictions > 0:
                print(f"\n✅ SUCCESS: {evictions} evictions triggered!")
                print(f"   This configuration WORKS for forcing evictions.")
                del llm
                return config  # Return working config
            else:
                print(f"\n❌ FAIL: Zero evictions")
                print(f"   Memory pressure insufficient.")

            del llm

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n💥 OOM: Memory too tight (this is actually good - means we're close)")
                print(f"   Try slightly higher GPU memory.")
            else:
                print(f"\n❌ Error: {e}")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print("❌ NONE of the configurations triggered evictions!")
    print("   This suggests a deeper issue with eviction triggering.")
    print("='*70}")
    return None


if __name__ == "__main__":
    working_config = test_eviction_trigger()

    if working_config:
        print(f"\n{'='*70}")
        print("🎯 WORKING CONFIGURATION FOUND:")
        print(f"  GPU memory utilization: {working_config['gpu']}")
        print(f"  Max model length: {working_config['max_len']}")
        print(f"\nUse these settings in your benchmarks:")
        print(f"  --gpu-memory-utilization {working_config['gpu']}")
        print(f"  --max-model-len {working_config['max_len']}")
        print(f"{'='*70}\n")
        sys.exit(0)
    else:
        print(f"\n{'='*70}")
        print("⚠️  INVESTIGATION NEEDED:")
        print("  Evictions did not trigger even with extreme memory pressure.")
        print("  Possible causes:")
        print("  1. Eviction counter is broken (not incrementing)")
        print("  2. Backend allocation strategy changed")
        print("  3. Offloading connector not being used")
        print(f"{'='*70}\n")
        sys.exit(1)
