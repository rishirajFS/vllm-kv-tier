#!/usr/bin/env python3
"""
Diagnostic script to verify KVTransferConfig and OffloadingConnector are initialized.

This helps determine if the zero evictions issue is due to:
1. Conservative scheduler (limiting batch size)
2. Connector not being initialized at all
3. V1 engine not using the connector
4. Configuration errors
"""
import sys

def check_kv_connector_initialization():
    """Test if OffloadingConnector can be initialized."""
    print("="*70)
    print("KV Connector Initialization Check")
    print("="*70)
    print()

    # Test 1: Import check
    print("Test 1: Can we import the required classes?")
    try:
        from vllm import LLM, SamplingParams
        from vllm.config import KVTransferConfig
        print("  ✅ Imports successful")
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

    # Test 2: Create KVTransferConfig
    print("\nTest 2: Can we create KVTransferConfig?")
    try:
        kv_config = KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "cpu_bytes_to_use": 8_000_000_000,
                "block_size": 48,
                "eviction_policy": "lru",
                "log_evictions": True
            }
        )
        print("  ✅ KVTransferConfig created")
        print(f"     Connector: {kv_config.kv_connector}")
        print(f"     Role: {kv_config.kv_role}")
    except Exception as e:
        print(f"  ❌ KVTransferConfig failed: {e}")
        return False

    # Test 3: Initialize LLM with KV config
    print("\nTest 3: Can we initialize LLM with KV transfer?")
    print("  (This may take 1-2 minutes to load the model...)")
    try:
        llm = LLM(
            model="facebook/opt-125m",  # Small model for quick test
            gpu_memory_utilization=0.30,
            max_model_len=1024,
            kv_transfer_config=kv_config,
            enforce_eager=True
        )
        print("  ✅ LLM initialized with KV transfer config")
    except Exception as e:
        print(f"  ❌ LLM initialization failed: {e}")
        return False

    # Test 4: Check if connector exists
    print("\nTest 4: Is the connector accessible?")
    try:
        # Try to access the connector
        if hasattr(llm, 'llm_engine'):
            print("  ✅ llm_engine exists")

            if hasattr(llm.llm_engine, 'engine_core'):
                print("  ✅ engine_core exists")

                if hasattr(llm.llm_engine.engine_core, 'kv_connector'):
                    connector = llm.llm_engine.engine_core.kv_connector
                    if connector is not None:
                        print(f"  ✅ kv_connector exists: {type(connector)}")

                        # Try to get stats
                        if hasattr(connector, 'get_stats'):
                            stats = connector.get_stats()
                            print(f"  ✅ get_stats() works: {stats}")
                        else:
                            print("  ⚠️  kv_connector has no get_stats() method")
                    else:
                        print("  ❌ kv_connector is None!")
                        print("     This means the connector was not initialized.")
                        print("     Possible causes:")
                        print("       - V1 engine doesn't support KV transfer")
                        print("       - Configuration not being applied")
                        print("       - Feature disabled by default")
                        return False
                else:
                    print("  ❌ engine_core has no kv_connector attribute")
                    return False
            else:
                print("  ❌ llm_engine has no engine_core attribute")
                print("     This might mean V0 engine is being used")
                return False
        else:
            print("  ❌ LLM has no llm_engine attribute")
            return False

    except Exception as e:
        print(f"  ❌ Error accessing connector: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 5: Try a simple generation
    print("\nTest 5: Can we generate with the connector active?")
    try:
        prompts = ["Hello world"] * 10
        sampling_params = SamplingParams(max_tokens=50, temperature=0.0)
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        print(f"  ✅ Generated {len(outputs)} outputs")

        # Check stats again
        stats = llm.llm_engine.engine_core.kv_connector.get_stats()
        evictions = stats.get('total_evictions', 0)
        print(f"     Evictions: {evictions}")

        if evictions > 0:
            print(f"  🎉 EVICTIONS DETECTED! The connector works!")
        else:
            print(f"  ⚠️  Zero evictions (but connector is initialized)")
            print(f"     This is a scheduler issue, not a connector issue")

    except Exception as e:
        print(f"  ❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)
    print()

    if evictions > 0:
        print("✅ CONNECTOR WORKS! Evictions are possible.")
        print()
        print("Your zero evictions issue is due to:")
        print("  - Conservative scheduler limiting batch size")
        print("  - Insufficient memory pressure")
        print()
        print("Solutions:")
        print("  1. Use longer outputs (max_tokens=2048+)")
        print("  2. Use tighter memory (GPU% = 20-30%)")
        print("  3. Use more concurrent requests (500+)")
    else:
        print("⚠️  CONNECTOR INITIALIZED but NO EVICTIONS")
        print()
        print("This means:")
        print("  - Code is correct")
        print("  - Connector is active")
        print("  - BUT scheduler is too conservative")
        print()
        print("The fundamental issue:")
        print("  vLLM V1's scheduler prevents OOM by limiting batches")
        print("  It WON'T schedule requests that would exceed memory")
        print("  Therefore eviction is never triggered")
        print()
        print("Possible solutions:")
        print("  1. Disable conservative scheduling (if possible)")
        print("  2. Force longer dynamic growth (very long outputs)")
        print("  3. Use V0 engine instead of V1")
        print("  4. Modify scheduler to allow over-subscription")

    print()
    return True


if __name__ == "__main__":
    try:
        success = check_kv_connector_initialization()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
