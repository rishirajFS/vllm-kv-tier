#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Adds debug print statements to attention_manager.py to verify evictions.

This creates a debug-instrumented version of the manager that prints
every eviction event to stdout.

Usage:
    python scripts/add_eviction_debug.py

This will create:
    vllm/v1/kv_offload/attention_manager_debug.py
"""
from pathlib import Path

def add_debug_prints():
    """Add debug prints to attention manager."""
    vllm_root = Path(__file__).parent.parent
    original_file = vllm_root / "vllm/v1/kv_offload/attention_manager.py"
    debug_file = vllm_root / "vllm/v1/kv_offload/attention_manager_debug.py"

    if not original_file.exists():
        print(f"❌ Error: {original_file} not found")
        return False

    # Read original file
    with open(original_file) as f:
        content = f.read()

    # Add debug prints at key locations
    debug_content = content.replace(
        "        num_to_evict = (",
        """        # [DEBUG] Check if eviction is needed
        num_free = self.backend.get_num_free_blocks()
        num_to_store = len(block_hashes_to_store)
        print(f"[EVICTION DEBUG] Need to store {num_to_store} blocks, have {num_free} free")

        num_to_evict = ("""
    )

    debug_content = debug_content.replace(
        "        if num_to_evict > 0:",
        """        if num_to_evict > 0:
            print(f"[EVICTION DEBUG] 🚨 EVICTING {num_to_evict} blocks!")"""
    )

    debug_content = debug_content.replace(
        "        # Evict selected blocks",
        """        # Evict selected blocks
        if to_evict:
            print(f"[EVICTION DEBUG] Selected {len(to_evict)} blocks for eviction: {to_evict[:3]}...")"""
    )

    debug_content = debug_content.replace(
        "            # Log eviction for instrumentation/visualization\n            if self.log_evictions:",
        """            # Log eviction for instrumentation/visualization
            print(f"[EVICTION DEBUG] Evicting block {block_hash}, score={meta.cumulative_attention_score:.4f}")
            if self.log_evictions:"""
    )

    debug_content = debug_content.replace(
        "        # Allocate new blocks\n        statuses = self.backend.allocate_blocks(block_hashes_to_store)",
        """        # Allocate new blocks
        print(f"[EVICTION DEBUG] Allocating {len(block_hashes_to_store)} new blocks")
        statuses = self.backend.allocate_blocks(block_hashes_to_store)
        print(f"[EVICTION DEBUG] Allocated successfully: {len(statuses)} blocks")"""
    )

    # Write debug version
    with open(debug_file, 'w') as f:
        f.write(debug_content)

    print(f"✅ Created debug version: {debug_file}")
    print(f"\nTo use the debug version:")
    print(f"1. Rename original: mv {original_file} {original_file}.backup")
    print(f"2. Use debug version: cp {debug_file} {original_file}")
    print(f"3. Run your benchmark")
    print(f"4. Restore original: mv {original_file}.backup {original_file}")
    print(f"\nOr edit your benchmark to import from attention_manager_debug instead")

    return True


if __name__ == "__main__":
    success = add_eviction_debug()
    exit(0 if success else 1)
