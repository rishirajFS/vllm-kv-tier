#!/bin/bash
# Quick test to find GPU memory setting that triggers evictions

set -e

echo "========================================="
echo "Quick Eviction Trigger Test"
echo "========================================="
echo ""

# Test 1: Extreme memory pressure
echo "Test 1: 4% GPU, 2K max_len (extreme pressure)"
python -c "
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

kv_config = KVTransferConfig(
    kv_connector='OffloadingConnector',
    kv_role='kv_both',
    kv_connector_extra_config={
        'cpu_bytes_to_use': 8_000_000_000,
        'eviction_policy': 'lru',
        'log_evictions': True,
    },
)

llm = LLM(
    model='Qwen/Qwen2.5-1.5B-Instruct',
    gpu_memory_utilization=0.04,
    max_model_len=2048,
    kv_transfer_config=kv_config,
)

# Long prompt to force memory usage
prompt = 'Explain machine learning in detail. ' * 100

outputs = llm.generate([prompt] * 3, SamplingParams(max_tokens=256))

# Check stats
try:
    stats = llm.llm_engine.engine_core.kv_connector.get_stats()
    evictions = stats.get('total_evictions', 0)
    print(f'')
    print(f'Test 1 Result: {evictions} evictions')
    if evictions > 0:
        print('✅ SUCCESS: Evictions triggered!')
        exit(0)
    else:
        print('❌ FAIL: No evictions')
except:
    print('❌ Could not get stats')
" || echo "Test 1 failed or OOM"

echo ""
echo "Test 2: 6% GPU, 2K max_len"
python -c "
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

kv_config = KVTransferConfig(
    kv_connector='OffloadingConnector',
    kv_role='kv_both',
    kv_connector_extra_config={
        'cpu_bytes_to_use': 8_000_000_000,
        'eviction_policy': 'lru',
    },
)

llm = LLM(
    model='Qwen/Qwen2.5-1.5B-Instruct',
    gpu_memory_utilization=0.06,
    max_model_len=2048,
    kv_transfer_config=kv_config,
)

prompt = 'Explain machine learning in detail. ' * 100
outputs = llm.generate([prompt] * 3, SamplingParams(max_tokens=256))

try:
    stats = llm.llm_engine.engine_core.kv_connector.get_stats()
    evictions = stats.get('total_evictions', 0)
    print(f'')
    print(f'Test 2 Result: {evictions} evictions')
    if evictions > 0:
        print('✅ SUCCESS: Evictions triggered!')
        exit(0)
except:
    print('❌ Could not get stats')
" || echo "Test 2 failed or OOM"

echo ""
echo "Test 3: 8% GPU, 2K max_len"
python -c "
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

kv_config = KVTransferConfig(
    kv_connector='OffloadingConnector',
    kv_role='kv_both',
    kv_connector_extra_config={
        'cpu_bytes_to_use': 8_000_000_000,
        'eviction_policy': 'lru',
    },
)

llm = LLM(
    model='Qwen/Qwen2.5-1.5B-Instruct',
    gpu_memory_utilization=0.08,
    max_model_len=2048,
    kv_transfer_config=kv_config,
)

prompt = 'Explain machine learning in detail. ' * 100
outputs = llm.generate([prompt] * 3, SamplingParams(max_tokens=256))

try:
    stats = llm.llm_engine.engine_core.kv_connector.get_stats()
    evictions = stats.get('total_evictions', 0)
    print(f'')
    print(f'Test 3 Result: {evictions} evictions')
    if evictions > 0:
        print('✅ SUCCESS: Evictions triggered!')
except:
    print('❌ Could not get stats')
" || echo "Test 3 failed"

echo ""
echo "========================================="
echo "Test Complete"
echo "========================================="
