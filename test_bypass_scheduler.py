from vllm import LLM, SamplingParams
import torch
import os
from vllm.v1.metrics.reader import get_metrics_snapshot, Histogram

# Direct memory control - bypass scheduler
# We use a tight gpu_memory_utilization to force the engine to use less room for KV cache
# which combined with long prompts should trigger evictions.
model = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    gpu_memory_utilization=0.80, # Increased to ensure model loads
    max_model_len=16384,
    enforce_eager=True,  # Disable CUDA graphs for more dynamic memory
    disable_log_stats=False, # Enable logging for telemetry
    kv_cache_metrics=True,  # Enable KV cache metrics for v1
    trust_remote_code=True
)

# Force long sequences to trigger eviction
# 10,000 "words" is roughly 13-15k tokens. 20 prompts = ~300k tokens.
long_prompts = ["Summarize this document: " + "word " * 10000] * 20

sampling_params = SamplingParams(max_tokens=512)

print(f"===============================================================")
print(f"Generating for {len(long_prompts)} long prompts...")
print(f"===============================================================")

# This should trigger eviction due to long context + tight memory
outputs = model.generate(long_prompts, sampling_params)

# Check eviction stats using v1 metrics reader
metrics = get_metrics_snapshot()
eviction_count = 0
for metric in metrics:
    # In v1, evictions are tracked in the lifetime histogram
    if metric.name == "vllm:kv_block_lifetime_seconds" and isinstance(metric, Histogram):
        eviction_count = metric.count
        break

print(f"\n===============================================================")
print(f"FINAL STATS:")
print(f"Total KV Cache Block Evictions: {eviction_count}")
print(f"===============================================================")
