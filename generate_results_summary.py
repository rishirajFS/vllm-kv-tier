import json
import glob
import os
from collections import defaultdict

def main():
    results_dir = "benchmark_results"
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    
    print(f"# KV Cache Tiering Benchmark Summary\n")
    print("Parsed all 12 existing results from 4 chronological benchmark runs.\n")
    
    runs = {
        "results_20260401_013454.json": "Run 1: opt-125m (Sequential, Synthetic, No Eviction Pressure)",
        "results_20260401_123047.json": "Run 2: Llama-3.2-1B (Sequential, Synthetic, No Eviction Pressure)",
        "results_20260401_151159.json": "Run 3: Llama-3.2-1B (Batched, Synthetic, No Eviction Pressure)",
        "results_sharegpt_20260401_230944.json": "Run 4: Llama-3.2-1B (Batched, ShareGPT, Extreme Eviction Pressure)"
    }
    
    for file_path in sorted(json_files):
        filename = os.path.basename(file_path)
        with open(file_path, "r") as f:
            data = json.load(f)
            
        print(f"## {runs.get(filename, filename)}")
        print(f"**Dataset:** {data[0].get('dataset', 'synthetic')}")
        print(f"**GPU Utilization:** {data[0]['config']['gpu_memory_utilization']}\n")
        
        print("| Policy | Tokens/sec | Avg Latency (ms) | P95 Latency (ms) | Total Time (s) |")
        print("|---|---|---|---|---|")
        
        for result in data:
            policy = result["policy"].title()
            tok_sec = result["tokens_per_second"]
            avg_lat = result["avg_latency_ms"]
            p95_lat = result["p95_latency_ms"]
            time_sec = result["total_time_seconds"]
            print(f"| {policy} | {tok_sec:.1f} | {avg_lat:.1f} | {p95_lat:.1f} | {time_sec:.1f} |")
        print("\n")

if __name__ == "__main__":
    main()
