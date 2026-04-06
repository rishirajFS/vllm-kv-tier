import json
import glob
import os
import collections

def main():
    results_dir = os.path.expanduser("~/workspace/vllm/benchmark_results")
    files = glob.glob(os.path.join(results_dir, "longbench_*.json"))
    
    if not files:
        print("No LongBench result files found in", results_dir)
        return
        
    print(f"Found {len(files)} result files. Parsing...\n")
    print("=" * 80)
    print(f"{'Dataset':<25} | {'Policy':<10} | {'Throughput (tok/s)':<18} | {'Total Evictions':<15}")
    print("-" * 80)
    
    for f in sorted(files):
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                for entry in data:
                    dataset = entry.get("dataset", "unknown")
                    policy = entry.get("policy", "unknown")
                    tps = entry.get("tokens_per_second", 0.0)
                    evictions = entry.get("total_evictions", 0)
                    
                    print(f"{dataset:<25} | {policy:<10} | {tps:<18.2f} | {evictions:<15}")
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    print("=" * 80)
    
if __name__ == "__main__":
    main()
