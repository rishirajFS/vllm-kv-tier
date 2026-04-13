import os
import json
import urllib.request
from pathlib import Path

TASKS = ["narrativeqa", "qasper", "multi_news"]
OUTPUT_DIR = Path("/Users/rishi/Downloads/LLMsys_Project/vllm/datasets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    for task in TASKS:
        url = f"https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data/{task}.jsonl"
        print(f"Downloading {url}...")
        
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                content = response.read().decode('utf-8')
                
            prompts = []
            for line in content.strip().split('\n'):
                if not line:
                    continue
                item = json.loads(line)
                
                # Format exactly as benchmark.py expects
                prompts.append({
                    "prompt": item.get('input', item.get('context', '')),
                    "expected_output": item.get('answers', [''])[0] if item.get('answers') else "",
                    "context_length": item.get('length', len(item.get('input', '').split())),
                    "task": task
                })
                
                # We only need 100 samples max for the benchmark
                if len(prompts) >= 100:
                    break
                    
            # The SLURM script uses narrative_qa, not narrativeqa
            out_task_name = "narrative_qa" if task == "narrativeqa" else task
            out_file = OUTPUT_DIR / f"longbench_{out_task_name}.json"
            
            with open(out_file, 'w') as f:
                json.dump(prompts, f, indent=2)
            
            print(f"Saved {len(prompts)} prompts to {out_file}")
            
        except Exception as e:
            print(f"Failed to download {task}: {e}")

if __name__ == "__main__":
    main()
