import urllib.request
import zipfile
import json
import os
from pathlib import Path

def main():
    zip_url = "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"
    zip_path = "data.zip"
    
    print("Downloading data.zip...")
    urllib.request.urlretrieve(zip_url, zip_path)
    
    print("Extracting datasets...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")
        
    out_dir = Path("datasets")
    out_dir.mkdir(exist_ok=True)
    
    tasks = {
        "narrativeqa": "narrative_qa",
        "qasper": "qasper",
        "multi_news": "multi_news"
    }
    
    for raw_name, out_name in tasks.items():
        in_file = f"data/{raw_name}.jsonl"
        if not os.path.exists(in_file):
            print(f"Missing {in_file}")
            continue
            
        prompts = []
        with open(in_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                context = item.get('context', '')
                input_txt = item.get('input', '')
                prompt = context + '\n\n' + input_txt if input_txt else context
                
                prompts.append({
                    "prompt": prompt,
                    "expected_output": item.get('answers', [''])[0] if item.get('answers') else "",
                    "context_length": item.get('length', len(prompt.split())),
                    "task": out_name
                })
                if len(prompts) >= 100:
                    break
                    
        out_file = out_dir / f"longbench_{out_name}.json"
        with open(out_file, 'w') as f:
            json.dump(prompts, f, indent=2)
        print(f"Saved {len(prompts)} to {out_file}")

if __name__ == "__main__":
    main()
