#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Download LongBench datasets via huggingface_hub parquet API.
Bypasses the deprecated dataset script loader and SSL certificate issues.

Usage:
    python scripts/download_longbench_direct.py --output ~/workspace/vllm/datasets
"""
import argparse
import json
import os
from pathlib import Path

HF_REPO = "THUDM/LongBench"

TASKS = {
    "qasper":       {"avg_length": 3619,  "category": "Single-doc QA"},
    "narrative_qa": {"avg_length": 18409, "category": "Single-doc QA"},
    "hotpotqa":     {"avg_length": 9151,  "category": "Multi-doc QA"},
    "multi_news":   {"avg_length": 2113,  "category": "Summarization"},
}


def download_task(task_name, task_info, output_dir, max_samples=200, hf_token=None):
    """Download a LongBench task via huggingface_hub + parquet."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"longbench_{task_name}.json"

    if out_file.exists():
        print(f"✅ {task_name}: already cached at {out_file}")
        return True

    print(f"Downloading {task_name} parquet...")
    try:
        # LongBench parquet is stored under refs/convert/parquet branch
        local_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=f"{task_name}/test/0000.parquet",
            repo_type="dataset",
            revision="refs/convert/parquet",
            token=hf_token,
        )
        table = pq.read_table(local_path)
        records = table.to_pylist()

    except Exception as e1:
        print(f"  parquet branch failed ({e1}), trying main branch json...")
        try:
            local_path = hf_hub_download(
                repo_id=HF_REPO,
                filename=f"data/{task_name}.jsonl",
                repo_type="dataset",
                token=hf_token,
            )
            with open(local_path) as f:
                records = [json.loads(l) for l in f if l.strip()]
        except Exception as e2:
            print(f"❌ {task_name}: both attempts failed: {e2}")
            return False

    prompts = []
    for item in records[:max_samples]:
        prompt_text = item.get("input", item.get("context", ""))
        answers = item.get("answers", [])
        expected = answers[0] if answers else ""
        context_length = item.get("length", len(prompt_text.split()))
        prompts.append({
            "prompt": prompt_text,
            "expected_output": expected,
            "context_length": context_length,
            "task": task_name,
            "category": task_info["category"],
        })

    with open(out_file, "w") as f:
        json.dump(prompts, f, indent=2)

    avg_len = sum(p["context_length"] for p in prompts) / max(len(prompts), 1)
    print(f"✅ {task_name}: {len(prompts)} samples, avg {avg_len:.0f} tokens → {out_file}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="~/workspace/vllm/datasets")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN")

    success = 0
    for task, info in TASKS.items():
        if download_task(task, info, args.output, args.max_samples, token):
            success += 1

    print(f"\n{'='*60}")
    print(f"Downloaded {success}/{len(TASKS)} tasks")
    return 0 if success > 0 else 1


if __name__ == "__main__":
    exit(main())


# Bridges-2 has broken CA certs for outbound HTTPS — disable SSL verification
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE
from pathlib import Path

# LongBench parquet files on HuggingFace Hub
# URL pattern: https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data/{task}.jsonl.gz
# OR the resolved parquet path
HF_BASE = "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data"

TASKS = {
    "qasper":       {"avg_length": 3619,  "category": "Single-doc QA"},
    "narrative_qa": {"avg_length": 18409, "category": "Single-doc QA"},
    "hotpotqa":     {"avg_length": 9151,  "category": "Multi-doc QA"},
    "multi_news":   {"avg_length": 2113,  "category": "Summarization"},
}


def download_task(task_name, task_info, output_dir, max_samples=200, hf_token=None):
    """Download a LongBench task via direct JSONL URL."""
    import gzip

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"longbench_{task_name}.json"

    if out_file.exists():
        print(f"✅ {task_name}: already exists at {out_file}")
        return True

    url = f"{HF_BASE}/{task_name}.jsonl.gz"
    print(f"Downloading {task_name} from {url} ...")

    headers = {"User-Agent": "Mozilla/5.0"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=120, context=_SSL_CTX) as resp:
            raw = resp.read()

        # Decompress gzip
        with gzip.open(io.BytesIO(raw)) as gz:
            lines = gz.read().decode("utf-8").strip().split("\n")

        prompts = []
        for line in lines[:max_samples]:
            if not line.strip():
                continue
            item = json.loads(line)
            prompt_text = item.get("input", "")
            answers = item.get("answers", [])
            expected = answers[0] if answers else ""
            context_length = item.get("length", len(prompt_text.split()))
            prompts.append({
                "prompt": prompt_text,
                "expected_output": expected,
                "context_length": context_length,
                "task": task_name,
                "category": task_info["category"],
            })

        with open(out_file, "w") as f:
            json.dump(prompts, f, indent=2)

        avg_len = sum(p["context_length"] for p in prompts) / len(prompts) if prompts else 0
        print(f"✅ {task_name}: {len(prompts)} samples, avg {avg_len:.0f} tokens → {out_file}")
        return True

    except Exception as e:
        print(f"❌ {task_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download LongBench via direct parquet/jsonl URLs")
    parser.add_argument("--output", default="~/workspace/vllm/datasets")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--hf-token", default=None, help="HuggingFace token for gated repos")
    args = parser.parse_args()

    import os
    token = args.hf_token or os.environ.get("HF_TOKEN")

    success = 0
    for task, info in TASKS.items():
        if download_task(task, info, args.output, args.max_samples, token):
            success += 1

    print(f"\n{'='*60}")
    print(f"Downloaded {success}/{len(TASKS)} tasks successfully")
    print(f"Output: {Path(args.output).expanduser()}")
    return 0 if success > 0 else 1


if __name__ == "__main__":
    exit(main())
