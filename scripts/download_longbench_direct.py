#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Download LongBench-v2 datasets via huggingface_hub.
LongBench-v2 is the updated accessible HuggingFace repo (THUDM/LongBench-v2).

Usage:
    python scripts/download_longbench_direct.py --output ~/workspace/vllm/datasets
"""
import argparse
import json
import os
from pathlib import Path

# Force real HuggingFace endpoint — Bridges-2 overrides this to a broken mirror
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ["HUGGINGFACE_HUB_URL"] = "https://huggingface.co"

HF_REPO = "THUDM/LongBench-v2"

# LongBench-v2 uses a single split; items have a "domain" field.
# We sample up to max_samples per domain for the 4 key domains.
DOMAINS = {
    "single_doc_qa":   {"avg_length": 8000,  "category": "Single-doc QA"},
    "multi_doc_qa":    {"avg_length": 15000, "category": "Multi-doc QA"},
    "long_in_context": {"avg_length": 20000, "category": "Long In-Context"},
    "summarization":   {"avg_length": 12000, "category": "Summarization"},
}


def download_and_convert(output_dir, max_samples_per_domain=50, hf_token=None):
    """Download LongBench-v2 and split by domain into separate JSON files."""
    from huggingface_hub import hf_hub_download

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing = [d for d in DOMAINS if (output_dir / f"longbench_{d}.json").exists()]
    if len(existing) == len(DOMAINS):
        print("✅ All LongBench-v2 datasets already cached.")
        return True

    print(f"Downloading {HF_REPO} parquet data...")
    try:
        # LongBench-v2 stores data in data/train-00000-of-00001.parquet
        local_path = hf_hub_download(
            repo_id=HF_REPO,
            filename="data/train-00000-of-00001.parquet",
            repo_type="dataset",
            token=hf_token,
        )
        import pyarrow.parquet as pq
        records = pq.read_table(local_path).to_pylist()
        print(f"  Loaded {len(records)} total records")
    except Exception as e:
        print(f"❌ Failed to download parquet: {e}")
        print("  Trying snapshot_download fallback...")
        try:
            from huggingface_hub import snapshot_download
            cache_dir = Path("~/workspace/vllm/hf_cache").expanduser()
            local_dir = snapshot_download(
                repo_id=HF_REPO,
                repo_type="dataset",
                local_dir=str(cache_dir / "longbench_v2"),
                allow_patterns=["*.parquet"],
                token=hf_token,
            )
            import glob, pyarrow.parquet as pq
            pq_files = glob.glob(f"{local_dir}/**/*.parquet", recursive=True)
            if not pq_files:
                raise RuntimeError("No parquet files found in snapshot")
            records = []
            for f in pq_files:
                records.extend(pq.read_table(f).to_pylist())
            print(f"  Loaded {len(records)} records from snapshot")
        except Exception as e2:
            print(f"❌ Both download methods failed: {e2}")
            return False

    # Group by domain
    by_domain = {}
    for item in records:
        domain = item.get("domain", "unknown")
        if domain in DOMAINS:
            by_domain.setdefault(domain, []).append(item)

    success = 0
    for domain, info in DOMAINS.items():
        out_file = output_dir / f"longbench_{domain}.json"
        if out_file.exists():
            print(f"✅ {domain}: already cached")
            success += 1
            continue

        items = by_domain.get(domain, [])[:max_samples_per_domain]
        if not items:
            print(f"⚠️  {domain}: no records found in dataset")
            continue

        prompts = []
        for item in items:
            prompt_text = item.get("context", item.get("input", ""))
            answers = item.get("answer", item.get("answers", []))
            expected = (answers[0] if isinstance(answers, list) else answers) or ""
            context_length = item.get("length", len(prompt_text.split()))
            prompts.append({
                "prompt": prompt_text,
                "expected_output": str(expected),
                "context_length": context_length,
                "task": domain,
                "category": info["category"],
            })

        with open(out_file, "w") as f:
            json.dump(prompts, f, indent=2)

        avg_len = sum(p["context_length"] for p in prompts) / max(len(prompts), 1)
        print(f"✅ {domain}: {len(prompts)} samples, avg {avg_len:.0f} tokens → {out_file}")
        success += 1

    print(f"\n{'='*60}")
    print(f"Saved {success}/{len(DOMAINS)} domain files")
    return success > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="~/workspace/vllm/datasets")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN")
    ok = download_and_convert(args.output, args.max_samples, token)
    return 0 if ok else 1


if __name__ == "__main__":
    exit(main())
