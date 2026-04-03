# Benchmark Datasets Guide

This guide provides instructions for downloading and preparing datasets for KV cache eviction policy benchmarks.

**Supported Datasets**:
1. **ShareGPT** - Multi-turn conversational data (already used for primary result)
2. **MS-MARCO** - Passage retrieval queries for RAG workloads
3. **HumanEval** - Python code completion problems
4. **Synthetic** - Generated prompts (built-in, no download needed)

---

## Directory Structure

Create a datasets directory in your vLLM workspace:

```bash
mkdir -p ~/workspace/vllm/datasets
cd ~/workspace/vllm/datasets
```

Expected structure after downloads:
```
~/workspace/vllm/datasets/
├── sharegpt.json          # Multi-turn conversations
├── msmarco.json           # RAG queries
└── humaneval.json         # Code completion problems
```

---

## 1. ShareGPT (Conversational Workload)

**Description**: Multi-turn conversations between users and AI assistants. Used for our **primary benchmark result** (9.2% throughput improvement).

**Dataset Size**: ~90K conversations, ~200 MB

**License**: Community dataset, research use

**Source**: Hugging Face `anon8231489123/ShareGPT_Vicuna_unfiltered`

### Download Instructions

**Method 1: Using Hugging Face Datasets (Recommended)**

```python
#!/usr/bin/env python3
"""
Download ShareGPT dataset from Hugging Face and save as JSON.
"""
import json
from datasets import load_dataset

print("Downloading ShareGPT dataset...")
ds = load_dataset('anon8231489123/ShareGPT_Vicuna_unfiltered', split='train')

print(f"Loaded {len(ds)} conversations")

# Save to JSON file
output_path = 'sharegpt.json'
with open(output_path, 'w') as f:
    json.dump(ds.to_list(), f, indent=2)

print(f"Saved to {output_path}")
print(f"File size: {os.path.getsize(output_path) / (1024**2):.1f} MB")
```

**Save as** `download_sharegpt.py` and run:
```bash
pip install datasets
python download_sharegpt.py
```

**Method 2: Manual Download**

```bash
# Alternative: use datasets CLI
datasets-cli download anon8231489123/ShareGPT_Vicuna_unfiltered
# Then convert to JSON manually
```

### Verification

```bash
# Check file exists and is valid JSON
python3 -c "import json; data = json.load(open('sharegpt.json')); print(f'Loaded {len(data)} conversations')"

# Expected output: "Loaded ~90000 conversations"
```

### Dataset Format

```json
[
  {
    "id": "conversation_id",
    "conversations": [
      {"from": "human", "value": "User message..."},
      {"from": "gpt", "value": "Assistant response..."},
      {"from": "human", "value": "Follow-up question..."},
      {"from": "gpt", "value": "Follow-up response..."}
    ]
  },
  ...
]
```

The benchmark harness extracts the first "human" turn from each conversation as the prompt.

---

## 2. MS-MARCO (RAG Workload)

**Description**: Microsoft Machine Reading Comprehension dataset with passage retrieval queries. Simulates RAG (Retrieval-Augmented Generation) workloads where a query is paired with retrieved context passages.

**Dataset Size**: 1M+ queries, ~500 MB (we use a subset)

**License**: Microsoft Research License (free for academic and research use)

**Source**: Hugging Face `ms_marco` dataset v2.1

**Use Case**: Test eviction policies on RAG workloads where attention patterns differ from conversational data (high attention to query tokens from passage tokens).

### Download Instructions

```python
#!/usr/bin/env python3
"""
Download MS-MARCO dataset and extract queries as prompts.
"""
import json
from datasets import load_dataset

print("Downloading MS-MARCO dataset (passage ranking)...")
# Load the passage ranking task (queries + passages)
ds = load_dataset('ms_marco', 'v2.1', split='train', streaming=True)

# Extract first 10,000 queries (adjust as needed)
queries = []
for i, item in enumerate(ds):
    if i >= 10000:
        break

    query = item.get('query', '')
    # Format as RAG prompt: query + instruction to answer based on passages
    prompt = f"Answer the following question: {query}"

    queries.append({
        "query_id": i,
        "query": query,
        "prompt": prompt
    })

    if (i + 1) % 1000 == 0:
        print(f"Processed {i + 1} queries...")

print(f"Extracted {len(queries)} queries")

# Save to JSON file
output_path = 'msmarco.json'
with open(output_path, 'w') as f:
    json.dump(queries, f, indent=2)

print(f"Saved to {output_path}")
```

**Save as** `download_msmarco.py` and run:
```bash
pip install datasets
python download_msmarco.py
```

### Alternative: Smaller Subset (Fast Testing)

For faster testing, use the dev set:

```python
ds = load_dataset('ms_marco', 'v2.1', split='dev')  # ~6980 queries
queries = []
for item in ds:
    query = item.get('query', '')
    queries.append({
        "query": query,
        "prompt": f"Answer the following question: {query}"
    })

with open('msmarco.json', 'w') as f:
    json.dump(queries, f, indent=2)
```

### Verification

```bash
python3 -c "import json; data = json.load(open('msmarco.json')); print(f'Loaded {len(data)} queries'); print(f'Sample: {data[0][\"prompt\"][:100]}...')"

# Expected: "Loaded 10000 queries"
```

### Dataset Format

```json
[
  {
    "query_id": 0,
    "query": "what is the weather in seattle",
    "prompt": "Answer the following question: what is the weather in seattle"
  },
  ...
]
```

The benchmark harness uses the `prompt` field or falls back to `query`.

---

## 3. HumanEval (Code Completion Workload)

**Description**: OpenAI HumanEval dataset with 164 Python programming problems. Each problem includes a function signature, docstring, and test cases.

**Dataset Size**: 164 problems, ~500 KB

**License**: MIT License (free for all use)

**Source**: Hugging Face `openai_humaneval`

**Use Case**: Test eviction policies on code completion workloads with different attention patterns (high attention to function signatures, variable definitions).

### Download Instructions

```python
#!/usr/bin/env python3
"""
Download HumanEval dataset and extract programming problems.
"""
import json
from datasets import load_dataset

print("Downloading HumanEval dataset...")
ds = load_dataset('openai_humaneval', split='test')

print(f"Loaded {len(ds)} programming problems")

# Extract prompts (function signatures + docstrings)
problems = []
for item in ds:
    task_id = item.get('task_id', '')
    prompt = item.get('prompt', '')  # Function signature + docstring

    problems.append({
        "task_id": task_id,
        "prompt": prompt,
        "canonical_solution": item.get('canonical_solution', ''),
        "test": item.get('test', '')
    })

# Save to JSON file
output_path = 'humaneval.json'
with open(output_path, 'w') as f:
    json.dump(problems, f, indent=2)

print(f"Saved to {output_path}")
print(f"Total problems: {len(problems)}")
```

**Save as** `download_humaneval.py` and run:
```bash
pip install datasets
python download_humaneval.py
```

### Verification

```bash
python3 -c "import json; data = json.load(open('humaneval.json')); print(f'Loaded {len(data)} problems'); print(f'Sample problem:\\n{data[0][\"prompt\"]}')"

# Expected: "Loaded 164 problems"
```

### Dataset Format

```json
[
  {
    "task_id": "HumanEval/0",
    "prompt": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
    "canonical_solution": "...",
    "test": "..."
  },
  ...
]
```

The benchmark harness uses the `prompt` field (function signature + docstring) as input.

---

## 4. Synthetic (Built-in, No Download)

**Description**: Programmatically generated prompts with varying lengths. Useful for controlled experiments without external dataset dependencies.

**Generation**: Built into the benchmark harness (`benchmark.py` lines 148-177)

**Use Case**: Quick testing, control experiments

### Usage

```bash
# No download needed, just use --dataset synthetic
python -m kv_cache_tiering.benchmarks.benchmark \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --dataset synthetic \
    --num-prompts 200 \
    --output results_synthetic.json

# Note: --dataset-path not needed for synthetic
```

### Generated Prompt Examples

The harness generates prompts like:
- "Explain the concept of machine learning in detail."
- "Write a comprehensive guide about distributed systems."
- "What are the advantages and disadvantages of quantum computing?"
- "Compare and contrast neural networks with operating systems."

Lengths vary by repeating the base text 1-5 times (i.e., prompt length proportional to `i % 5`).

---

## Dataset Comparison Table

| Dataset | Workload Type | Problems | Avg Tokens/Prompt | Download Size | License | Expected Improvement |
|---------|---------------|----------|-------------------|---------------|---------|---------------------|
| **ShareGPT** | Conversation | 90K | 100-500 | ~200 MB | Community | **+9.2%** (measured) |
| **MS-MARCO** | RAG | 1M | 50-200 | ~500 MB | MSR License | +5-15% (predicted) |
| **HumanEval** | Code | 164 | 50-150 | ~500 KB | MIT | +3-7% (predicted) |
| **Synthetic** | Generic | Unlimited | 50-300 | N/A (generated) | N/A | ±0% (control) |

---

## Complete Setup Script

**Save as** `download_all_datasets.sh`:

```bash
#!/bin/bash
# Download all benchmark datasets

set -e  # Exit on error

DATASETS_DIR="${DATASETS_DIR:-$HOME/workspace/vllm/datasets}"
mkdir -p "$DATASETS_DIR"
cd "$DATASETS_DIR"

echo "Dataset directory: $DATASETS_DIR"
echo "========================================"

# Install dependencies
pip install datasets huggingface_hub --quiet

# ShareGPT
echo "1. Downloading ShareGPT..."
python3 << 'EOF'
import json
from datasets import load_dataset

ds = load_dataset('anon8231489123/ShareGPT_Vicuna_unfiltered', split='train')
with open('sharegpt.json', 'w') as f:
    json.dump(ds.to_list(), f, indent=2)
print(f"ShareGPT: {len(ds)} conversations saved to sharegpt.json")
EOF

# MS-MARCO
echo "2. Downloading MS-MARCO (10K queries)..."
python3 << 'EOF'
import json
from datasets import load_dataset

ds = load_dataset('ms_marco', 'v2.1', split='train', streaming=True)
queries = []
for i, item in enumerate(ds):
    if i >= 10000:
        break
    query = item.get('query', '')
    queries.append({
        "query_id": i,
        "query": query,
        "prompt": f"Answer the following question: {query}"
    })
with open('msmarco.json', 'w') as f:
    json.dump(queries, f, indent=2)
print(f"MS-MARCO: {len(queries)} queries saved to msmarco.json")
EOF

# HumanEval
echo "3. Downloading HumanEval..."
python3 << 'EOF'
import json
from datasets import load_dataset

ds = load_dataset('openai_humaneval', split='test')
problems = []
for item in ds:
    problems.append({
        "task_id": item.get('task_id', ''),
        "prompt": item.get('prompt', ''),
        "canonical_solution": item.get('canonical_solution', ''),
        "test": item.get('test', '')
    })
with open('humaneval.json', 'w') as f:
    json.dump(problems, f, indent=2)
print(f"HumanEval: {len(problems)} problems saved to humaneval.json")
EOF

echo "========================================"
echo "All datasets downloaded successfully!"
echo ""
ls -lh *.json
```

**Run**:
```bash
chmod +x download_all_datasets.sh
./download_all_datasets.sh
```

---

## Usage in Benchmarks

After downloading datasets, use them with the benchmark harness:

### ShareGPT
```bash
python -m kv_cache_tiering.benchmarks.benchmark \
    --dataset sharegpt \
    --dataset-path ~/workspace/vllm/datasets/sharegpt.json \
    --num-prompts 200 \
    --max-tokens 1024 \
    --gpu-mem-util 0.12
```

### MS-MARCO
```bash
python -m kv_cache_tiering.benchmarks.benchmark \
    --dataset msmarco \
    --dataset-path ~/workspace/vllm/datasets/msmarco.json \
    --num-prompts 200 \
    --max-tokens 1024 \
    --gpu-mem-util 0.12
```

### HumanEval
```bash
python -m kv_cache_tiering.benchmarks.benchmark \
    --dataset humaneval \
    --dataset-path ~/workspace/vllm/datasets/humaneval.json \
    --num-prompts 164 \
    --max-tokens 512 \
    --gpu-mem-util 0.12
```

Or use the convenience wrapper scripts (see `scripts/run_msmarco_benchmark.sh` and `scripts/run_humaneval_benchmark.sh`).

---

## Troubleshooting

### Issue: Hugging Face Download Timeout

**Solution**: Set longer timeout
```bash
export HF_DATASETS_OFFLINE=0
export HF_HUB_DOWNLOAD_TIMEOUT=600  # 10 minutes
```

### Issue: Disk Space

**Solution**: Download to external storage
```bash
export DATASETS_DIR=/path/to/external/storage/datasets
./download_all_datasets.sh
```

### Issue: Authentication Required

**Solution**: Login to Hugging Face
```bash
pip install huggingface_hub
huggingface-cli login
# Enter your HF token
```

### Issue: JSON Load Error

**Solution**: Validate JSON format
```bash
python3 -m json.tool sharegpt.json > /dev/null && echo "Valid JSON" || echo "Invalid JSON"
```

---

## License Summary

| Dataset | License | Commercial Use | Attribution Required |
|---------|---------|----------------|---------------------|
| ShareGPT | Community | Unclear (use for research) | No |
| MS-MARCO | MSR License | No (research/academic only) | Yes |
| HumanEval | MIT | Yes | No |
| Synthetic | N/A | Yes | No |

**Recommendation**: For academic research and course projects, all datasets are permissible. For commercial deployment, use HumanEval (MIT) or Synthetic only.

---

## Contact

For dataset issues or questions:
- **Researcher**: Rishi Nagaraj (rnagaraj@andrew.cmu.edu)
- **Course**: 11-868 LLM Systems, CMU Spring 2026
