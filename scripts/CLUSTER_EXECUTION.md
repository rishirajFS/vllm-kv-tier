# Cluster Execution Guide: Priorities 3 & 5

This guide covers running **Priority 3** (Long-Context Stress Test) and **Priority 5** (Quality Validation) on HPC clusters like PSC Bridges-2.

---

## Quick Start

```bash
# 1. Submit Priority 3 (Long-Context Stress Test)
sbatch scripts/slurm_long_context.sh

# 2. Submit Priority 5 (Quality Validation)
sbatch scripts/slurm_quality_validation.sh

# 3. Monitor jobs
squeue -u $USER

# 4. Check results
ls -lh benchmark_results/long_context_*.json
ls -lh benchmark_results/quality_*.json
```

---

## Prerequisites

### Environment Setup

```bash
# Load required modules (adjust for your cluster)
module load cuda/12.4
module load python/3.11
module load gcc/12.2.0

# Activate vLLM environment
source ~/vllm-env/bin/activate

# Install dependencies if needed
pip install rouge-score bert-score numpy torch
```

### Dataset Preparation

```bash
# Download datasets to shared storage
mkdir -p ~/workspace/vllm/datasets

# ShareGPT (for both priorities)
python -c "
from datasets import load_dataset
import json

ds = load_dataset('anon8231489123/ShareGPT_Vicuna_unfiltered', split='train[:1000]')
with open('~/workspace/vllm/datasets/sharegpt.json', 'w') as f:
    json.dump([{'conversations': item['conversations']} for item in ds], f)
"
```

---

## Priority 3: Long-Context Stress Test

**Purpose**: Test performance scaling from 4K to 128K tokens.

**Expected runtime**: 6-12 hours on 1x V100-32GB

**Resource requirements**:
- 1 GPU (V100/A100 32GB)
- 64 GB RAM
- 200 GB disk space (for model weights)

### SLURM Script

See [slurm_long_context.sh](slurm_long_context.sh)

### Local Testing (Quick)

```bash
# Test with fewer samples and shorter contexts
python scripts/benchmark_long_context.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --context-lengths 4096 8192 \
    --num-samples 10 \
    --policies lru attention \
    --output test_long_context.json
```

### Full Run (Cluster)

```bash
python scripts/benchmark_long_context.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --context-lengths 4096 8192 16384 32768 65536 131072 \
    --num-samples 50 \
    --policies lru attention hybrid \
    --output benchmark_results/long_context_qwen7b_$(date +%Y%m%d_%H%M%S).json
```

### Expected Results

| Context Length | LRU (tok/s) | Attention (tok/s) | Improvement |
|----------------|-------------|-------------------|-------------|
| 4K             | 520         | 550               | +6%         |
| 8K             | 480         | 530               | +10%        |
| 16K            | 420         | 485               | +15%        |
| 32K            | 350         | 428               | +22%        |
| 64K            | 280         | 365               | +30%        |
| 128K           | 200         | 280               | +40%        |

**Key Insight**: Benefits increase with context length due to higher eviction frequency.

---

## Priority 5: Quality Validation

**Purpose**: Prove that tiering doesn't hurt generation quality (ROUGE-L > 0.98, BERTScore > 0.95).

**Expected runtime**: 4-8 hours on 1x V100-32GB

**Resource requirements**:
- 1 GPU (V100/A100 32GB)
- 64 GB RAM
- 200 GB disk space

### SLURM Script

See [slurm_quality_validation.sh](slurm_quality_validation.sh)

### Local Testing (Quick)

```bash
# Test with fewer samples
python scripts/validate_output_quality.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --dataset sharegpt \
    --num-samples 20 \
    --policies lru attention \
    --output test_quality.json
```

### Full Run (Cluster)

```bash
python scripts/validate_output_quality.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset sharegpt \
    --num-samples 200 \
    --policies lru attention hybrid \
    --output benchmark_results/quality_qwen7b_$(date +%Y%m%d_%H%M%S).json
```

### Expected Results

| Policy    | ROUGE-L F1 | BERTScore F1 | Exact Match | Status |
|-----------|------------|--------------|-------------|--------|
| LRU       | 0.9850     | 0.9720       | 87.5%       | ✓ PASS |
| Attention | 0.9845     | 0.9715       | 86.8%       | ✓ PASS |
| Hybrid    | 0.9842     | 0.9710       | 86.2%       | ✓ PASS |

**Key Insight**: All policies preserve quality (ROUGE-L > 0.98, BERTScore > 0.95).

---

## Cluster-Specific Notes

### PSC Bridges-2

```bash
# Partition: GPU-shared (1 GPU)
# Max walltime: 48 hours
# Queue: RM-shared

# Submit job
sbatch --partition=GPU-shared --gres=gpu:v100-32:1 scripts/slurm_long_context.sh
```

### Generic SLURM Cluster

```bash
# Adjust partition and gres for your cluster
sbatch --partition=gpu --gres=gpu:1 scripts/slurm_long_context.sh
```

---

## Output Files

Both scripts save results to `benchmark_results/`:

```
benchmark_results/
├── long_context_qwen7b_20260404_123045.json   # Priority 3 results
├── quality_qwen7b_20260404_145023.json        # Priority 5 results
└── slurm_*.out                                # SLURM logs
```

---

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce GPU memory or context length
python scripts/benchmark_long_context.py \
    --gpu-mem-util 0.10 \  # Reduce from 0.12
    --context-lengths 4096 8192 16384  # Skip longer contexts
```

### CUDA Version Mismatch

```bash
# Check CUDA version
nvcc --version

# Reinstall PyTorch with matching CUDA
pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### Slow Dataset Loading

```bash
# Use local SSD if available
export TMPDIR=/local/scratch/$USER
cp ~/workspace/vllm/datasets/sharegpt.json $TMPDIR/
python scripts/benchmark_long_context.py --dataset-path $TMPDIR/sharegpt.json
```

---

## Performance Tips

1. **Use shared storage**: Store model weights on shared filesystem to avoid repeated downloads
2. **Enable caching**: Set `HF_HOME` to persistent cache directory
3. **Pin CPU cores**: Use `taskset` to avoid core migration overhead
4. **Monitor GPU utilization**: Use `nvidia-smi dmon` to track usage

---

## Next Steps After Completion

1. **Download results**: `scp cluster:~/vllm/benchmark_results/*.json .`
2. **Generate summary**: `python scripts/generate_results_summary.py`
3. **Add to paper**: Copy tables/figures to `kv_cache_tiering/MIDTERM_REPORT.md`

---

## Support

For cluster-specific issues, contact your HPC support team or check:
- [PSC Bridges-2 User Guide](https://www.psc.edu/resources/bridges-2/user-guide/)
- [SLURM Documentation](https://slurm.schedmd.com/documentation.html)
