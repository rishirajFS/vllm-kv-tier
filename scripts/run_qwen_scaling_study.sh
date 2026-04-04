#!/usr/bin/env bash
#SBATCH --job-name=vllm_qwen_scaling
#SBATCH --output=vllm_qwen_scaling_%A.log
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=12:00:00
# SPDX-License-Identifier: Apache-2.0
#
# Run scaling study across multiple Qwen model sizes.
# Tests 1.5B, 3B, and 7B to show how improvements scale with model size.
#
# Usage:
#   sbatch scripts/run_qwen_scaling_study.sh
#   # OR locally:
#   bash scripts/run_qwen_scaling_study.sh
#
# Configuration via environment variables:
#   MODEL_SIZES        Space-separated model sizes (default: "1.5 3 7")
#   DATASET_DIR        Directory containing datasets
#   OUTPUT_DIR         Directory for results
#   SKIP_MEMORY_TEST   Set to "true" to skip memory efficiency tests

set -euo pipefail

# Environment Setup
cd ~/workspace/vllm
source .venv/bin/activate
export HF_HOME="/jet/home/rnagaraj/workspace/vllm/hf_cache"
export TRITON_CACHE_DIR="/jet/home/rnagaraj/workspace/vllm/triton_cache"
export XDG_CACHE_HOME="/jet/home/rnagaraj/workspace/vllm/xdg_cache"

# Configuration
MODEL_SIZES="${MODEL_SIZES:-1.5 3 7}"
DATASET_DIR="${DATASET_DIR:-$HOME/workspace/vllm/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-./benchmark_results}"
SKIP_MEMORY_TEST="${SKIP_MEMORY_TEST:-false}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print header
echo -e "${MAGENTA}"
echo "================================================================"
echo "  Qwen Model Scaling Study"
echo "  Testing Model Sizes: $MODEL_SIZES"
echo "================================================================"
echo -e "${NC}"

START_TIME=$(date +%s)
FAILED_MODELS=()

# Function to run benchmark suite for a single model size
run_model_benchmark() {
    local model_size=$1
    local model_name="Qwen/Qwen2.5-${model_size}B-Instruct"

    echo -e "\n${BLUE}================================================================${NC}"
    echo -e "${BLUE}  Model: $model_name${NC}"
    echo -e "${BLUE}================================================================${NC}\n"

    # Run throughput benchmarks
    echo -e "${GREEN}Running throughput benchmarks...${NC}"

    # ShareGPT
    echo -e "\n${YELLOW}[1/3] ShareGPT Benchmark${NC}"
    python -m kv_cache_tiering.benchmarks.benchmark \
        --model "$model_name" \
        --policies lru attention hybrid \
        --dataset sharegpt \
        --dataset-path "${DATASET_DIR}/sharegpt.json" \
        --num-prompts 200 \
        --max-tokens 1024 \
        --gpu-mem-util 0.12 \
        --cpu-bytes 8000000000 \
        --max-model-len 32768 \
        --output "${OUTPUT_DIR}/results_qwen${model_size}b_sharegpt_$(date +%Y%m%d_%H%M%S).json" \
        || { echo -e "${RED}✗ ShareGPT failed${NC}"; return 1; }

    # MS-MARCO
    echo -e "\n${YELLOW}[2/3] MS-MARCO Benchmark${NC}"
    python -m kv_cache_tiering.benchmarks.benchmark \
        --model "$model_name" \
        --policies lru attention hybrid \
        --dataset msmarco \
        --dataset-path "${DATASET_DIR}/msmarco.json" \
        --num-prompts 200 \
        --max-tokens 1024 \
        --gpu-mem-util 0.12 \
        --cpu-bytes 8000000000 \
        --max-model-len 32768 \
        --output "${OUTPUT_DIR}/results_qwen${model_size}b_msmarco_$(date +%Y%m%d_%H%M%S).json" \
        || { echo -e "${RED}✗ MS-MARCO failed${NC}"; return 1; }

    # HumanEval
    echo -e "\n${YELLOW}[3/3] HumanEval Benchmark${NC}"
    python -m kv_cache_tiering.benchmarks.benchmark \
        --model "$model_name" \
        --policies lru attention hybrid \
        --dataset humaneval \
        --dataset-path "${DATASET_DIR}/humaneval.json" \
        --num-prompts 164 \
        --max-tokens 512 \
        --gpu-mem-util 0.12 \
        --cpu-bytes 8000000000 \
        --max-model-len 32768 \
        --output "${OUTPUT_DIR}/results_qwen${model_size}b_humaneval_$(date +%Y%m%d_%H%M%S).json" \
        || { echo -e "${RED}✗ HumanEval failed${NC}"; return 1; }

    # Memory efficiency test (optional)
    if [ "$SKIP_MEMORY_TEST" != "true" ]; then
        echo -e "\n${YELLOW}Running memory efficiency test...${NC}"
        python scripts/benchmark_memory_efficiency.py \
            --model "$model_name" \
            --output "${OUTPUT_DIR}/memory_efficiency_qwen${model_size}b.json" \
            --gpu-memory-baseline 0.9 \
            --gpu-memory-tiered 0.12 \
            --cpu-tier-size 8000000000 \
            --start-length 2048 \
            --step-size 2048 \
            || echo -e "${YELLOW}⚠ Memory efficiency test failed (non-critical)${NC}"
    fi

    echo -e "${GREEN}✓ Model $model_name completed successfully${NC}"
    return 0
}

# Run benchmarks for each model size
for size in $MODEL_SIZES; do
    echo -e "\n${MAGENTA}Starting benchmark for Qwen ${size}B...${NC}"

    if run_model_benchmark "$size"; then
        echo -e "${GREEN}✓ Qwen ${size}B completed${NC}"
    else
        echo -e "${RED}✗ Qwen ${size}B failed${NC}"
        FAILED_MODELS+=("${size}B")
    fi
done

# Calculate total time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

# Final summary
echo -e "\n${MAGENTA}================================================================"
echo "  Scaling Study Complete"
echo "================================================================${NC}"
echo -e "Total Time: ${HOURS}h ${MINUTES}m"

if [ ${#FAILED_MODELS[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ All model sizes completed successfully!${NC}"
else
    echo -e "${YELLOW}⚠ ${#FAILED_MODELS[@]} model(s) failed: ${FAILED_MODELS[*]}${NC}"
fi

# Generate consolidated summary
echo -e "\n${BLUE}Generating consolidated summary...${NC}"
python scripts/generate_results_summary.py \
    --results-dir "$OUTPUT_DIR" \
    --output "${OUTPUT_DIR}/SCALING_SUMMARY.md" \
    && echo -e "${GREEN}✓ Summary saved to ${OUTPUT_DIR}/SCALING_SUMMARY.md${NC}"

# Show quick comparison
echo -e "\n${BLUE}Quick Scaling Comparison (ShareGPT - Attention Policy):${NC}"
echo "Model Size | Throughput  | Improvement over LRU"
echo "-----------|-------------|---------------------"

for size in $MODEL_SIZES; do
    result_file=$(ls "${OUTPUT_DIR}"/results_qwen${size}b_sharegpt_*.json 2>/dev/null | tail -1)
    if [ -f "$result_file" ]; then
        python3 -c "
import json
with open('$result_file') as f:
    results = json.load(f)
lru = next((r for r in results if r['policy'] == 'lru'), None)
attn = next((r for r in results if r['policy'] == 'attention'), None)
if lru and attn:
    improvement = ((attn['tokens_per_second'] - lru['tokens_per_second']) / lru['tokens_per_second']) * 100
    print(f'{size:>10}B | {attn[\"tokens_per_second\"]:>10.1f} | {improvement:>+8.1f}%')
"
    fi
done

echo -e "\n${GREEN}Next Steps:${NC}"
echo "1. Review scaling summary: cat ${OUTPUT_DIR}/SCALING_SUMMARY.md"
echo "2. Check memory efficiency: cat ${OUTPUT_DIR}/memory_efficiency_qwen*.json"
echo "3. Update BENCHMARK_RESULTS.md with scaling findings"
echo "4. Create scaling plots for paper (throughput vs model size)"

echo -e "\n${MAGENTA}================================================================${NC}\n"
