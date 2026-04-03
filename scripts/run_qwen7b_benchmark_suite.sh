#!/usr/bin/env bash
#SBATCH --job-name=vllm_qwen7b
#SBATCH --output=vllm_qwen7b_suite.log
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=04:00:00
# SPDX-License-Identifier: Apache-2.0
#
# Run complete benchmark suite for Qwen 2.5 7B across all workloads.
# This script runs ShareGPT, MS-MARCO, and HumanEval benchmarks sequentially.
#
# Usage:
#   sbatch scripts/run_qwen7b_benchmark_suite.sh
#   # OR locally:
#   bash scripts/run_qwen7b_benchmark_suite.sh
#
# Configuration via environment variables:
#   DATASET_DIR        Directory containing datasets (default: ~/workspace/vllm/datasets)
#   OUTPUT_DIR         Directory for results (default: ./benchmark_results)
#   GPU_MEM_UTIL       GPU memory utilization (default: 0.12 = 12%)
#   CPU_BYTES          CPU offload memory (default: 8GB)

set -euo pipefail

# Environment Setup
cd ~/workspace/vllm
source .venv/bin/activate

# Configuration
MODEL="Qwen/Qwen2.5-7B-Instruct"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.12}"
CPU_BYTES="${CPU_BYTES:-8000000000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
BLOCK_SIZE="${BLOCK_SIZE:-48}"
POLICIES="${POLICIES:-lru attention hybrid}"

# Paths
DATASET_DIR="${DATASET_DIR:-$HOME/workspace/vllm/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-./benchmark_results}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print header
echo -e "${BLUE}========================================"
echo "Qwen 2.5 7B Benchmark Suite"
echo "========================================"
echo "Model:        $MODEL"
echo "GPU Memory:   ${GPU_MEM_UTIL} ($(echo "$GPU_MEM_UTIL * 100" | bc)%)"
echo "CPU Memory:   $(echo "$CPU_BYTES / 1024 / 1024 / 1024" | bc) GB"
echo "Max Context:  $MAX_MODEL_LEN tokens"
echo "Policies:     $POLICIES"
echo "Timestamp:    $TIMESTAMP"
echo -e "========================================${NC}"
echo

# Function to run benchmark for a single dataset
run_benchmark() {
    local dataset=$1
    local dataset_path=$2
    local num_prompts=$3
    local max_tokens=$4
    local output_file="${OUTPUT_DIR}/results_qwen7b_${dataset}_${TIMESTAMP}.json"

    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Running ${dataset^^} Benchmark${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo "Dataset:      $dataset"
    echo "Path:         $dataset_path"
    echo "Prompts:      $num_prompts"
    echo "Max Tokens:   $max_tokens"
    echo "Output:       $output_file"
    echo

    # Validate dataset exists
    if [ ! -f "$dataset_path" ]; then
        echo -e "${RED}ERROR: Dataset not found: $dataset_path${NC}"
        echo "Skipping $dataset benchmark..."
        echo
        return 1
    fi

    # Run benchmark
    python -m kv_cache_tiering.benchmarks.benchmark \
        --model "$MODEL" \
        --policies $POLICIES \
        --dataset "$dataset" \
        --dataset-path "$dataset_path" \
        --num-prompts "$num_prompts" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-tokens "$max_tokens" \
        --gpu-mem-util "$GPU_MEM_UTIL" \
        --cpu-bytes "$CPU_BYTES" \
        --block-size "$BLOCK_SIZE" \
        --output "$output_file"

    if [ $? -eq 0 ]; then
        echo
        echo -e "${GREEN}✓ ${dataset^^} benchmark completed successfully${NC}"
        echo "Results: $output_file"
        echo

        # Show quick summary
        echo "Quick Results:"
        python3 -c "
import json
with open('$output_file') as f:
    results = json.load(f)
for r in results:
    policy = r['policy']
    throughput = r['tokens_per_second']
    latency = r['avg_latency_ms']
    ttft = r.get('avg_ttft_ms', 0)
    print(f'  {policy:10s}: {throughput:7.1f} tok/s | Latency: {latency:6.1f} ms | TTFT: {ttft:6.1f} ms')
"
        echo
    else
        echo -e "${RED}✗ ${dataset^^} benchmark failed${NC}"
        return 1
    fi
}

# Track overall success
FAILED_BENCHMARKS=()

# Benchmark 1: ShareGPT (Conversational)
echo -e "${BLUE}[1/3] ShareGPT Benchmark (Conversational)${NC}"
if ! run_benchmark \
    "sharegpt" \
    "${DATASET_DIR}/sharegpt.json" \
    200 \
    1024; then
    FAILED_BENCHMARKS+=("ShareGPT")
fi

# Benchmark 2: MS-MARCO (RAG)
echo -e "${BLUE}[2/3] MS-MARCO Benchmark (RAG)${NC}"
if ! run_benchmark \
    "msmarco" \
    "${DATASET_DIR}/msmarco.json" \
    200 \
    1024; then
    FAILED_BENCHMARKS+=("MS-MARCO")
fi

# Benchmark 3: HumanEval (Code Completion)
echo -e "${BLUE}[3/3] HumanEval Benchmark (Code Completion)${NC}"
if ! run_benchmark \
    "humaneval" \
    "${DATASET_DIR}/humaneval.json" \
    164 \
    512; then
    FAILED_BENCHMARKS+=("HumanEval")
fi

# Final summary
echo
echo -e "${BLUE}========================================"
echo "Benchmark Suite Complete"
echo -e "========================================${NC}"

if [ ${#FAILED_BENCHMARKS[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ All benchmarks completed successfully!${NC}"
    echo
    echo "Results saved to:"
    ls -lh "${OUTPUT_DIR}/results_qwen7b_"*"_${TIMESTAMP}.json"
    echo
    echo "Generate consolidated summary:"
    echo "  python scripts/generate_results_summary.py --results-dir $OUTPUT_DIR"
else
    echo -e "${YELLOW}⚠ ${#FAILED_BENCHMARKS[@]} benchmark(s) failed:${NC}"
    for benchmark in "${FAILED_BENCHMARKS[@]}"; do
        echo "  - $benchmark"
    done
    exit 1
fi

echo
echo -e "${GREEN}Next steps:${NC}"
echo "1. Review results: cat ${OUTPUT_DIR}/results_qwen7b_*_${TIMESTAMP}.json | jq"
echo "2. Generate summary: python scripts/generate_results_summary.py --results-dir $OUTPUT_DIR --output $OUTPUT_DIR/SUMMARY.md"
echo "3. Compare with LLaMA: Compare results_qwen7b_* vs results_sharegpt_*/results_msmarco_*/results_humaneval_*"
