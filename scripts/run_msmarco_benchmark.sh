#!/usr/bin/env bash
#SBATCH -A cis250224p
#SBATCH --job-name=vllm_marco
#SBATCH --output=vllm_marco.log
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
# SPDX-License-Identifier: Apache-2.0
#
# Run MS-MARCO benchmark with same memory pressure settings as ShareGPT.
#
# Usage:
#   bash scripts/run_msmarco_benchmark.sh
#
# Configuration via environment variables:
#   MODEL              Model to benchmark (default: meta-llama/Llama-3.2-1B-Instruct)
#   NUM_PROMPTS        Number of prompts to test (default: 200)
#   GPU_MEM_UTIL       GPU memory utilization fraction (default: 0.12 = 12%)
#   CPU_BYTES          CPU offload memory in bytes (default: 8GB)
#   MAX_TOKENS         Max generated tokens per request (default: 1024)
#   DATASET_PATH       Path to msmarco.json (default: ~/workspace/vllm/datasets/msmarco.json)
#   OUTPUT_DIR         Directory for results (default: ./benchmark_results)

set -euo pipefail  # Exit on error, undefined variable, or pipe failure

# Configuration
MODEL="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.12}"
CPU_BYTES="${CPU_BYTES:-8000000000}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
BLOCK_SIZE="${BLOCK_SIZE:-48}"
POLICIES="${POLICIES:-lru attention hybrid}"

# Paths
DATASET_PATH="${DATASET_PATH:-$HOME/workspace/vllm/datasets/msmarco.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./benchmark_results}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/results_msmarco_${TIMESTAMP}.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

# Print configuration
echo "========================================"
echo "MS-MARCO Benchmark Configuration"
echo "========================================"
echo "Model:             $MODEL"
echo "Dataset:           $DATASET_PATH"
echo "Num Prompts:       $NUM_PROMPTS"
echo "Max Tokens:        $MAX_TOKENS"
echo "GPU Memory:        ${GPU_MEM_UTIL} ($(echo "$GPU_MEM_UTIL * 100" | bc)%)"
echo "CPU Memory:        $(echo "$CPU_BYTES / 1024 / 1024 / 1024" | bc) GB"
echo "Block Size:        $BLOCK_SIZE"
echo "Eviction Policies: $POLICIES"
echo "Output:            $OUTPUT_FILE"
echo "========================================"
echo

# Validate dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo -e "${RED}ERROR: Dataset file not found: $DATASET_PATH${NC}"
    echo
    echo "Please download the MS-MARCO dataset first:"
    echo "  1. See kv_cache_tiering/benchmarks/DATASETS.md for instructions"
    echo "  2. Or run: python download_msmarco.py"
    echo "  3. Or set DATASET_PATH to the correct location"
    echo
    exit 1
fi

# Verify dataset is valid JSON and has expected format
echo -e "${YELLOW}Validating dataset...${NC}"
DATASET_SIZE=$(python3 -c "
import json
import sys
try:
    with open('$DATASET_PATH') as f:
        data = json.load(f)
    if not isinstance(data, list):
        print('ERROR: Dataset must be a JSON array', file=sys.stderr)
        sys.exit(1)
    if len(data) == 0:
        print('ERROR: Dataset is empty', file=sys.stderr)
        sys.exit(1)
    # Check first item has 'prompt' or 'query' field
    first_item = data[0]
    if 'prompt' not in first_item and 'query' not in first_item:
        print('WARNING: Dataset items should have prompt or query field', file=sys.stderr)
    print(len(data))
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1)

if [ $? -ne 0 ]; then
    echo -e "${RED}$DATASET_SIZE${NC}"
    exit 1
fi

echo -e "${GREEN}Dataset valid: $DATASET_SIZE queries${NC}"

# Warn if num_prompts exceeds dataset size
if [ "$NUM_PROMPTS" -gt "$DATASET_SIZE" ]; then
    echo -e "${YELLOW}WARNING: NUM_PROMPTS ($NUM_PROMPTS) exceeds dataset size ($DATASET_SIZE)${NC}"
    echo -e "${YELLOW}         Will use all $DATASET_SIZE queries${NC}"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run benchmark
echo
echo -e "${GREEN}Starting MS-MARCO benchmark...${NC}"
echo

python -m kv_cache_tiering.benchmarks.benchmark \
    --model "$MODEL" \
    --policies $POLICIES \
    --dataset msmarco \
    --dataset-path "$DATASET_PATH" \
    --num-prompts "$NUM_PROMPTS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-tokens "$MAX_TOKENS" \
    --gpu-mem-util "$GPU_MEM_UTIL" \
    --cpu-bytes "$CPU_BYTES" \
    --block-size "$BLOCK_SIZE" \
    --output "$OUTPUT_FILE"

# Check if benchmark succeeded
if [ $? -eq 0 ]; then
    echo
    echo -e "${GREEN}========================================"
    echo "Benchmark completed successfully!"
    echo "========================================${NC}"
    echo "Results saved to: $OUTPUT_FILE"
    echo

    # Show quick summary
    echo "Quick Summary:"
    python3 -c "
import json
with open('$OUTPUT_FILE') as f:
    results = json.load(f)
for r in results:
    policy = r['policy']
    throughput = r['tokens_per_second']
    latency = r['avg_latency_ms']
    print(f'  {policy:10s}: {throughput:6.1f} tok/s, {latency:6.1f} ms avg latency')
"
    echo
    echo "For detailed analysis, see: kv_cache_tiering/BENCHMARK_RESULTS.md"
else
    echo -e "${RED}ERROR: Benchmark failed${NC}"
    exit 1
fi
