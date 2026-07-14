#!/usr/bin/env bash

############################
# Server configuration
############################

CUDA_DEVICES="0"
TENSOR_PARALLEL_SIZE=1
GPU_ALLOC_GIB=100
MAX_NUM_SEQS="256"

declare -A MODELS=(
  [12301]="Qwen/Qwen2.5-Coder-7B-Instruct"
  [12302]="Qwen/Qwen2.5-Math-7B-Instruct"
)

############################
# Sharing configuration
############################

SHARED_SPEC="merged_spec_up_to_cutoff.json"

############################
# Benchmark configuration
############################

BENCH_TARGETS=(default)

BENCH_MODEL_default="Qwen/Qwen2.5-Coder-7B-Instruct"
REQUEST_RATES_default=(3 5 7 10)
NUM_PROMPTS_default=150
INPUT_LEN_default=100
OUTPUT_LEN_default=900

############################
# Output directories
############################

if [[ -z "${RUN_BASE_DIR:-}" ]]; then
    echo "RUN_BASE_DIR must be set before sourcing config.sh"
    return 1 2>/dev/null || exit 1
fi

SERVER_LOG_DIR="$RUN_BASE_DIR/logs/servers"
BENCH_LOG_DIR="$RUN_BASE_DIR/logs/benchmarks"
RESULTS_DIR="$RUN_BASE_DIR/results"
PLOT_DIR="$RESULTS_DIR/plots"

mkdir -p "$SERVER_LOG_DIR"
mkdir -p "$BENCH_LOG_DIR"
mkdir -p "$PLOT_DIR"
