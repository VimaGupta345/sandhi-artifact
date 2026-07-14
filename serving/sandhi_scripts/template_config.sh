#!/usr/bin/env bash

############################
# Server configuration
############################

CUDA_DEVICES="5,6"
TENSOR_PARALLEL_SIZE=2
GPU_ALLOC_GIB=0
MAX_NUM_SEQS="128"

declare -A MODELS=(
  [12301]="TsinghuaC3I/Llama-3-8B-UltraMedical"
  [12302]="HiTZ/Llama-3.1-8B-Instruct-multi-truth-judge"
  [12303]="K-intelligence/Llama-SafetyGuard-Content-Binary"
  [12304]="MaziyarPanahi/calme-2.3-legalkit-8b"
  [12305]="us4/fin-llama3.1-8b"
  [12306]="Qwen/Qwen2.5-Coder-7B-Instruct"
  [12307]="Qwen/Qwen2.5-Math-7B-Instruct"
)

############################
# Sharing configuration
############################

SHARED_SPEC="/nethome/nmeda6/vllm/merged_spec_up_to_cutoff.json"

############################
# Benchmark configuration
############################

BENCH_TARGETS=(default)

BENCH_MODEL_default="meta-llama/Llama-3.1-8B"
REQUEST_RATES_default=(25 75)
NUM_PROMPTS_default=750
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
