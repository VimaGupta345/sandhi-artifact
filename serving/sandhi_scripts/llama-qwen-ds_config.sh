#!/usr/bin/env bash

############################
# Server configuration
############################

CUDA_DEVICES="0,1"
TENSOR_PARALLEL_SIZE=2
GPU_ALLOC_GIB=12

declare -A MODELS=(
  [12301]="TsinghuaC3I/Llama-3-8B-UltraMedical"
  [12302]="HiTZ/Llama-3.1-8B-Instruct-multi-truth-judge"
  [12303]="K-intelligence/Llama-SafetyGuard-Content-Binary"
  [12304]="MaziyarPanahi/calme-2.3-legalkit-8b"
  [12305]="us4/fin-llama3.1-8b"
  [12306]="Qwen/Qwen2.5-Coder-7B-Instruct"
  [12307]="Qwen/Qwen2.5-Math-7B-Instruct"
  [12308]="deepseek-ai/deepseek-coder-7b-instruct-v1.5"
  [12309]="deepseek-ai/deepseek-math-7b-instruct"
)

############################
# Sharing configuration
############################

SHARED_SPEC="merged_spec_up_to_cutoff.json"

############################
# Benchmark configuration
############################

BENCH_TARGETS=(llama qwen ds)

BENCH_MODEL_llama="TsinghuaC3I/Llama-3-8B-UltraMedical"
REQUEST_RATES_llama=(3 5 7 10)
NUM_PROMPTS_llama=200
INPUT_LEN_llama=100
OUTPUT_LEN_llama=900

BENCH_MODEL_qwen="Qwen/Qwen2.5-Coder-7B-Instruct"
REQUEST_RATES_qwen=(7 10 15)
NUM_PROMPTS_qwen=250
INPUT_LEN_qwen=100
OUTPUT_LEN_qwen=900

BENCH_MODEL_ds="deepseek-ai/deepseek-coder-7b-instruct-v1.5"
REQUEST_RATES_ds=(1 2 3 5)
NUM_PROMPTS_ds=50
INPUT_LEN_ds=100
OUTPUT_LEN_ds=900

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
