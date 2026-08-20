#!/usr/bin/env bash

############################
# Server configuration
############################

CUDA_DEVICES="0"
TENSOR_PARALLEL_SIZE=1
GPU_ALLOC_GIB=40

declare -A MODELS=(
  [12301]="TsinghuaC3I/Llama-3.1-8B-UltraMedical"
  [12302]="HiTZ/Llama-3.1-8B-Instruct-multi-truth-judge"
  [12303]="K-intelligence/Llama-SafetyGuard-Content-Binary"
  [12304]="MaziyarPanahi/calme-2.3-legalkit-8b"
  [12305]="mukaj/Llama-3.1-Hawkish-8B"
)

############################
# Sharing configuration
############################

SHARED_SPEC="llama5_spec.json"

############################
# Benchmark configuration
############################

BENCH_TARGETS=(default)

BENCH_MODEL_default="TsinghuaC3I/Llama-3.1-8B-UltraMedical"
REQUEST_RATES_default=(20 25 50 75)
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

# Optional: serve the MATERIALIZED merged variants in the sandhi arm (exact
# weights validated by Figure 5). Build them with the merging pipeline's
# replay (merging/GENERATE_VARIANTS.md, Cpm cutoff 52 for this pool), mount
# them, and uncomment:
# declare -A MODELS_SANDHI=(
#   [12301]="/variants/Llama-3.1-8B-UltraMedical"
#   [12302]="/variants/Llama-3.1-8B-Instruct-multi-truth-judge"
#   [12303]="/variants/Llama-SafetyGuard-Content-Binary"
#   [12304]="/variants/calme-2.3-legalkit-8b"
#   [12305]="/variants/Llama-3.1-Hawkish-8B"
# )
