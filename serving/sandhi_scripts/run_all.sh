#!/usr/bin/env bash
set -e

# source vllm_venv2/bin/activate

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

usage() {
    echo "Usage: $0 --config /path/to/config.sh --run-base-dir /path/to/run_base_dir"
}

CONFIG_FILE=""
RUN_BASE_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --config"
                usage
                exit 1
            fi
            CONFIG_FILE="$2"
            shift 2
            ;;
        --run-base-dir)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --run-base-dir"
                usage
                exit 1
            fi
            RUN_BASE_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$CONFIG_FILE" || -z "$RUN_BASE_DIR" ]]; then
    echo "Both --config and --run-base-dir are required."
    usage
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file not found: $CONFIG_FILE"
    exit 1
fi

export RUN_BASE_DIR
source "$CONFIG_FILE"
source "$SCRIPT_DIR/server_utils.sh"
source "$SCRIPT_DIR/benchmark.sh"

export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VMM_SOCKET_PATH="$RUN_BASE_DIR/vmm_sockets"

SHARED_HANDLES="$RUN_BASE_DIR/handles.jsonl"
GPU_ALLOC_PIDS=()

start_gpu_allocators() {
    local alloc_gib="${GPU_ALLOC_GIB:-}"
    local gpu=""

    if [[ -z "$alloc_gib" ]]; then
        return 0
    fi

    if ! [[ "$alloc_gib" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ ]]; then
        echo "GPU_ALLOC_GIB must be a non-negative number, got: $alloc_gib"
        exit 1
    fi

    if ! awk "BEGIN { exit !($alloc_gib > 0) }"; then
        return 0
    fi

    IFS=',' read -r -a gpu_list <<< "$CUDA_DEVICES"
    for gpu in "${gpu_list[@]}"; do
        gpu="${gpu//[[:space:]]/}"
        if [[ -z "$gpu" ]]; then
            continue
        fi

        CUDA_VISIBLE_DEVICES="$gpu" python3 "$SCRIPT_DIR/gpu_alloc.py" "$alloc_gib" &
        GPU_ALLOC_PIDS+=("$!")
        sleep 5s
    done
}

stop_gpu_allocators() {
    local pid=""

    for pid in "${GPU_ALLOC_PIDS[@]:-}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done

    GPU_ALLOC_PIDS=()
}

cleanup() {
    stop_gpu_allocators || true
    stop_servers || true
    rm -f "$SHARED_HANDLES"
    rm -rf "$VMM_SOCKET_PATH"
}

trap cleanup EXIT

rm -f "$SHARED_HANDLES"
rm -rf "$VMM_SOCKET_PATH"
mkdir -p "$VMM_SOCKET_PATH"

start_gpu_allocators

if [[ ${#BENCH_TARGETS[@]} -eq 0 ]]; then
    echo "BENCH_TARGETS must contain at least one benchmark target."
    exit 1
fi

run_benchmarks_for_all_targets() {
    local mode="$1"
    local target_name=""

    for target_name in "${BENCH_TARGETS[@]}"; do
        echo ""
        echo "--------------------------------------"
        echo "Benchmark target: $target_name"
        echo "--------------------------------------"
        run_benchmarks "$mode" "$target_name"
    done
}

####################################
# Baseline
####################################

start_servers baseline

run_benchmarks_for_all_targets baseline

stop_servers

####################################
# Sandhi
####################################

start_servers sandhi

run_benchmarks_for_all_targets sandhi

stop_servers

python3 "$SCRIPT_DIR/parse_and_plot_results.py" \
    --bench-log-dir "$BENCH_LOG_DIR" \
    --output-dir "$RESULTS_DIR"

echo ""
echo "======================================"
echo "Finished all experiments."
echo "======================================"
