#!/usr/bin/env bash

PIDS=()

start_servers() {

    local mode="$1"
    local -a max_num_seqs_flags=()

    mkdir -p "$SERVER_LOG_DIR/$mode"

    local sharing_flags=""

    if [[ "$mode" == "sandhi" ]]; then
        # Fail fast on a missing or mismatched spec: SHARED_SPEC must exist and
        # contain every model in this config's pool. Configs use distinct spec
        # filenames per scenario, but a copy/paste mistake here would otherwise
        # produce a silently under-shared "sandhi" run.
        if [[ ! -f "$SHARED_SPEC" ]]; then
            echo "SHARED_SPEC not found: $SHARED_SPEC (run from the directory holding the spec, or set an absolute path in the config)"
            exit 1
        fi
        if ! python3 - "$SHARED_SPEC" "${MODELS[@]}" <<'PYEOF'
import json, sys
spec_path, models = sys.argv[1], sys.argv[2:]
spec_models = {e["model"] for g in json.load(open(spec_path)) for e in g}
missing = [m for m in models if m.split("/")[-1] not in spec_models]
if missing:
    print(f"SHARED_SPEC {spec_path} covers {sorted(spec_models)} but is missing: {missing}")
    sys.exit(1)
PYEOF
        then
            echo "Spec/pool mismatch — refusing to start sandhi-mode servers."
            exit 1
        fi
        sharing_flags="\
            --shared-layers-ptrs-path $SHARED_HANDLES \
            --shared-layers-spec-path $SHARED_SPEC"
    fi

    if [[ -n "${MAX_NUM_SEQS:-}" ]]; then
        max_num_seqs_flags=(--max-num-seqs "$MAX_NUM_SEQS")
    fi

    PIDS=()

    echo "======================================"
    echo "Starting servers ($mode)"
    echo "======================================"

    for PORT in $(printf "%s\n" "${!MODELS[@]}" | sort -n); do

        MODEL="${MODELS[$PORT]}"
        local ipc_name="kvcached_instance_${PORT}"

        echo "Launching $MODEL on port $PORT"
        echo "  KVCACHED_IPC_NAME=$ipc_name"

        CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
        KVCACHED_IPC_NAME="$ipc_name" \
        vllm serve "$MODEL" \
            --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
            --port "$PORT" \
            --no-enable-prefix-caching \
            --disable-log-requests \
            "${max_num_seqs_flags[@]}" \
            $sharing_flags \
            > "$SERVER_LOG_DIR/$mode/server_${PORT}.log" 2>&1 &

        PID=$!
        PIDS+=("$PID")

        echo "  pid=$PID"

        READY=false

        for i in {1..180}; do
            if curl -sf "http://localhost:$PORT/health" >/dev/null; then
                READY=true
                echo "  Port $PORT ready."
                break
            fi
            sleep 1
        done

        if [[ "$READY" != true ]]; then
            echo "Server on port $PORT failed to start."

            stop_servers

            exit 1
        fi

    done

    echo "All servers started."
}

stop_servers() {
    local PID=""

    echo "Stopping servers..."

    for PID in "${PIDS[@]}"; do
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
        fi
    done

    for PID in "${PIDS[@]}"; do
        wait "$PID" 2>/dev/null || true
    done

    PIDS=()

    echo "All servers stopped."
}
