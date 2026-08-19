#!/usr/bin/env bash
# Figures 8 & 9 — CPU-offloading experiments (reconstructed launcher; the
# recorded runs in results_*/ were produced with exactly these server/bench
# parameters, extracted from their logs).
#
# Fig 9 (2x Llama-3.1-8B, 1xA100-40GB budget): per-model KV budget 2 GB.
#   baseline : weights don't fit alongside KV -> --cpu-offload-gb 2
#   SANDHI   : dedup frees enough weight memory -> no offload
# Fig 8 (3x Qwen3-32B, 2xA100-80GB budget): per-model KV budget 32 GB.
#   baseline : --cpu-offload-gb 14
#   SANDHI   : reduced offload volume -> --cpu-offload-gb 4.39
#
# Usage: run_offload_experiment.sh <model_path> <kv_bytes> <offload_gb|0> <outdir> <rates...>
#   e.g. run_offload_experiment.sh /models/fin-llama3.1-8b 2000000000 2.0 results_llama_kv2gb_offload2gb 2 5 7 10 15 20
#        run_offload_experiment.sh /models/Qwen3-32B 32000000000 0 results_qwen3_kv32gb_no-offload 1 2 5 10
set -e
MODEL="$1"; KV_BYTES="$2"; OFFLOAD_GB="$3"; OUT="$4"; shift 4
RATES=("$@")
PORT=8008
mkdir -p "$OUT"

offload_flags=()
if [[ "$OFFLOAD_GB" != "0" ]]; then
    offload_flags=(--cpu-offload-gb "$OFFLOAD_GB")
fi

vllm serve "$MODEL" \
    --port "$PORT" \
    --max-model-len 4096 \
    --kv-cache-memory-bytes "$KV_BYTES" \
    --max-num-seqs 250 \
    "${offload_flags[@]}" \
    > "$OUT/server_${PORT}.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

for i in $(seq 1 600); do
    curl -sf "http://localhost:$PORT/health" >/dev/null && break
    sleep 1
done

for RPS in "${RATES[@]}"; do
    vllm bench serve \
        --model "$MODEL" \
        --port "$PORT" \
        --num-prompts 100 \
        --random-input-len 100 \
        --random-output-len 900 \
        --request-rate "$RPS" \
        --ignore-eos \
        --metric-percentiles 95,99 \
        >> "$OUT/bench_${PORT}.log" 2>&1
done
