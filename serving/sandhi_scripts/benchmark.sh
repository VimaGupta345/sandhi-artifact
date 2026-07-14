#!/usr/bin/env bash

slugify() {
    local value="$1"
    value="${value,,}"
    value="${value//[^a-z0-9]/_}"
    value="$(printf '%s' "$value" | sed -E 's/_+/_/g; s/^_+//; s/_+$//')"
    printf '%s\n' "$value"
}

run_benchmarks() {

    local mode="$1"
    local target_name="$2"
    local bench_port=""
    local bench_model_var="BENCH_MODEL_${target_name}"
    local num_prompts_var="NUM_PROMPTS_${target_name}"
    local input_len_var="INPUT_LEN_${target_name}"
    local output_len_var="OUTPUT_LEN_${target_name}"
    local bench_model="${!bench_model_var:-}"
    local num_prompts="${!num_prompts_var:-}"
    local input_len="${!input_len_var:-}"
    local output_len="${!output_len_var:-}"
    local request_rates_var="REQUEST_RATES_${target_name}"
    local -n request_rates_ref="$request_rates_var"
    local model_slug=""

    if [[ -z "$target_name" ]]; then
        echo "run_benchmarks requires a benchmark target name."
        return 1
    fi

    if [[ -z "$bench_model" ]]; then
        echo "BENCH_MODEL_${target_name} is not set."
        return 1
    fi

    if [[ -z "$num_prompts" || -z "$input_len" || -z "$output_len" ]]; then
        echo "Benchmark settings for ${target_name} are incomplete."
        return 1
    fi

    if [[ ${#request_rates_ref[@]} -eq 0 ]]; then
        echo "REQUEST_RATES_${target_name} is empty."
        return 1
    fi

    model_slug="$(slugify "$bench_model")"
    LOGFILE="$BENCH_LOG_DIR/${mode}__${model_slug}.log"

    echo "" > "$LOGFILE"

    echo "==================================" | tee -a "$LOGFILE"
    echo "Running benchmarks ($mode / $target_name)" | tee -a "$LOGFILE"
    echo "==================================" | tee -a "$LOGFILE"

    for PORT in $(printf "%s\n" "${!MODELS[@]}" | sort -n); do
        if [[ "${MODELS[$PORT]}" == "$bench_model" ]]; then
            bench_port="$PORT"
            break
        fi
    done

    if [[ -z "$bench_port" ]]; then
        echo "Benchmark model $bench_model not found in MODELS." | tee -a "$LOGFILE"
        return 1
    fi

    echo "" | tee -a "$LOGFILE"
    echo "==================================" | tee -a "$LOGFILE"
    echo "Target: $target_name" | tee -a "$LOGFILE"
    echo "Port: $bench_port" | tee -a "$LOGFILE"
    echo "Model: $bench_model" | tee -a "$LOGFILE"
    echo "==================================" | tee -a "$LOGFILE"

    for RPS in "${request_rates_ref[@]}"; do

        echo "RPS=$RPS" | tee -a "$LOGFILE"

        CUDA_VISIBLE_DEVICES="${CUDA_DEVICES%%,*}" \
        vllm bench serve \
            --model "$bench_model" \
            --port "$bench_port" \
            --num-prompts "$num_prompts" \
            --metric-percentiles 95,99 \
            --random-input-len "$input_len" \
            --random-output-len "$output_len" \
            --request-rate "$RPS" \
            --ignore-eos \
            >> "$LOGFILE" 2>&1

    done
}
