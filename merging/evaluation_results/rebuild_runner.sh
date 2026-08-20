#!/bin/bash
# Self-contained eval runner, executed INSIDE the merge-tools container.
# Usage: rebuild_runner.sh lmeval <name> <task> <model-rel-path> [more triples...]
#        rebuild_runner.sh dsmath
# Results land in evaluation_results/ under the artifact naming convention;
# a console log per eval lands in evaluation_results/rebuild_logs/.
set -u
ROOT=/workspace/merge_tools
LOGDIR=$ROOT/evaluation_results/rebuild_logs
mkdir -p "$LOGDIR"

run_lmeval() {
  local name=$1 task=$2 rel=$3
  local log=$LOGDIR/$name.log tmp=$ROOT/evaluation_results/tmp_out_$name
  echo "=== $name : $task on $rel  $(date -Is)" > "$log"
  lm_eval --model hf --model_args "pretrained=$ROOT/$rel,dtype=bfloat16" \
    --tasks "$task" --batch_size auto --device cuda \
    --output_path "$tmp" >> "$log" 2>&1
  local rc=$?
  local ts res
  ts=$(date +%Y-%m-%dT%H-%M-%S.%6N)
  res=$(find "$tmp" -name 'results*.json' 2>/dev/null | head -1)
  if [[ $rc -eq 0 && -n "$res" ]]; then
    mv "$res" "$ROOT/evaluation_results/${task}_output_${ts}.json"
    rm -rf "$tmp"
    echo "[$name] OK -> ${task}_output_${ts}.json  $(date -Is)" >> "$log"
  else
    echo "[$name] FAILED rc=$rc  $(date -Is)" >> "$log"
  fi
}

run_dsmath() {
  # DS-Math merged (scaled build) via the vendored math harness. The harness
  # writes to a fixed path, so save the shipped baseline outputs first and
  # restore them afterwards; archive the new outputs under distinct names.
  local log=$LOGDIR/dsmath_merged.log
  local mh=$ROOT/vendor/math-evaluation-harness
  local base=$mh/output/gsm8k/test_cot_-1_seed0_t0.0_s0_e-1
  echo "=== dsmath_merged : gsm8k-cot on variants/fig6_scaled/deepseek-math-7b-instruct  $(date -Is)" > "$log"
  cp -p "${base}.jsonl" "${base}.jsonl.baseline_keep" 2>>"$log"
  cp -p "${base}_cot_metrics.json" "${base}_cot_metrics.json.baseline_keep" 2>>"$log"
  cd "$mh" || { echo "[dsmath_merged] FAILED: no harness dir" >> "$log"; return 1; }
  python math_eval.py \
    --model_name_or_path "$ROOT/variants/fig6_scaled/deepseek-math-7b-instruct" \
    --data_names gsm8k --prompt_type cot --use_vllm --temperature 0.0 \
    --save_outputs --overwrite --batch_size 32 --use_safetensors >> "$log" 2>&1
  local rc=$?
  local ts; ts=$(date +%Y-%m-%dT%H-%M-%S.%6N)
  local newm; newm=$(find output -name '*metrics.json' -newer "${base}_cot_metrics.json.baseline_keep" 2>/dev/null | grep -v baseline_keep | head -1)
  if [[ $rc -eq 0 && -n "$newm" ]]; then
    cp "$newm" "$ROOT/evaluation_results/gsm8k_cot_dsmath_merged_metrics_${ts}.json"
    local newj=${newm%_cot_metrics.json}.jsonl
    [[ -f $newj ]] && cp "$newj" "$ROOT/evaluation_results/gsm8k_cot_dsmath_merged_${ts}.jsonl"
    echo "[dsmath_merged] OK -> gsm8k_cot_dsmath_merged_metrics_${ts}.json" >> "$log"
  else
    echo "[dsmath_merged] FAILED rc=$rc" >> "$log"
  fi
  # restore the shipped baseline outputs
  mv -f "${base}.jsonl.baseline_keep" "${base}.jsonl" 2>>"$log"
  mv -f "${base}_cot_metrics.json.baseline_keep" "${base}_cot_metrics.json" 2>>"$log"
  echo "[dsmath_merged] baseline outputs restored  $(date -Is)" >> "$log"
}

run_replay_eval() {
  # Rebuild a fig5a c94 variant by replaying the recorded steps.csv, then
  # evaluate it. Usage: replay_eval <name> <target_label> <domain/task> <rel-out-dir>
  local name=$1 target=$2 task=$3 out=$4
  local log=$LOGDIR/${name}_replay.log
  local run=runs/qwen32b3_rerun rs=qwen32b3
  echo "=== ${name}_replay : $target -> cutoff 94 -> $out  $(date -Is)" > "$log"
  cd "$ROOT" || return 1
  python micr/run_eval.py \
    --ops_step_csvs_dir $run/clustering/$rs \
    --label_map_json    $run/label_map.json \
    --target_label      "$target" \
    --domain            "$task" \
    --replay_steps_csv  $run/micr/$rs/$target/steps.csv \
    --replay_cutoff 94 --replay_cutoff_mode step_idx \
    --scaling off --merge_device cuda \
    --save_variant_dir  "$out" --no_eval \
    --working_root /scratch_work/vwork_$name \
    --results_csv /tmp/vsteps_$name.csv --output_dir /tmp/veval_$name \
    --gpu_ids 0 >> "$log" 2>&1
  local rc=$?
  rm -rf "/scratch_work/vwork_$name"
  if [[ $rc -ne 0 || ! -f "$ROOT/$out/config.json" ]]; then
    echo "[${name}_replay] FAILED rc=$rc" >> "$log"; return 1
  fi
  echo "[${name}_replay] OK -> $out  $(date -Is)" >> "$log"
  run_lmeval "$name" "$task" "$out"
}

mode=$1; shift
if [[ $mode == dsmath ]]; then
  run_dsmath
elif [[ $mode == replay_eval ]]; then
  run_replay_eval "$@"
else
  while [[ $# -ge 3 ]]; do
    run_lmeval "$1" "$2" "$3"; shift 3
  done
fi
echo "RUNNER DONE $(date -Is)" >> "$LOGDIR/_gpu_done_$(date +%s).log"
