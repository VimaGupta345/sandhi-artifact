## `micr/` — merge/apply ops + evaluate (unified runner)

This folder contains **single-target “apply merge ops then evaluate”** runners.

You almost always want to run:

- `micr/run_eval_unified.py` — **one CLI**, choose implementation with `--mode`

Underlying implementations (still usable directly):

- `micr/run_eval.py` — **normal** full-weight merges (7B/8B-ish)
- `micr/run_eval_32b.py` — **large-model** entry point (thin shim: same CLI, delegates to `run_eval.py`)
- `micr/run_eval_lora.py` — **LoRA adapter** merges (averages LoRA A/B matrices)

---

## Prereqs / environment

- **Python deps**: `torch`, `transformers`, `pandas`
- **LoRA mode additionally**: `peft`
- **Evaluation harness**: self-contained in `micr/eval_harness.py`. No external checkout and no
  separate conda environment: evaluations run as subprocesses of the current interpreter.
  - `humaneval` and the `mmlu_*`/`truthfulqa`/`sst2`/`ifeval`/`sciq` tasks go through `lm_eval`.
  - `gsm8k-cot`/`gsm8k-pal` go through the vendored `vendor/math-evaluation-harness`, which is
    resolved relative to this repo. Override only if you moved it:

```bash
export MICR_MATH_HARNESS_DIR="/path/to/math-evaluation-harness"
```

---

## Inputs

### 1) Ops CSV format (`--ops_csv` or `--ops_step_csvs_dir`)

You must provide **exactly one**:

- `--ops_csv`: a single “legacy” CSV file
- `--ops_step_csvs_dir`: a directory containing per-target files:
  - `ops_step1_<target_label_sanitized>.csv`
  - `ops_step2_<target_label_sanitized>.csv` (optional)

The runners expect the ops CSV schema:

- **Required columns**: `op, component, layer, models`
- **Typical**:
  - `op`: usually `merge` (and sometimes `replace` depending on your generator)
  - `component`: one of:
    - attention subcomponents: `attn_q`, `attn_k`, `attn_v`, `attn_o`
    - MLP subcomponents: `mlp_gate`, `mlp_up`, `mlp_down`
  - `layer`: integer layer index (0-based)
  - `models`: comma-separated participants, usually quoted because of commas

**Participant encoding** (in the `models` column):

- format: `label:layer,label:layer,...`
- example row:

```csv
op,component,layer,models
merge,mlp_gate,3,"deepseek-coder-7b-instruct-v1.5:3,deepseek-math-7b-instruct:3"
```

Notes:

- If a participant omits `:layer`, the scripts treat it as “use the op’s `layer`”.
- The scripts only apply an op when **`target_label` is one of the participants**.

### 2) Label map (`--label_map_json`) for `normal` / `32b`

For `--mode normal` or `--mode 32b`, you must supply:

- `--label_map_json`: JSON mapping from label → **local HF model folder**

Example:

```json
{
  "deepseek-coder-7b-instruct-v1.5": "/path/to/models/deepseek-coder-7b-instruct-v1.5",
  "deepseek-math-7b-instruct": "/path/to/models/deepseek-math-7b-instruct"
}
```

### 3) LoRA paths (LoRA mode)

`--mode lora` does **not** use `--label_map_json`.
Instead, the LoRA script uses these locations (both env-overridable):

- base model: `BASE_MODEL_PATH` — env `MICR_LORA_BASE_MODEL`, default
  `meta-llama/Llama-3.1-8B`. **Note:** this HF repo is gated — accept the Meta
  Llama 3.1 license on Hugging Face and authenticate (`HF_TOKEN` /
  `huggingface-cli login`) before running, or point the env var at a local copy.
- adapter root: `ADAPTER_ROOT` — env `MICR_LORA_ADAPTER_ROOT`, default is a
  cluster-internal path. Download the per-domain adapters from
  `anjohn0077/NEXS-lora-adapters` (see `BASELINES.md`) into a local directory
  laid out per `LABEL_TO_ADAPTER_DIR`, and set
  `MICR_LORA_ADAPTER_ROOT=/path/to/adapters`.
- label → adapter dir mapping: `LABEL_TO_ADAPTER_DIR` (see `run_eval_lora.py`)

---

## Recommended usage: `run_eval_unified.py`

Run from the repo root:

```bash
python micr/run_eval_unified.py -h
```

### Modes

- **`--mode normal`**
  - Full-weight merges using `micr/run_eval.py`
  - Best for “regular size” models where loading donors is feasible
- **`--mode 32b`** (synonyms: `large`, `streaming`)
  - Full-weight merges using `micr/run_eval_32b.py` (a thin shim that delegates to `run_eval.py`; donors load via low_cpu_mem_usage mmap)
  - vLLM `tensor_parallel_size` is derived from the number of GPUs in `--gpu_ids` (in `micr/eval_harness.py`)
- **`--mode lora`**
  - Adapter merges using `micr/run_eval_lora.py`
  - Averages LoRA A/B matrices for the target adapter, then merges adapter → base model for eval

---

## CLI arguments (unified runner)

The unified runner exposes a **union** of args. Some only apply to certain modes.

### Required

- **`--mode`**: `normal` | `32b` | `lora`
- **Exactly one of**:
  - **`--ops_csv`**
  - **`--ops_step_csvs_dir`**
- **`--target_label`**: label of the model (or adapter label) to modify
- **`--domain`**: evaluation domain or registry task key (see “Task mapping” below)

### Required for `normal` / `32b`

- **`--label_map_json`**: label → model path JSON

### Evaluation / resource controls

- **`--gpu_ids`**: sets `CUDA_VISIBLE_DEVICES` for evaluation (default: `0`)
  - `32b` mode: the number of IDs also drives vLLM `tensor_parallel_size`
- **`--no_eval`**: disable evaluation; steps are accepted without scoring
- **`--output_dir`**: evaluation outputs directory (default: `./evaluation_results`)
- **`--timeout_minutes`**: per-eval timeout (default: 15)
- **`--batch_size`**:
  - default `64` for `normal/32b`
  - default `32` for `lora`
- **`--temperature`**: decoding temperature for evaluators (default: `0.0`)
- **`--drop_tolerance`**: reject step if score drops by more than this (absolute % points; default: `2.0`)

### Working directories / logs

- **`--working_root`**
  - `normal/32b` default: `/tmp/micr_merged_models (override with MICR_WORKING_ROOT)`
  - `lora` default: `./lora_work`
- **`--results_csv`**
  - `normal/32b` default: `./ops_step_csvs/target_steps.csv`
  - `lora` default: `lora_results.csv`
- **`--sorted_ops_out`**: legacy compatibility (some pipelines no longer re-export sorted ops)

### Scheduling / donor filtering (`normal` / `32b`)

- **`--sort_mode`**: `normal` | `separate` | `together`
- **`--ignore-other-families`**: filter donors to the same “family” as the target

### Only implemented in `normal` (`micr/run_eval.py`)

- **`--initial_baseline`**: force baseline score (skip lookup/eval)
- **`--force-calc-baseline`**: compute baseline even if a hardcoded one exists
- **`--enable-scaling`**: attempt per-component std scaling (implementation-dependent)

---

## Task mapping (`--domain`)

The scripts map some friendly domains to `eval_harness.TASK_REGISTRY` task keys.
Examples (not exhaustive; see `DOMAIN_TO_REGISTRY_TASK` in each runner):

- `finance` → `mmlu_econometrics`
- `legal` → `mmlu_professional_law`
- `medical` → `medqa_4options`
- `truthfulness` → `truthfulqa_mc2`
- `toxicity` → `sst2`
- `math` → `gsm8k-cot`
- `coder` / `code` → `humaneval`

You can also pass a registry key directly if it exists in the harness `TASK_REGISTRY`.

---

## Usage examples

### Normal weights (per-target step CSVs)

```bash
python micr/run_eval_unified.py --mode normal \
  --ops_step_csvs_dir clustering_algorithm/ops_step_csvs/solo_ds \
  --label_map_json clustering_algorithm/label_map.json \
  --target_label deepseek-coder-7b-instruct-v1.5 \
  --domain coder \
  --gpu_ids 0 \
  --drop_tolerance 2.0 \
  --results_csv output/micr/deepseek_coder_steps.csv \
  --output_dir output/micr/eval_runs
```

### 32B / large-model weights

```bash
python micr/run_eval_unified.py --mode 32b \
  --ops_step_csvs_dir clustering_algorithm/ops_step_csvs \
  --label_map_json clustering_algorithm/label_map.json \
  --target_label Light-IF-32B \
  --domain ifeval \
  --gpu_ids 0,1 \
  --drop_tolerance 1.0
```

### LoRA adapter merging

```bash
python micr/run_eval_unified.py --mode lora \
  --ops_step_csvs_dir clustering_algorithm/ops_step_csvs \
  --target_label fin-llama3.1-8b \
  --domain finance \
  --gpu_ids 0 \
  --working_root ./lora_work \
  --results_csv ./lora_work/finance_steps.csv \
  --output_dir ./lora_work/eval_outputs
```

---

## Outputs / what to expect

### 1) Working model directory

#### `normal` / `32b`

- The runner creates a working copy at:
  - `--working_root/<target_model_folder_name>/`
- Temporary candidates are saved under:
  - `--working_root/tmp_eval_single/<random_tmp_dir>/`
- If a step is **accepted**, the candidate temp dir is moved into the working directory (replacing it).

#### `lora`

- The base model + adapters are loaded (usually on CPU).
- Each evaluation creates a temporary full model folder under:
  - `--working_root/tmp_eval_lora/<random_tmp_dir>/`
  - This temp dir is removed after the evaluation step.

### 2) Per-step results CSV (`--results_csv`)

All three pipelines append step outcomes to a CSV with columns:

- `timestamp`: UTC ISO-8601
- `step_idx`: sequential step counter (0-based)
- `stage`: stage id (1 or 2 for step CSVs; may be absent in legacy inputs)
- `op`: typically `merge`
- `component`: `attn_*` or `mlp_*` (or grouped name depending on pipeline)
- `layer`: integer layer index
- `label`: your `--target_label`
- `score`: evaluated score (percent; numeric)
- `threshold`: drop tolerance used for accept/reject
- `decision`: `accepted` | `rejected`

Notes:

- In the LoRA pipeline, older rows may have an empty `threshold` field (it depends on the row dict written).

### 3) Evaluation artifacts directory (`--output_dir`)

Each evaluation run writes artifacts under subfolders such as:

- `--output_dir/<target_label>/baseline/`
- `--output_dir/<target_label>/step_<N>/`
- `--output_dir/<target_label>/final/`

The exact files are produced by the evaluation harness (task-dependent).

---

## Troubleshooting notes

- **vLLM fails / engine init fails**: these runners generally try vLLM first and fall back to a non-vLLM path when possible.
- **32B save failures**: `run_eval.py` sanitizes `generation_config` to satisfy newer `transformers` validation during `save_pretrained()` (ported from the former `run_eval_32b.py`).
- **LoRA OOM / fragmentation**: LoRA eval context sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and limits vLLM memory utilization.

