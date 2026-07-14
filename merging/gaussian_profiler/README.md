# Gaussian Profiler for Model Merging

This directory contains `gaussian_profiler.py`, a tool designed to profile Large Language Models (LLMs) by applying layer-wise Gaussian noise perturbations. By measuring the performance impact of noise on specific layers and components (Attention vs. MLP), this tool helps identify which parts of the model are sensitive or robust.

## Prerequisites

- **Evaluation harness**: in-repo at `micr/eval_harness.py`; no external checkout or env var needed.
- **Models**: Models should be available locally.
- **GPUs**: Evaluation requires GPU resources. Set `CUDA_VISIBLE_DEVICES` or use the `--gpus` argument.

## Script: `gaussian_profiler.py`

> **Note:** This script was formerly named `gaussian_experiment.py`.

This is the main driver for the Gaussian noise sensitivity profiling. It performs a sweep across model layers, applying noise to specific parameter groups and evaluating the resulting model on downstream tasks.

### Features

- **Layer-wise Sweep**: Iterates through specified layers.
- **Component Targeting**: Target `attn` or `mlp` sub-modules.
- **Perturbation Types**: `avg` (average noise) and `replace` (replace with noise).
- **Automated Evaluation**: Uses `micr/eval_harness.py`.
- **CSV Logging**: Records detailed results for every step.

### Usage

The script now accepts all parameters via command-line arguments (no config file required).

```bash
python gaussian_profiler.py \
  --model <path_to_model> \
  [--tasks <task1,task2>] \
  [--output_csv <path_to_csv>] \
  [--gpus <gpu_ids>] \
  [--tmp_dir <tmp_dir>] \
  [--start_layer <int>] \
  [--end_layer <int>] \
  [--perturbation <avg,replace,add>] \
  [--debug] \
  [--quantized] \
  [--eval_4bit] \
  [--eval_4bit_backend <bnb>] \
  [--eval_4bit_quant_type <nf4|fp4>]
```

### Arguments

- `--model`: (Required) Path to the model to profile.
- `--tasks`: Comma-separated list of tasks to evaluate (e.g., `math,coder`). Default: `math,coder`.
- `--output_csv`: Path where the results CSV will be saved (appended if exists). Default: `output/gaussian_sanity_results.csv`.
- `--gpus`: Optional `CUDA_VISIBLE_DEVICES` string (e.g., `0,1`).
- `--tmp_dir`: Directory for temporary model saves. Default: `<TMP_DIR>`.
- `--start_layer`: (Optional) Start index for the layer sweep (inclusive). Defaults to 0.
- `--end_layer`: (Optional) End index for the layer sweep (inclusive). Defaults to the last layer.
- `--debug`: (Flag) Enable verbose debug output, printing state transitions and perturbation details.
- `--perturbation`: Comma-separated list of perturbations to apply (`avg`, `replace`, `add`). Default: `avg,replace`.
- `--quantized`: (Flag) Enable quantized model mode (direct FP8 perturbation). Auto-detected from `config.json` if not set.
- `--eval_4bit`: (Flag) Evaluate via a temporary **4-bit** checkpoint: perturb full precision weights, then quantize+save a temp 4-bit model before evaluation.
- `--eval_4bit_backend`: 4-bit backend for `--eval_4bit`. Currently supported: `bnb`.
- `--eval_4bit_quant_type`: 4-bit quant type for `--eval_4bit`: `nf4` (default) or `fp4`.

### Example

Profile a coder model on the `coder` task from layer 0 to 5, using GPU 2:

```bash
python gaussian_profiler.py \
  --model <MODELS_DIR>/deepseek-coder-7b \
  --tasks coder \
  --output_csv ./results/profiling_coder.csv \
  --start_layer 0 \
  --end_layer 5 \
  --gpus 2
```

## Output

The script produces a CSV file with the following columns:

- `timestamp`: Time of the run.
- `model`: Path of the model being profiled.
- `task`: Evaluation task.
- `layer`: Layer index being perturbed (`-1` indicates baseline).
- `variant`: Component targeted (`attn`, `mlp`, or `none`).
- `perturbation`: Type of noise applied (`avg`, `replace`, `baseline`).
- `quantized`: Whether the model was profiled in quantized mode (`True`/`False`).
- `eval_4bit`: Whether evaluation used a temporary 4-bit checkpoint (`True`/`False`).
- `eval_4bit_backend`: 4-bit backend used (currently `bnb` or empty).
- `eval_4bit_quant_type`: 4-bit quant type used (currently `nf4`/`fp4` or empty).
- `score`: Evaluation score.

## Quantized Model Support (FP8)

The profiler supports **FP8 block-quantized** models produced by [llmcompressor](https://github.com/vllm-project/llmcompressor) with the `FP8_BLOCK` scheme (compressed-tensors format).

### How It Works

The profiler perturbs FP8 weight tensors directly, without modifying the quantization scales:

1. **Casts** the FP8 weight to bfloat16 (required for arithmetic — PyTorch does not support math ops on float8 dtypes).
2. **Generates** Gaussian noise matching the weight's mean and standard deviation.
3. **Applies** the perturbation (add, avg, or replace) in bfloat16.
4. **Clamps** the result to the representable FP8 range (±448) and casts back to `float8_e4m3fn`.

The `weight_scale` tensors are never touched — they are an intrinsic part of the quantized representation. This logic lives in `quantized_utils.py` and is invoked transparently when `--quantized` is set.

### Prerequisites

In addition to the base prerequisites:

- **compressed-tensors** (`pip install compressed-tensors`): Required for loading FP8 models.
- **Quantized models**: FP8-BLOCK quantized model directories (each containing `config.json` with `quantization_config`).

### Arguments

All original arguments are supported. Additional:

- `--quantized`: (Flag) Enable quantized model mode. If omitted, the script will **auto-detect** compressed-tensors models from `config.json`.

### Example

Profile a single quantized model:

```bash
python gaussian_profiler.py \
  --model <MODELS_DIR>/Llama-3.1-8B-UltraMedical-FP8-BLOCK \
  --tasks math \
  --output_csv output/profiling_ultramedical_fp8.csv \
  --quantized \
  --start_layer 0 \
  --end_layer 31 \
  --gpus 0
```

### Batch Script: `run_profiling_quantized.sh`

Profiles all 5 quantized models in sequence:

```bash
# Usage: ./run_profiling_quantized.sh [GPU_ID] [TASKS]
./run_profiling_quantized.sh 0 math
```

Models profiled:
- `calme-2.3-legalkit-8b-FP8-BLOCK`
- `Llama-3.1-8B-Instruct-multi-truth-judge-FP8-BLOCK`
- `Llama-3.1-8B-UltraMedical-FP8-BLOCK`
- `Llama-3.1-Hawkish-8B-FP8-BLOCK`
- `Llama-SafetyGuard-Content-Binary-FP8-BLOCK`

### Output

The CSV output includes an additional `quantized` column (`True`/`False`) to distinguish runs on quantized vs. standard models.

## 4-bit Evaluation Mode (INT4, via BitsAndBytes)

If you want to **operate on full-precision weights** (e.g., for perturbation/merging) but **evaluate in 4-bit**, use `--eval_4bit`.

### How It Works

For each perturbation step:

1. Load the baseline model in full precision.
2. Apply the perturbation to the requested layer/module weights.
3. Save a temporary **full-precision** checkpoint directory.
4. Quantize that temporary checkpoint to **4-bit** (creating a second temporary directory).
5. Evaluate the 4-bit checkpoint.
6. Delete both temporary directories (baseline is always reloaded per step).

### Marlin Compatibility

This workflow is **conceptually compatible** with Marlin-style 4-bit inference (perturb FP → quantize → eval), but **Marlin specifically requires GPTQ/AWQ-style quantized weights**. The current implementation uses the `bnb` (BitsAndBytes) backend because GPTQ/AWQ quantization libraries are not included here by default. If you later add a GPTQ/AWQ backend, the same two-temp-dir workflow can be reused to enable Marlin execution.

### Sample Script: `run_profiling_4bit.sh`

Runs the profiler on a model under `<MODELS_DIR>` with `--eval_4bit` enabled.
