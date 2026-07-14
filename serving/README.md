# SANDHI serving-deployment harness — Figure 6

This directory reproduces **Figure 6** of the paper: token **throughput** (top)
and **P95 TTFT** (bottom) across deployment configurations, comparing
independent serving (*baseline*) against SANDHI's deduplicated serving
(*sandhi*). SANDHI's memory savings expand KV-cache capacity, delaying
saturation and enabling higher throughput at increased load. Headline results:
in the 9-model scenario, DeepSeek-7B (480 KB/token KV) shows up to **2.93×
throughput** and **1052× P95 TTFT** improvement; Qwen-2.5-7B (56 KB/token)
shows **2.11×** and **564×** respectively.

## Contents

| path | what it is |
|---|---|
| `sandhi_scripts/` | the benchmark harness (from the SANDHI vLLM fork, see *Provenance*) |
| `sandhi_scripts/run_all.sh` | one-command driver: starts the model servers (baseline, then sandhi), runs all load sweeps, parses logs, renders plots |
| `sandhi_scripts/*_config.sh` | one config per deployment scenario (see table below) |
| `sandhi_scripts/parse_and_plot_results.py` | log parser + plot renderer (throughput and P95 TTFT vs request rate, baseline vs sandhi) |
| `sandhi_scripts/gpu_alloc.py` | ballast allocator — pins `GPU_ALLOC_GIB` GiB per GPU to emulate a smaller-memory deployment |
| `specs/` | SANDHI shared-layer merge specs consumed by the sandhi-mode servers (see *Merge specs*) |
| `specs/convert_spec.py` | converts a merging-pipeline spec (`../merging/`) into the serving spec format |

## Provenance

`sandhi_scripts/` is copied unmodified from the SANDHI vLLM fork:
<https://github.com/nandanmeda1999/vllm_merged_model>, branch
`users/nmeda6/shared-components`, commit
`a90cc51e9e19b16172fc28090ba1b0189b50427b`. That fork (a vLLM derivative,
Apache-2.0) contains the SANDHI serving stack itself — the
`--shared-layers-ptrs-path` / `--shared-layers-spec-path` server flags used
below are implemented there. The prebuilt runtime ships as a Docker image (next
section), so there is no need to build the fork.

## Requirements

- **Hardware:** NVIDIA GPUs with enough memory for the scenario (see the
  scenario table; single-pool scenarios run on 1 GPU, cross-family pools use 2
  GPUs with tensor parallelism 2). The paper's constrained-memory scenarios are
  emulated by the ballast allocator, so larger GPUs are fine.
- **Software:** Docker with the NVIDIA container runtime.
- **Network/storage:** the models are pulled from Hugging Face on first launch
  (~15 GB per 7–8B model; the 9-model pool needs ~140 GB of HF cache). Mount a
  persistent HF cache directory (shown below) to avoid re-downloading.
- All model repos in the shipped configs are **ungated** on Hugging Face; no
  token is needed. (Only `template_config.sh`'s example bench model
  `meta-llama/Llama-3.1-8B` is gated — not used by any paper scenario.)

## Setup

1. Pull the prebuilt serving image (~22 GB) and tag it with the short name the
   instructions use:

   ```bash
   docker pull nandanmeda1999/sandhi-inference:latest
   docker tag  nandanmeda1999/sandhi-inference:latest sandhi:latest
   ```

2. Start the container (adjust `--gpus` and the HF-cache host path):

   ```bash
   docker run --rm -it --runtime nvidia --name sandhi_eval \
       --gpus '"device=0,1"' \
       --ipc=host \
       -p 8000:8000 \
       -v /path/to/hf_cache:/root/.cache/huggingface \
       --entrypoint /bin/bash \
       sandhi:latest
   ```

3. From the host, copy the harness **and the merge specs** into the container:

   ```bash
   docker cp serving/sandhi_scripts/ sandhi_eval:/vllm-workspace/
   docker cp serving/specs/llama5/llama_merged_spec_up_to_cutoff.json sandhi_eval:/vllm-workspace/sandhi_scripts/
   docker cp serving/specs/ds2/ds_merged_spec_up_to_cutoff.json      sandhi_eval:/vllm-workspace/sandhi_scripts/
   docker cp serving/specs/qwen2/merged_spec_up_to_cutoff.json       sandhi_eval:/vllm-workspace/sandhi_scripts/
   ```

   (Configs reference the spec by the relative filename in `SHARED_SPEC`, so
   the spec for the scenario you run must sit next to the scripts — or edit
   `SHARED_SPEC` in the config to an absolute path.)

## Run

Inside the container:

```bash
cd /vllm-workspace/sandhi_scripts
bash run_all.sh --config <config_file> --run-base-dir /vllm-workspace/<result_dir>
```

`run_all.sh` does everything for one scenario: starts the ballast allocators,
launches one vLLM server per model in **baseline** mode, sweeps the configured
request rates with the vLLM serving benchmark, restarts the servers in
**sandhi** mode (shared-layer dedup per the spec), repeats the sweep, then
renders the comparison plots. Logs stream to
`<result_dir>/logs/{servers,benchmarks}/`; watch them from a second terminal
via `docker exec -it sandhi_eval /bin/bash`.

## Deployment scenarios (configs)

| config | pool | GPUs (TP) | ballast/GPU | spec file (`SHARED_SPEC`) |
|---|---|---|---|---|
| `ds2_40gb_config.sh` | 2× DeepSeek-7B (coder, math) | 1 | 100 GiB | `ds_merged_spec_up_to_cutoff.json` (shipped: `specs/ds2/`) |
| `qwen2_40gb_config.sh` | 2× Qwen2.5-7B (Coder, Math) | 1 | 100 GiB | `merged_spec_up_to_cutoff.json` (shipped: `specs/qwen2/`) |
| `llama5_config.sh` | 5× Llama-3.1/3-8B domain fine-tunes | 1 | 20 GiB | `llama_merged_spec_up_to_cutoff.json` (shipped: `specs/llama5/`) |
| `llama-qwen_config.sh` | 7 models: 5× Llama + 2× Qwen2.5-7B | 2 (TP=2) | 30 GiB | `merged_spec_up_to_cutoff.json` (generate — see *Merge specs*) |
| `llama-qwen-ds_config.sh` | 9 models: 5× Llama + 2× Qwen + 2× DeepSeek | 2 (TP=2) | 12 GiB | `merged_spec_up_to_cutoff.json` (generate — see *Merge specs*) |
| `template_config.sh` | template for new pools | — | — | — |

The ballast (`GPU_ALLOC_GIB`) pins that much GPU memory before the servers
start, emulating the paper's constrained-memory deployments on whatever GPU you
have; scale it to your GPU size so that the *free* memory matches the intended
scenario. Request rates, prompt counts, and input/output lengths per benchmark
target are set in each config.

Approximate wall-clock per scenario: dominated by model downloads on first run;
the sweeps themselves are minutes per request rate per target
(`NUM_PROMPTS × rates × targets × 2 modes`). <!-- TODO(user): fill measured runtimes -->

## Merge specs

The sandhi-mode servers share weights according to a JSON spec: a list of
groups, each group a list of `{model, layer, component}` tensors deduplicated
into one. **These specs are the output of the merging pipeline in
[`../merging/`](../merging/)** — this is the hand-off point between the paper's
two pipelines (Figure 5 → Figure 6).

- Shipped, ready to use: `specs/llama5/`, `specs/ds2/` (the specs used for the
  paper's serving runs) and `specs/qwen2/` (the shipped `C` operating point of
  the `qwen25_2` pool).
- To produce a spec for any pool yourself: run the merging pipeline
  (`../merging/scripts/run_figures.py --sets 6a|6b|6c|6d|6e …`, see
  `../merging/README.md`) and take the operating-point spec
  `runs/<run>/analysis/<set>/C.json` (or `B/Bpm/Cpm/Kpm.json`) — it is directly
  consumable as `SHARED_SPEC`. Specs emitted by
  `../merging/scripts/build_merge_groups.py` use short attention names; pass
  them through `specs/convert_spec.py` to normalize.
- The 7- and 9-model cross-family specs used for the paper's Figure 6 runs are
  regenerated this way (compose `llama5` + `qwen25_2` [+ `deepseek2`] run-sets;
  the shipped per-pool profiles and results in `../merging/results/` make this
  cheap). <!-- TODO(user): if you still have the exact paper spec files for the
  7- and 9-model serving runs, drop them into specs/llama-qwen/ and
  specs/llama-qwen-ds/ -->

## Results

After a run, `<result_dir>/results/` contains:

- `plots/*.png` — per benchmark target: **token throughput vs request rate**
  and **P95 TTFT vs request rate**, baseline vs sandhi — the Figure 6 panels;
- a parsed metrics table (P95 TTFT ms, P95 ITL ms, output token throughput
  tok/s per mode × target × request rate) extracted from the benchmark logs.

Expected outcome: sandhi mode sustains higher request rates before saturation —
throughput and P95 TTFT curves separate sharply from baseline at the upper
request rates, with the largest gaps for models with large per-token KV caches
(DeepSeek-7B).
