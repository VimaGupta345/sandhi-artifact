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
| `specs/` | one merge spec per scenario, emitted by the merging pipeline (see *Merge specs*) |
| `specs/convert_spec.py` | normalizes legacy merging-side spec naming into the serving spec format |

## Provenance

`sandhi_scripts/` originates from the SANDHI vLLM fork:
<https://github.com/nandanmeda1999/vllm_merged_model>, branch
`users/nmeda6/shared-components`, commit
`a90cc51e9e19b16172fc28090ba1b0189b50427b`, with artifact-side fixes applied
here (pool aligned to Table 2, per-scenario spec filenames, and the sandhi-mode
spec-coverage guard in `server_utils.sh`). That fork (a vLLM derivative,
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

   Pin by digest for stability:
   `nandanmeda1999/sandhi-inference@sha256:3e5c79604bbf18ae48f9d7668971d9953fe34b012267eff721a56520f8605f9f`.
   The image contains the SANDHI vLLM build (`0.11.1.dev14+gf0dd2fcb6`, from
   the fork/commit above) and **[kvcached](https://github.com/ovg-project/kvcached)
   v0.1.3**, an elastic GPU-virtual-memory KV-cache allocator. The harness runs
   **both arms — baseline and sandhi — with kvcached enabled**
   (`ENABLE_KVCACHED=true`, `KVCACHED_AUTOPATCH=1` in `run_all.sh`), so the
   comparison isolates SANDHI's weight dedup: the baseline is independent
   serving with the same elastic KV allocator, not stock vLLM static
   allocation.

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
   docker cp serving/specs/. sandhi_eval:/vllm-workspace/sandhi_scripts/
   ```

   Each scenario config names its own spec file (`SHARED_SPEC`, distinct per
   scenario), resolved relative to the working directory — keep the specs next
   to the scripts, or set an absolute path in the config. Before starting
   sandhi-mode servers, `server_utils.sh` verifies the spec exists and covers
   every model in the pool, and refuses to start otherwise.

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

| config | pool | merging set | GPUs (TP) | ballast/GPU | spec (`SHARED_SPEC`) |
|---|---|---|---|---|---|
| `ds2_40gb_config.sh` | 2× DeepSeek-7B (coder, math) | `fig6a` | 1 | 100 GiB | `ds2_spec.json` |
| `qwen2_40gb_config.sh` | 2× Qwen2.5-7B (Coder, Math) | `fig6b` | 1 | 100 GiB | `qwen2_spec.json` |
| `llama5_config.sh` | 5× Llama-3.1-8B domain fine-tunes | `fig6d` | 1 | 20 GiB | `llama5_spec.json` |
| `llama-qwen_config.sh` | 7 models: 5× Llama + 2× Qwen2.5-7B | `fig6f` | 2 (TP=2) | 30 GiB | `llama_qwen_spec.json` |
| `llama-qwen-ds_config.sh` | 9 models: 5× Llama + 2× Qwen + 2× DeepSeek | `fig6e` | 2 (TP=2) | 12 GiB | `llama_qwen_ds_spec.json` |
| `template_config.sh` | template for new pools | — | — | — | — |

The pools use exactly the models the merging pipeline evaluates (Table 2 of the
paper). Earlier revisions of this harness served two models the pipeline never
profiled (`TsinghuaC3I/Llama-3-8B-UltraMedical`, `us4/fin-llama3.1-8b`); the
configs were reconciled to the Table 2 pool (`Llama-3.1-8B-UltraMedical`,
`Llama-3.1-Hawkish-8B`) so that every serving spec is a pipeline output.

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

- The five shipped specs are each pool's **`Cpm` operating point** (every model
  at its own deepest cutoff with ≤2% M-split drop — the accuracy budget stated
  in the paper), taken from `../merging/results/fig6{a,b,d,e,f}/Cpm.json`. For
  the single-family pools `Cpm` coincides with the global `C` point; for the
  cross-family pools it is the point where one fragile model does not cap the
  others.
- To regenerate them (or produce a spec for a new pool): run the merging
  pipeline's analysis stage over the composed run-sets and take the
  operating-point spec of your choice — e.g.
  `python scripts/run_figures.py --run-name specs --sets 6a,6b,6d,6e,6f
  --stages analysis,collect` against the shipped run data, then
  `runs/specs/analysis/<set>/{B,C,Bpm,Cpm,Kpm}.json`. Any of these is directly
  consumable as `SHARED_SPEC`. (Specs emitted by the legacy
  `scripts/build_merge_groups.py` use short attention names; pass them through
  `specs/convert_spec.py` to normalize.)

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
