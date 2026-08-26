# SANDHI serving-deployment harness — Figure 6

This directory reproduces **Figure 6** of the paper: token **throughput** (top)
and **P95 TTFT** (bottom) across deployment configurations, comparing
independent serving (*baseline*) against SANDHI's deduplicated serving
(*sandhi*). SANDHI's memory savings expand KV-cache capacity, delaying
saturation and enabling higher throughput at increased load. Headline results
(paper §5.3): in the 9-model scenario, DeepSeek-7B (480 KB/token KV) shows up
to **2.93× throughput** and **1052× P95 TTFT** improvement, Llama-3.1-8B
(128 KB/token) **2.11×** and **564×**, and Qwen-2.5-7B (56 KB/token) **1.72×**
and **~300×**. (The paper's Figure 6 caption misattributes the Llama numbers
to Qwen; §5.3's text has the correct per-family assignment used here.)

## Contents

| path | what it is |
|---|---|
| `sandhi_scripts/` | the benchmark harness |
| `sandhi_scripts/run_all.sh` | one-command driver: starts the model servers (baseline, then sandhi), runs all load sweeps, parses logs, renders plots |
| `sandhi_scripts/*_config.sh` | one config per deployment scenario (see table below) |
| `sandhi_scripts/parse_and_plot_results.py` | log parser + plot renderer (throughput and P95 TTFT vs request rate, baseline vs sandhi) |
| `sandhi_scripts/gpu_alloc.py` | ballast allocator — pins `GPU_ALLOC_GIB` GiB per GPU to set the scenario's memory budget |
| `specs/` | one merge spec per scenario, emitted by the merging pipeline (see *Merge specs*) |
| `specs/convert_spec.py` | normalizes legacy merging-side spec naming into the serving spec format |
| `results/` | recorded reference runs for the five deployment scenarios (see `results/README.md`) |
| `paper_plots/` | renders the Figure 6 panels in the paper's style from any results directory |
| `offloading/` | supplementary recorded offloading runs — outside the artifact's claims |

The SANDHI serving stack itself — the `--shared-layers-ptrs-path` /
`--shared-layers-spec-path` server flags used below — is implemented in the
SANDHI vLLM fork (<https://github.com/nandanmeda1999/vllm_merged_model>, a
vLLM derivative, Apache-2.0), from which `sandhi_scripts/` originates. The
prebuilt runtime ships as a Docker image (below), so there is no need to build
the fork.

## Requirements

- **Hardware:** NVIDIA GPUs with enough memory for the scenario (see the
  scenario table; single-pool scenarios run on 1 GPU, cross-family pools use 2
  GPUs with tensor parallelism 2). The ballast allocator sets each scenario's
  memory budget, so larger GPUs are fine.
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
   The image contains the SANDHI vLLM build (`0.11.1.dev14+gf0dd2fcb6`, built
   from the fork above) and **[kvcached](https://github.com/ovg-project/kvcached)
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

   **Model-name matching caveat:** the loader matches spec entries against
   `basename(model)` of what the engine was launched with. HF repo ids
   (`Org/Name`) match fine, and so do local model directories named after the
   model (`/models/Name`). But if vLLM resolves an HF id to its cache
   *snapshot path* (this happens under `HF_HUB_OFFLINE=1`), the basename
   becomes the snapshot hash, **no spec entry matches, and the sandhi arm
   silently degrades to independent serving**. When running offline, point
   `MODELS` at named local model directories instead of HF ids.

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

| config | pool | merging set | GPUs (TP) | memory budget | ballast/GPU (H200) | spec (`SHARED_SPEC`) |
|---|---|---|---|---|---|---|
| `ds2_40gb_config.sh` | 2× DeepSeek-7B (coder, math) | `fig6a` | 1 | A100-40GB | 100 GiB | `ds2_spec.json` |
| `qwen2_40gb_config.sh` | 2× Qwen2.5-7B (Coder, Math) | `fig6b` | 1 | A100-40GB | 100 GiB | `qwen2_spec.json` |
| `llama5_config.sh` | 5× Llama-3.1-8B domain fine-tunes | `fig6d` | 1 | 80 GB | 40 GiB | `llama5_spec.json` |
| `llama-qwen_config.sh` | 7 models: 5× Llama + 2× Qwen2.5-7B | `fig6f` | 2 (TP=2) | 2× 80 GB | 38 GiB | `llama_qwen_spec.json` |
| `llama-qwen-ds_config.sh` | 9 models: 5× Llama + 2× Qwen + 2× DeepSeek | `fig6e` | 2 (TP=2) | constrained | 12 GiB | `llama_qwen_ds_spec.json` |
| `template_config.sh` | template for new pools | — | — | — | — | — |

The pools use exactly the models the merging pipeline evaluates (Table 2 of
the paper), so every serving spec is a pipeline output. The *memory budget*
column is the paper's deployment size for each scenario
(§5.3); the ballast pins enough of an H200's 141.6 GiB to leave that budget
free, so the improvement dynamics depend on the deployment budget, not on the
physical card. Note the harness pays a real per-server overhead beyond weights
(~4 GiB per server at TP=1, ~6 GiB per rank at TP=2), so the ballast values
above are calibrated to leave the *paper's KV headroom* after servers boot; if
your GPUs differ, adjust `GPU_ALLOC_GIB` so free-memory-after-boot matches
the scenario's KV budget rather than applying `GPU_GiB − budget` blindly.

The ballast (`GPU_ALLOC_GIB`) pins that much GPU memory before the servers
start, applying the paper's deployment budget on whatever GPU you
have; scale it to your GPU size so that the *free* memory matches the intended
scenario. Request rates, prompt counts, and input/output lengths per benchmark
target are set in each config.

Approximate wall-clock per scenario: dominated by model downloads on first run;
the sweeps themselves are minutes per request rate per target
(`NUM_PROMPTS × rates × targets × 2 modes`).

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

## Serving the exact merged weights

By default both arms serve the original fine-tuned checkpoints; the sandhi
arm deduplicates each spec group onto the owner's tensor. Throughput/TTFT
depend only on this sharing structure, not on tensor values. To serve the
*exact merged weights Figure 5 validates* instead:

1. Materialize the variants with the merging pipeline's replay
   (`../merging/GENERATE_VARIANTS.md`, § *Rebuilding the recorded reference
   variants*) — the merged models are too large to ship, so replay recreates
   them byte-identically from the recorded merge trajectories.
2. Define `MODELS_SANDHI` in the scenario config: an array parallel to
   `MODELS` (same ports), pointed at the replay's output directories. The
   harness serves it in the sandhi arm only; a commented example is in
   `llama5_config.sh`.

No runtime scaling support is needed: same-pretrain variants share
byte-identical tensors by construction, and cross-pretrain members have the
per-member scaling (`*.scaling_factors.npz`) baked into the weights at
replay. The recorded `results/*_variants/` runs serve these builds for all
five scenarios, with ratios statistically identical to the owner-tensor runs
(`llama5_variants`: 7.1× vs 7.1× throughput).

## Results

**Recorded reference runs for all five scenarios ship in
[`results/`](results/)** — full server logs (both arms), raw benchmark
sweeps, and rendered plots, with a summary table in `results/README.md`.
The `results/*_variants/` runs are the reference measurements: their sandhi
arm serves the **materialized merged variants** (replayed from the recorded
MICR journals via `../merging/GENERATE_VARIANTS.md`) — byte-identical shared
tensors for the unscaled pools, per-member scaled-baked weights for
cross-pretrain members. Accuracy is the merging pipeline's result — full-set
scores for every model with an accuracy figure are in
`../merging/results/full_set_scores.csv` — while this harness measures
throughput and TTFT.

After a run of your own, `<result_dir>/results/` contains:

- `plots/*.png` — per benchmark target: **token throughput vs request rate**
  and **P95 TTFT vs request rate**, baseline vs sandhi — the Figure 6 panels.

The per-rate metrics themselves (P95 TTFT, output token throughput per mode ×
target × request rate) are in the raw benchmark logs; to extract them into a
single CSV, run `paper_plots/parse_bench_logs.py` against the results
directory.

To render the panels **in the paper's own style** (annotated log-scale bars,
paper fonts/colors) from any recorded or fresh results directory, see
[`paper_plots/`](paper_plots/) — two commands: `parse_bench_logs.py` then
`performance_plots.py`.

Expected outcome: sandhi mode sustains higher request rates before saturation —
throughput and P95 TTFT curves separate sharply from baseline at the upper
request rates, with the largest gaps for models with large per-token KV caches
(DeepSeek-7B).
