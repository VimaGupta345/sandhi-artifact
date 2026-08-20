# SANDHI merging pipeline

Reproduces the memory/accuracy results behind Figures 5 and 6: gaussian
profiling → component clustering → MICR merge-and-evaluate → memory analysis
(distinct-tensor freed / sum of full on-disk model sizes), emitting Pareto
plots and operating-point merge specs.

## Scope

The pipeline produces the **model sets** (memory/accuracy Pareto frontier and
operating-point merge specs) for every Figure 5 and Figure 6 pool:

| target | how |
|---|---|
| Model sets — all fig5 + fig6 pools (`report.csv`, `pareto.png`, per-point `.json`+`.jsonl`) | §A below / `REPRODUCE_DOCKER.md` |
| Figure 5 plots (accuracy + memory-savings bars) | §C / `plots/README.md` |
| Fig 5 comparison baselines (No-merge · Full-merge · LoRA) | §B / `BASELINES.md` |
| Figure 6 serving plots | the serving harness in [`../serving/`](../serving/) consumes this pipeline's specs |
| Operating points per set | A / B / C (global) · Bpm / Cpm / Kpm (per-model) · P (paper-close) — see § Operating points |

### A. Model sets — all fig5 + fig6 pools

Run the four atomic run-sets once, then compose every pool at the analysis
stage — full commands in `REPRODUCE_DOCKER.md`. Each pool emits
`analysis/<set>/`: `report.csv` (savings % + per-model drops), `sweep.csv`,
`pareto.png`, and per-point `.json`/`.jsonl` merge specs. For the fig6 pools
the pipeline's job ends at the Pareto frontier + specs; `../serving/` measures
the deployment plots. The merged model weights themselves are too large to
ship (tens of GB per pool, ~300 GB across the reference sets), so they are not
committed — `GENERATE_VARIANTS.md` recreates any variant deterministically by
replaying the recorded `steps.csv` to the chosen cutoff, byte-identical to
what the pipeline scored and the serving runs served.

### B. Fig 5 comparison baselines (No-merge · Full-merge · LoRA)

The bar charts compare SANDHI against three baselines the merge pipeline does
not itself emit. Each is reproducible (`BASELINES.md`); results ship
pre-computed in `plots/data/`:

- **No-merge** — each source model evaluated unmerged (`plots/data/vllm-no-merge.csv`).
- **Full-merge** — a multi-SLERP merge of the pool into one model, built with
  [mergekit](https://github.com/arcee-ai/mergekit) (`plots/data/full_merge/*.csv`).
- **LoRA** — rank-128 adapters at `anjohn0077/NEXS-lora-adapters` served on the
  base model with vLLM; `manifest.json` there maps domain → source model →
  adapter → benchmark (`plots/data/lora/*.csv`).

### C. Figure 5 plots

`plots/source/accuracy_memory_figures.py` renders the Fig 5 panels from the
aggregated tables in `plots/data/` → `plots/figures/*.pdf` (see
`plots/README.md`). `scripts/build_plot_data.py` regenerates the SANDHI columns
of those tables from each run's `report.csv`.

## One command per figure set

    python scripts/run_figures.py --run-name <name> --sets 5b --gpus 0
    python scripts/run_figures.py --run-name <name> --list-sets

Figure 5 sets: 5a (3× Qwen3-32B), 5b (5× Llama-3.1-8B), 5c (5b + 2× DeepSeek),
5d (12 models). Figure 6 sets: 6a–6f — compositions of the same run-sets, no
new profiling. Stages (resumable): prereq, profiler, clustering, micr,
analysis, finaleval, collect. `collect` copies the reportables (step CSVs,
sweep table, Pareto plot, operating-point specs) into `results/`.

## Reproduce in Docker

The reference environment ships as a Docker Hub image; the pipeline code is
mounted at runtime, so no rebuild is needed:

    docker pull oytunkuday/merge-tools:reference
    docker tag  oytunkuday/merge-tools:reference merge-tools:reference

Digest:
`oytunkuday/merge-tools@sha256:3fe1edb3ca9a4f1bf53707dd580511bc94d187c92a6634b5fed648a8c8a6004f`;
rebuildable from
[`ikhyunAn/merge-tools-docker`](https://github.com/ikhyunAn/merge-tools-docker).
The container runs as your own user (no root, no `chown`) and is
self-contained: one repo-local cache (`hf_cache/{models,datasets,modules}`)
holds all weights and datasets. Full runbook: `REPRODUCE_DOCKER.md`.

    CODE=/path/to/this/repo
    dock() { docker run --rm --gpus all --shm-size=32g --ulimit nofile=524288:524288 \
      --user "$(id -u):$(id -g)" -e USER="$(id -un)" -e HOME=/tmp \
      -v $CODE:/workspace/merge_tools \
      -w /workspace/merge_tools merge-tools:reference \
      python scripts/run_figures.py "$@"; }

    dock --run-name smoke --sets 5b --stages clustering --dry-run --gpus 0   # sanity (no compute)

`--user` keeps outputs owned by you; `-e USER/HOME` satisfy libraries that look
the user up by UID; the driver always uses the persistent repo-local
`hf_cache` (a deliberate `HF_HOME`/`--hf-home` is honored);
`--ulimit nofile` is required for the 7B profilers (two vLLM-spawning evals
exhaust the default 1024-FD limit); `--shm-size=32g` is a `/dev/shm` cap with
headroom for the 5-GPU Llama run and 32B/vLLM work.

### Profiling modes (`--profiles`)

Profiling is per-model and set-independent; complete profiles ship in
`results/profiler/`.

- **Reuse shipped profiles (fast).** Seed the run's profiler dir, then
  `--profiles reuse` re-sweeps only models whose CSV is absent:

      mkdir -p $CODE/runs/r1/profiler
      cp -p $CODE/results/profiler/gaussian_*.csv* $CODE/runs/r1/profiler/
      dock --run-name r1 --sets 5b --stages clustering,micr,analysis,collect --gpus auto --profiles reuse

- **Profile from scratch.** `--profiles redo` re-sweeps every model:

      dock --run-name r1_scratch --sets 5b --stages profiler,clustering,micr,analysis,collect --gpus auto --profiles redo

  `--profiles ask` (default) prompts per model when interactive and falls back
  to reuse when non-interactive. A truncated profile is always re-swept.

### Where results land

Runs write into the mounted repo, so results appear on the host under
`runs/<run-name>/`:

- `runs/<run>/analysis/<set>/` — `report.csv` (savings % + per-model drops),
  `sweep.csv` (full cutoff table), `pareto.png`, per-point `.json`/`.jsonl`
  merge specs
- `runs/<run>/{profiler/gaussian_*.csv, clustering/<set>/, micr/<set>/<model>/steps.csv}` — intermediates
- `collect` copies reportables into `results/<set>/` and profiles into the
  shared per-model `results/profiler/`

The figure→model→benchmark map is in `FIGURE_MODEL_MAP.md`.

## Operating points, per-model cutoffs, and figure composition

`build_operating_points.py` emits these points per set (`report.csv` +
`<point>.jsonl` merge specs):

| point | rule | cutoff |
|---|---|---|
| `A` | unmerged reference | none |
| `B` / `C` | global worst-model drop ≤1% / ≤2% | one cutoff for the whole pool (`cutoff` col) |
| `Bpm` / `Cpm` | per-model ≤1% / ≤2% | each model its own deepest in-budget cutoff (`<model>__cutoff` cols) |
| `Kpm` | per-model savings/accuracy knee | per-model (recommended balance) |
| `P` | closest to the paper's reported memory | one global cutoff — single-run figures only (fig5a/b/c) |

Per-model cutoffs let a fragile member stay shallow while a robust one merges
deep, recovering savings a single global cutoff cannot. Cutoffs are
run-specific: "cutoff N" indexes each run's own `steps.csv`, so a model's
cutoff differs between run-sets.

Figure composition is additive. Five base runs cover every pool — `llama5`,
`deepseek2`, `qwen25_2`, `qwen32b3`, plus the joint cross-family run
`llama5_deepseek2`. Only Llama↔DeepSeek can cross-merge (both hidden=4096);
Qwen2.5 (3584) and Qwen3-32B (5120) share with nothing, so their freed bytes
add. Every figure pool composes as `savings = (Σ atom freed)/(Σ atom size)` —
e.g. fig5d (12 models) = 37.7% ≈ paper 38% — with no new MICR for the
9/12-model pools.

- `scripts/compose_figures.py` — composes every figure from the available base
  runs → `results/FIGURE_COMPOSITIONS.md`
- `scripts/suggest_variants.py` — suggested variants per set →
  `results/VARIANT_SUGGESTIONS.md`
- `GENERATE_VARIANTS.md` — how to materialize any variant by replaying
  `steps.csv` to the chosen cutoff(s)

## Locked methodology

- profiler evaluates on split P, MICR gates on split M (seeded 50/50, seed 42),
  final numbers on the full dataset; each phase measures its own baseline
- perturbation: avg only; groups attn,mlp; noise seed 1234 (deterministic per cell)
- clustering avgability threshold 5%; MICR drop tolerance 2.0
- scaling: auto per-op (cross-family or deepseek/qwen participants); pure-Llama off
- memory: per (layer,component) slot, models sharing an identical merge recipe
  store one tensor (freed = (k−1) × size); denominator = sum of full model sizes

## Reproducibility: generative vs. multiple-choice tasks

The pipeline is deterministic except where it gates merges on a generative
benchmark:

- **Multiple-choice / loglikelihood tasks** — `llama5` (truthfulqa, medqa,
  sst2, mmlu-law, mmlu-econ) and `qwen32b3` (tinyMMLU, medqa, m_mmlu_ru) —
  are bitwise-deterministic: profiles, MICR trajectory, and operating points
  reproduce exactly across reruns.
- **Generative tasks** — `humaneval` and `gsm8k-cot`, i.e. the `deepseek2` and
  `qwen25_2` run-sets — are scored by vLLM, which is not bit-deterministic
  even at temperature 0. A few flipped problems can flip a single
  accept/reject near the tolerance boundary, after which the sequential MICR
  trajectory diverges and the B/C cutoffs shift. The Pareto shape is stable;
  the exact chosen cutoff is not. Prefer the shipped profiles/results for
  these two pools; the MC pools are the ones to re-run for a clean-room check.

## Requirements

Python environment with torch / transformers / vllm / lm_eval / datasets /
pandas / matplotlib — exact pins in `requirements.txt` (Python 3.12,
torch 2.9+cu128; `flashinfer-python` and `flashinfer-cubin` must be the same
version or vLLM engines refuse to boot). Models resolve to one local cache
root: `$SANDHI_MODELS_DIR` if set, else `$HF_HOME/models` (`HF_HOME` defaults
to the repo-local `./hf_cache`). Missing weights download there from the
`hf_repo` in the model registry (`models_download/hf_repos.json`); dataset and
module caches are repo-local and set automatically.

## Large (32B) models: GPU memory layout

Profiler/MICR jobs keep the target model resident on the GPU for the whole run
(in-place perturb/revert + delta-shard saves), while every evaluation runs as a
subprocess loading the saved candidate from disk — so one card carries two
copies of the model during evals. Budget per job on a 140 GiB-class card
(H200):

- 7–8B models: 15 GB resident + eval copy — one GPU, no tuning needed. vLLM
  eval engines run `gpu_memory_utilization=0.8`.
- 32B, HF-backend tasks (multiple choice): 62 GB resident + 62 GB eval copy +
  activations ≈ 127 GB — fits one GPU; auto batch sizing absorbs the reduced
  headroom.
- 32B, vLLM-backend tasks (generative, e.g. ifeval): vLLM reserves
  `utilization × total` at boot, so a 62 GB parent forbids the standard 0.8.
  Single-GPU works at `gpu_memory_utilization ≈ 0.55`; alternatively give the
  job 2 GPUs or CPU-offload the parent during evals.
- Concurrent jobs must not share a tmp root (`--tmp_dir` per job; the driver
  handles this automatically).

## Recorded references

`micr/results/{7_llamas,6_deepseek}_standardized` and
`clustering/candidates/...` are the recorded runs used for fig5c reuse and for
the memory-model validation:

    python scripts/build_operating_points.py --pool 7_llamas --validate

must report ~34.5% at the reference cutoffs.

All numbers in `report.csv`/`sweep.csv` (including `Bpm`/`Cpm`/`Kpm`/`P`) are
on the M-split eval subset — the gating signal. Full-set accuracy is recorded
in `results/full_set_scores.csv` for every model that appears in an accuracy
figure: full-set unmerged baseline vs full-set merged-variant score under a
matched protocol per model (e.g. UltraMedical is 1-shot + chat template on
both sides); the raw lm_eval outputs and the commands to reproduce either
side of any row are documented in `evaluation_results/README.md`. The
Qwen2.5 pair appears in no standalone accuracy figure — its
accuracy enters the paper only through the Fig 5d composition, on the M-split
footing of `report.csv`, and its serving scenarios report throughput/TTFT
only.

**Note — Qwen2.5 pair (under investigation).** The pair's full-dataset
accuracy at its shipped operating point (Coder: −4.88 on humaneval) is under
investigation. Until resolved, the pair carries no accuracy claim and the
2-Qwen serving scenario (Figure 6b) should be skipped; every other pool
replicates as documented.

**Fig 5a — operating point selected on full-set accuracy (48.2% / 92.6 GB).**
The M-split baseline was unstable for this pool (the M-split is a small
evaluation subsample), so the reported operating point was selected to
satisfy the ≤2% accuracy-drop budget on the **full datasets**: cutoff 94
(`results/fig5a/sweep.csv`), at which Light-IF is +3.09, T-pro −0.13, and
MedGo 0.00 (`results/full_set_scores.csv`). This point frees **92.6 GB —
48.2%** under this pipeline's memory model (192 GB, the sum of full on-disk
model sizes); the paper quotes the same point as 49.8% of its stated 186 GB
pool total.

Two notes for readers of `results/fig5a/report.csv`:
- The `P` row (cutoff 104, 49.78%) mechanically targets the paper's
  *percentage* under this pipeline's denominator; the reported operating
  point is cutoff 94, validated on the full datasets as above.
- All sizes here and in the paper are binary gigabytes (GiB), written "GB"
  following the paper's convention.
