# SANDHI merging pipeline — artifact

Reproduces the Figure 5 AND Figure 6 memory/accuracy results: gaussian profiling
-> component clustering -> MICR merge-and-evaluate -> corrected memory analysis
(distinct-tensor freed / sum of full on-disk model sizes) with Pareto plots
and operating-point merge specs.

## What this artifact reproduces — and how

The pipeline reproduces the **model sets** (memory/accuracy Pareto frontier +
operating-point merge specs) for **every** Figure 5 AND Figure 6 pool. The
paper's rendered *figures* have narrower scope — read this before starting:

| target | reproducible here? | how |
|---|---|---|
| **Model sets — all fig5 + fig6 pools** (`report.csv`, `pareto.png`, per-point `.json`+`.jsonl`) | **yes** | §A / `REPRODUCE_DOCKER.md` "All 9 figures" |
| **Figure 5 plots** (accuracy + memory-savings bars) | **yes — from our `report.csv`** | §C / `plots/README.md` (`scripts/build_plot_data.py` ETL) |
| **Fig 5 comparison baselines** (No-merge · Full-merge · LoRA) | **yes — all on the eval subset (M)** | §B: No-merge (`report.csv` A) · Full-merge (`anjohn0077/NEXS-multislerp-merges`) · LoRA (`anjohn0077/NEXS-lora-adapters`) |
| **Figure 6 figures** (serving-deployment plots) | **no — out of scope** | only fig6's *model sets* (row 1) come from this pipeline |
| **Operating points** (per set) | **yes** | A / B / C (global) · Bpm / Cpm / Kpm (per-model) · P (paper-close) — see § Operating points |

### A. Model sets — all 9 fig5 + fig6 pools (the pipeline's core deliverable)
Run the 4 atomic run-sets once, then compose every pool at the analysis stage —
full commands in **`REPRODUCE_DOCKER.md` → "All 9 figures"**. Each pool emits
`analysis/<set>/`: `report.csv` (savings % + per-model drops), `sweep.csv`,
`pareto.png`, `B/C.json` + `B/C.jsonl` (operating-point specs). **This includes
the fig6 pools (6a–6e)** — for Figure 6 the pipeline's job ends at the Pareto
frontier + specs.
- *32B note:* the accuracy footing for the figures is the **M-split eval subset**
  (already in `report.csv`), so no full-set 32B eval is needed. If you do want the
  32B full-set number, `run_eval.py` replay now aligns the 32B plan to `steps.csv`
  (the B1 fix) — use it directly, not `run_eval_32b.py`. See `CHANGES.md`.
- *Caveat:* `deepseek2`/`qwen25_2` gate on generative evals → run-to-run cutoff
  variance (§ Reproducibility below); prefer shipped results, or run several and
  keep the best.
- To materialize the actual **merged model weights** at any operating point
  (`B`/`C` global or `Bpm`/`Cpm`/`Kpm` per-model, `P` paper-close) →
  `GENERATE_VARIANTS.md`.

### B. Fig 5a/5b comparison baselines (No-merge · Full-merge · LoRA) — reproducible separately
The bar charts compare SANDHI against three baselines that the SANDHI *merge*
pipeline doesn't itself emit, but which are each reproducible (results ship
pre-computed in `plots/data/`; re-running regenerates the baseline bars):
- **No-merge** — eval each source model on its benchmark (plain lm_eval; this is
  the unmerged point-A reference). Shipped: `plots/data/vllm-no-merge.csv`.
- **Full-merge** — a DARE / SLERP merge of the pool's models into one (the paper
  uses **multi-SLERP**), then eval. Compute with
  [mergekit](https://github.com/arcee-ai/mergekit). Shipped:
  `plots/data/full_merge/*.csv` (slerp/dare columns).
- **LoRA** — rank-128 adapters (mergekit-extracted from the same domain
  fine-tunes) at **`anjohn0077/NEXS-lora-adapters`** — a manifest repo pointing to
  per-domain adapters (`anjohn0077/NEXS-<domain>-lora`, + 6 Qwen3-32B). Serve each
  on its base model with vLLM+LoRA and eval per that repo's **README serving/eval
  guide**; `manifest.json` maps domain → source model → adapter repo → benchmark.
  Shipped: `plots/data/lora/*.csv`.
See `BASELINES.md` for the concrete commands.

### C. Figure 5 plots (accuracy / memory-savings bars)
`plots/source/accuracy_memory_figures.py` renders the Fig 5 panels from
`plots/data/` → `plots/figures/*.pdf` (see **`plots/README.md`**). Those tables
are the paper's **aggregated** numbers (SANDHI results + the §B external
baselines), **not** auto-derived from a fresh pipeline run — so the figures
reproduce **from the shipped tables**, not end-to-end from §A. (End-to-end would
need an ETL from `report.csv` into the SANDHI columns plus the §B baselines.)

### D. Figure 6 (serving-deployment figures) — out of scope
Figure 6's serving-benchmark plots come from a downstream serving harness that is
**not** part of this artifact. What the pipeline reproduces for Figure 6 is each
pool's **model set** (Pareto + B/C specs — row 1 / §A, sets `fig6a`–`fig6e`).

## One command per figure set

    python scripts/run_figures.py --run-name <name> --sets 5b --gpus 0
    python scripts/run_figures.py --run-name <name> --list-sets

Figure 5: 5a (3x Qwen3-32B), 5b (5x Llama-3.1-8B), 5c (5b + 2x DeepSeek), 5d (12
models). Figure 6: 6a–6e serving-deployment pools (compositions of the same
run-sets + recorded DeepSeek reuse; no new profiling). List several sharing a
run-set in one `--sets` to analyze them off a single profiling pass. Stages
(resumable): prereq, profiler, clustering, micr, analysis, finaleval, collect.
`collect` copies the reportables (step CSVs, sweep table, Pareto plot,
operating-point jsonls) into results/.

## Reproduce in Docker (recommended for evaluators)

The reference environment ships as a Docker Hub image; the pipeline **code is
mounted at runtime**, so no rebuild is needed. **Get the image once** (pull and
tag to the short name the wrapper uses):

    docker pull oytunkuday/merge-tools:reference
    docker tag  oytunkuday/merge-tools:reference merge-tools:reference

(digest `oytunkuday/merge-tools@sha256:3fe1edb3ca9a4f1bf53707dd580511bc94d187c92a6634b5fed648a8c8a6004f`;
rebuildable from `ikhyunAn/merge-tools-docker`.) It runs as **your own user — no
`sudo`, no `root`, no `chown`** and is **self-contained**: one local cache
(`$CODE/hf_cache/{models,datasets,modules}`) holds everything, so no `/scratch`
mount or `HF_HOME` is required. See `REPRODUCE_DOCKER.md` for the full runbook.

    CODE=/path/to/this/repo
    dock() { docker run --rm --gpus all --shm-size=32g --ulimit nofile=524288:524288 \
      --user "$(id -u):$(id -g)" -e USER="$(id -un)" -e HOME=/tmp \
      -v $CODE:/workspace/merge_tools \
      -w /workspace/merge_tools merge-tools:reference \
      python scripts/run_figures.py "$@"; }

    dock --run-name smoke --sets 5b --stages clustering --dry-run --gpus 0   # sanity (no compute)

`--user` makes outputs land owned by you; `-e USER/HOME` satisfy libraries that
look up the user by UID; the driver **always uses a persistent repo-local cache**
(`<repo>/hf_cache`), ignoring the image's baked (root-owned, ephemeral) `/cache`;
and lm_eval's
external-`--include_path` crash is fixed in-process by `micr/_lmeval_shim`
(auto-added to `PYTHONPATH`) — nothing in the root-owned image is modified.
`--shm-size=32g` is a `/dev/shm` cap (not reserved) with headroom for the 5-GPU
Llama run and 32B/vLLM work.

### Two profiling modes (`--profiles`)

Profiling is per-MODEL and set-independent; complete profiles ship in
`results/profiler/`. `--profiles` governs the profiler stage:

- **Reuse shipped profiles (fast).** Seed the run's profiler dir from
  `results/profiler/`, then `--profiles reuse` re-sweeps only models whose CSV
  is absent:

      mkdir -p $CODE/runs/r1/profiler
      cp -p $CODE/results/profiler/gaussian_*.csv* $CODE/runs/r1/profiler/
      dock --run-name r1 --sets 5b --stages clustering,micr,analysis,collect --gpus auto --profiles reuse

- **Profile from scratch (clean-room).** Don't seed; run the full pipeline with
  `--profiles redo` (re-sweeps every model from scratch):

      dock --run-name r1_scratch --sets 5b --stages profiler,clustering,micr,analysis,collect --gpus auto --profiles redo

  `--profiles ask` (default) prompts per model when interactive, and falls back
  to reuse when non-interactive so Docker/batch never hangs. A poisoned or
  truncated profile is always re-swept regardless of the flag.

### Where results land

Runs write into the mounted repo, so results appear on the host at
`runs/<run-name>/`:

- `runs/<run>/analysis/<set>/` — **report.csv** (savings % + per-model A/B/C
  drops), sweep.csv (full cutoff table), **pareto.png**, **B.jsonl / C.jsonl**
  (operating-point merge specs)
- `runs/<run>/{profiler/gaussian_*.csv, clustering/<set>/, micr/<set>/<model>/steps.csv}` — intermediates
- `collect` also copies reportables into `results/<set>/` and profiles into the
  shared, per-model `results/profiler/`

Distinct `--run-name`s keep multiple runs (e.g. two medical benchmarks) side by
side. A full, cluster-specific walkthrough is in `REPRODUCE_DOCKER.md`; the
figure→model→benchmark map is in `FIGURE_MODEL_MAP.md`.

## Operating points, per-model cutoffs & figure composition

`build_operating_points.py` emits these points per set (`report.csv` +
`<point>.jsonl` merge specs):

| point | rule | cutoff |
|---|---|---|
| `A` | unmerged reference | none |
| `B` / `C` | global worst-model drop ≤1% / ≤2% | one cutoff for the whole pool (`cutoff` col) |
| `Bpm` / `Cpm` | **per-model** ≤1% / ≤2% | each model its own deepest in-budget cutoff (`<model>__cutoff` cols; `cutoff` reads `pm`) |
| `Kpm` | per-model savings/accuracy **knee** | per-model (recommended balance) |
| `P` | closest to the paper's reported memory | one global cutoff — **single-run figures only** (fig5a/b/c) |

**Per-model cutoffs** let a fragile member stay shallow while a robust one merges
deep (e.g. the cross-7 pool: llama→c55, deepseek-coder→c14), recovering savings a
single global cutoff can't (17% → 33%). Cutoffs are **run-specific** — "cutoff N"
indexes each run's own `steps.csv`, so a model's cutoff differs between the
5-llama run and the cross-7 run.

**Figure composition is additive.** Only **five base runs** are ever needed —
`llama5`, `deepseek2`, `qwen25_2`, `qwen32b3` (within-family) plus the **one**
joint cross-family run `llama5_deepseek2`. Only llama↔deepseek can cross-merge
(both hidden=4096); Qwen2.5 (3584) and Qwen3-32B (5120) share with nothing, so
their freed bytes just add. Every figure pool composes as `savings = (Σ atom
freed)/(Σ atom size)` — e.g. fig5d(12) = **37.7% ≈ paper 38%**, fig6e(9), fig6c(7)
— with **no new MICR** for the 9/12-model pools.

- `scripts/compose_figures.py` — auto-detects available base runs and composes
  every figure → `results/FIGURE_COMPOSITIONS.md`.
- `scripts/suggest_variants.py` — the four suggested variants per set →
  `results/VARIANT_SUGGESTIONS.md`.
- `GENERATE_VARIANTS.md` — how to materialize any variant (per-model or global)
  by replaying `steps.csv` to the point's cutoff(s).

## Locked methodology

- profiler evaluates on split P, MICR gates on split M (seeded 50/50, seed 42),
  final numbers on the full dataset; each phase measures its own baseline
- perturbation: avg only; groups attn,mlp; noise seed 1234 (deterministic per cell)
- clustering avgability threshold 5%; MICR drop tolerance 2.0
- scaling: auto per-op (cross-family or deepseek/qwen participants); pure-Llama off
- memory: per (layer,component) slot, models sharing an identical merge recipe
  store one tensor (freed = (k-1) x size); denominator = sum of FULL model sizes

## Reproducibility: generative vs. multiple-choice tasks

The pipeline is deterministic **except** where it gates merges on a *generative*
benchmark. MICR accepts/rejects each merge on its split-M eval score, so:

- **Multiple-choice / loglikelihood tasks** — `llama5` (truthfulqa, medqa, sst2,
  mmlu-law, mmlu-econ) and `qwen32b3` (tinyMMLU, medqa, m_mmlu_ru) — are
  bitwise-deterministic. These run-sets reproduce **exactly**: profiles, MICR
  trajectory, and B/C operating points are identical across reruns.
  Note that `qwen32b3` is deterministic **because Light-IF-32B is gated on
  `tinyMMLU`** (a deterministic MC proxy), *not* on its native `ifeval` — which
  is a vLLM **generative** task and would be non-deterministic like the pool
  below. If you re-point Light-IF to `ifeval`, qwen32b3 inherits the same
  generative-eval variance.
- **Generative tasks** — `humaneval` (Coder) and `gsm8k-cot` (Math), i.e. the
  **`deepseek2`** and **`qwen25_2`** run-sets — are scored by vLLM, which is
  **not bit-deterministic even at temperature 0** (batching/kernel
  nondeterminism). A few flipped problems move a merge's score by a point or
  two; when that flips a single accept/reject near the drop-tolerance boundary,
  the sequential MICR trajectory diverges from there and the B/C cutoffs shift.
  The **profiler scores themselves are stable** (~0.25 pt noise) — it's the
  step-by-step gating that amplifies it.

**Therefore, re-running `deepseek2` / `qwen25_2` from scratch is NOT expected to
reproduce the exact operating points, and is not recommended** — prefer the
shipped profiles/results for those two pools, or treat their B/C cutoffs as
having run-to-run variance. The Pareto **shape** (memory-vs-accuracy trade-off)
is stable; the exact chosen cutoff is not. The MC/loglikelihood pools reproduce
bit-exact and are the ones to re-run for a clean-room check.

## Requirements

Python env with torch / transformers / vllm / lm_eval / datasets / pandas /
matplotlib — exact pins in `requirements.txt` (Python 3.12, torch 2.9+cu128;
note flashinfer-python and flashinfer-cubin MUST be the same version or every
vLLM engine refuses to boot). Models resolve to ONE local cache root:
`$SANDHI_MODELS_DIR` if set, else `$HF_HOME/models` (HF_HOME is always the
repo-local `./hf_cache`; the driver ignores the image's baked, ephemeral
`/cache` unless you deliberately set `HF_HOME`/`--hf-home`). Weights download
there from `hf_repo` when missing, so no
second model source or machine-specific path is needed. Dataset/module caches are
user-writable repo-local dirs (the script sets them automatically). Model
registry: models_download/hf_repos.json (relative dir-names).

## Large (32B) models: GPU memory layout

Profiler/MICR jobs keep the target model RESIDENT on the GPU for the whole run
(in-place perturb/revert + delta-shard saves need it), while every evaluation
runs as a subprocess that loads the saved candidate from disk — so one card
carries TWO copies of the model during evals ("co-residency"). Budget per job
on a 140 GiB-class card (H200):

- 7–8B models: 15 GB resident + eval copy — one GPU, no tuning needed. vLLM
  eval engines run `gpu_memory_utilization=0.8` (the default 0.9 demands
  125.8 GiB free and cannot boot next to any resident parent).
- 32B, HF-backend tasks (multiple choice): 62 GB resident + 62 GB eval copy +
  activations ≈ 127 GB — fits ONE GPU; auto batch sizing absorbs the reduced
  headroom (smaller eval batches, slightly slower cells).
- 32B, vLLM-backend tasks (generative, e.g. ifeval): vLLM RESERVES
  `utilization x total` at boot, so a 62 GB parent forbids the standard 0.8
  (111.8 GiB needed vs ~77 GiB free). Single-GPU works only at
  `gpu_memory_utilization ~= 0.55` (engine weights + ~15 GiB KV; fine for
  capped evals). Alternatives: give the job 2 GPUs, or CPU-offload the parent
  during evals (~10–20 s per cell extra at 32B).
- Concurrent jobs MUST NOT share a tmp root (`--tmp_dir` per job; the driver
  does this automatically): the persistent-eval candidate dir has a fixed
  basename inside it, and shared roots race.

## Recorded references

micr/results/{7_llamas,6_deepseek}_standardized + clustering/candidates/... are
the recorded runs used for fig5c reuse and for the memory-model validation:
`python scripts/build_operating_points.py --pool 7_llamas --validate`
must report ~34.5% at the reference cutoffs.
