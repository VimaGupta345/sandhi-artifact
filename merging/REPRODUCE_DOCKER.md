# Reproduce Figures 5 & 6 in Docker — evaluator runbook

Runs the SANDHI pipeline (gaussian profiler → clustering → MICR → analysis →
Pareto + operating-point specs) inside the reference container. **Code is mounted
at runtime — no image rebuild.** Everything runs as **your own user — no `sudo`,
no `root`, no `chown`** — and is **self-contained by default** (models/datasets
download into your repo; nothing under `/scratch` needs mounting).

## Two ways to reproduce — pick one
- **A · Skip straight to the measurements** (no GPU, minutes). Every shipped run
  under `runs/<run>/` already carries the merge trajectory
  (`micr/<set>/<model>/steps.csv`) **and** the finished measurements
  (`analysis/<set>/report.csv` — memory savings + per-model accuracy at each
  operating point A/B/C/Bpm/Cpm/Kpm/P). If you only want the numbers or the
  Figure-5 plots, read those / re-render (Part 3) — nothing to recompute. To
  rebuild the actual merged **weights** at an operating point *without*
  re-profiling or re-running MICR, **replay** the recorded `steps.csv` — see
  `GENERATE_VARIANTS.md`.
- **B · Re-run the full pipeline from the base models** (GPU, hours). Re-derive
  profiler → clustering → MICR → analysis from scratch, as below — use this to
  reproduce the merge *selection* itself, not just consume it.

## What you need
- Docker with GPU support.
- The **reference image**, pulled from Docker Hub and tagged to the short name
  the `dock()` wrapper uses:
  ```bash
  docker pull oytunkuday/merge-tools:reference
  docker tag  oytunkuday/merge-tools:reference merge-tools:reference
  ```
  Pin by digest for stability: `oytunkuday/merge-tools@sha256:3fe1edb3ca9a4f1bf53707dd580511bc94d187c92a6634b5fed648a8c8a6004f`.
  (Built from the `ikhyunAn/merge-tools-docker` repo if you need to rebuild it.)
- The artifact repo checked out somewhere you can write (call it `$CODE`).
- Network on first run (weights + datasets download once into `$CODE/hf_cache`).

You do **not** build anything, edit any site-packages, or run any privileged
command. One cache holds everything: `$CODE/hf_cache/{models,datasets,modules}`.

## The wrapper (paste once per shell)
```bash
CODE=/path/to/artifacts_worktree      # <-- the artifact repo (you own it)

dock() {   # dock <run_figures args...>
  docker run --rm --gpus all --shm-size=32g --ulimit nofile=524288:524288 \
    --user "$(id -u):$(id -g)" -e USER="$(id -un)" -e HOME=/tmp \
    -v "$CODE":/workspace/merge_tools \
    -w /workspace/merge_tools merge-tools:reference \
    python scripts/run_figures.py "$@"
}
```
The `dock()` wrapper is just shorthand — **each run is one explicit `docker
run`**. E.g. the Qwen-2.5 pair, force-reserving two cards another user is on:
```bash
docker run --rm --gpus all --shm-size=32g --ulimit nofile=524288:524288 \
  --user "$(id -u):$(id -g)" -e USER="$(id -un)" -e HOME=/tmp \
  -v "$CODE":/workspace/merge_tools -w /workspace/merge_tools merge-tools:reference \
  python scripts/run_figures.py --run-name qwen25_2 --sets qwen25_2 \
    --stages all --gpus auto --exclude-gpus 1,2 --profiles redo
```
That is the whole thing — no model mounts, no `HF_HOME`. Why it's sudo-free:
- `--user "$(id -u):$(id -g)"` — the container runs as **you**; outputs under
  `runs/` are yours (no root-owned files, no cleanup sudo).
- `-e USER=$(id -un) -e HOME=/tmp` — satisfy libraries that look the user up by
  UID (`getpass.getuser()`), without editing `/etc/passwd`.
- The driver **always uses a persistent repo-local cache** (`<repo>/hf_cache`) —
  it ignores the image's baked, root-owned `HF_HOME=/cache` (which is also
  ephemeral under `--rm`), so downloads persist and there's no permission issue
  whether you run as your user or root. lm_eval's external-`--include_path` crash
  is fixed in-process by `micr/_lmeval_shim` (auto-added to `PYTHONPATH`). Nothing
  in the root-owned image is modified. (A deliberate `HF_HOME`/`--hf-home` — e.g.
  a shared cluster cache — is still honored.)
- `--shm-size=32g` — `/dev/shm` cap (not reserved). 16g is enough for the 2-GPU
  DeepSeek run; 32g gives headroom for the 5-GPU Llama run and 32B/vLLM work.
  Raise it if you ever see `Bus error` / `/dev/shm` errors.

### Reuse this cluster's existing downloads (skip re-download)
The team already has the 7–8B weights at `/scratch/shared_dir/unified_models`.
Point one env var at them (read-only mount is fine) to skip re-downloading:
```bash
dock() {
  docker run --rm --gpus all --shm-size=32g --ulimit nofile=524288:524288 \
    --user "$(id -u):$(id -g)" -e USER="$(id -un)" -e HOME=/tmp \
    -e SANDHI_MODELS_DIR=/scratch/shared_dir/unified_models \
    -v /scratch/shared_dir/unified_models:/scratch/shared_dir/unified_models \
    -v "$CODE":/workspace/merge_tools \
    -w /workspace/merge_tools merge-tools:reference \
    python scripts/run_figures.py "$@"
}
```
`SANDHI_MODELS_DIR` is the single model-cache root (default `$HF_HOME/models`);
absent models still download there. (The 32B Qwen models for fig5a are not in
`unified_models`; they download on first use.)

## Sanity check first (no compute, ~5 s)
```bash
dock --run-name smoke --sets llama5 --stages clustering --dry-run --gpus 0
```
Prints `SANDHI pipeline · llama5` and a `DRY ... clustering.py ...` line, then
`done`. (Delete `runs/smoke` afterward.)

## Run the pipeline (Path B) — the essential commands
Each is one self-contained, resumable command writing under `runs/<run-name>/`.
`--stages all` = prereq → profiler → clustering → MICR → analysis → finaleval →
collect. `--gpus auto` grabs every free card and the dispatcher skips busy ones
(<8 GiB free); add `--exclude-gpus 1,2` to force-reserve cards another user is
on. Each `auto` takes all free cards, so run these **sequentially**.

### 1 · The four run-sets (the only expensive step)
```bash
# panel (a) — 3× Qwen-3-32B (1 GPU/model; --n-gpus-32b 2 to scale up)
dock --run-name qwen32b3  --sets fig5a     --stages all --gpus auto --profiles redo
# panel (b) — 5 Llama; UltraMedical evaluated on medqa (1-shot)
dock --run-name llama5    --sets llama5    --stages all --gpus auto --profiles redo
# DeepSeek pair (scaling ON) — feeds cross-family panel (c)
dock --run-name deepseek2 --sets deepseek2 --stages all --gpus auto --profiles redo
# Qwen-2.5 7B pair — feeds 12-model panel (d)
dock --run-name qwen25_2  --sets qwen25_2  --stages all --gpus auto --profiles redo
```

### 2 · Panel (c) — the 5-Llama + 2-DeepSeek JOINT cross-family merge
All 7 clustered together with the scaler ON so shape-compatible cross-family
tensors can merge; reuses the llama5 + deepseek2 profiles from step 1:
```bash
dock --run-name llama5_deepseek2 --sets llama5_deepseek2 --stages all --gpus auto --profiles reuse
```

> **Reproducibility note.** `llama5` and `qwen32b3` gate on multiple-choice /
> loglikelihood tasks and reproduce **bit-exact** on rerun. `deepseek2` and
> `qwen25_2` gate merges on **generative** evals (`humaneval`, `gsm8k-cot`) that
> vLLM does **not** score bit-deterministically (even at temp 0), so re-running
> those two from scratch can land on **different B/C cutoffs** — **rerunning them
> is not recommended**; prefer the shipped profiles/results. The Pareto *shape*
> is stable; the exact cutoff is not.

## Where results land (all user-owned)
`runs/<run-name>/analysis/<set>/`:
- **report.csv** — savings % + per-model accuracy drops
- **sweep.csv** — full cutoff table · **pareto.png** — the frontier
- **B.jsonl / C.jsonl** and **B.json / C.json** (and `Bpm`/`Cpm`/`Kpm`/`P`) —
  operating-point merge specs (one group per line, and a single JSON array;
  `self_attn.*` naming; single-model groups dropped — only k≥2 shared recipes
  that actually save memory). These are the compact *recipe*.
Intermediates: `runs/<run>/{profiler,clustering,micr,finaleval}/`. `collect` also
copies reportables into `results/<set>/` and profiles into `results/profiler/`.

To **materialize the actual merged model weights** at B/C from a run's
profiles/MICR outputs (a full HF model dir you can serve or eval), see
**`GENERATE_VARIANTS.md`** — it replays each model's `steps.csv` to the operating
cutoff and writes the merged safetensors to `variants/<run>/<point>_<model>/`.

## Expected time (H200-class GPUs)
- deepseek2 / qwen25_2: ~40–60 min each · llama5 from scratch: ~90–130 min on
  5 GPUs · qwen32b3: ~60–90 min · llama5_deepseek2 (profiles reused):
  ~40–60 min · rendering the panels: seconds (no GPU).

## Profiles: reuse vs from-scratch (`--profiles`)
Per-model, set-independent. `redo` = re-sweep every model from scratch;
`reuse` = profile only models whose CSV is absent in the run's `profiler/` dir;
`ask` (default) = prompt when interactive, **fall back to reuse when
non-interactive** so Docker never hangs. A poisoned/truncated profile is always
re-swept regardless.

## Render the Figure-5 panels (a–d)
The bar charts read the aggregated tables in `plots/data/`, produced from each
run's `report.csv` by `scripts/build_plot_data.py`. Panel (d) (12-model) is the
**additive composition** of the qwen32b3 + qwen25_2 + llama5_deepseek2 reports —
assembled inside `build_plot_data.py`, so no extra run is needed.
```bash
# 1) tables: each run's report.csv (M-split) -> plots/data/*.csv
docker run --rm --user "$(id -u):$(id -g)" -e USER="$(id -un)" -e HOME=/tmp \
  -v "$CODE":/workspace/merge_tools -w /workspace/merge_tools merge-tools:reference \
  python scripts/build_plot_data.py

# 2) charts:
docker run --rm --user "$(id -u):$(id -g)" -e USER="$(id -un)" -e HOME=/tmp \
  -v "$CODE":/workspace/merge_tools -w /workspace/merge_tools/plots/source \
  merge-tools:reference python accuracy_memory_figures.py
# -> plots/figures/{3-Qwen-32B,5-llama,7-5-llama-2-DS,12-model}.pdf
```
Config→panel map and provenance are in `plots/README.md`.
