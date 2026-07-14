# Changes made for this artifact

What we modified to turn the research code into a **self-contained, no-sudo,
artifact-evaluation-ready** reproduction of Figures 5 & 6.

## Reproduction driver (`scripts/run_figures.py`)
- Renamed `run_figure5.py` → `run_figures.py`; added the **Figure 6** deployment
  pools (`fig6a`–`fig6e`) alongside Figure 5.
- `--gpus auto` auto-detects GPUs; `--exclude-gpus N` reserves cards; per-model
  dispatch only uses cards with <8 GiB already in use (busy/other-user GPUs are
  skipped automatically).
- Profiles **auto-adopt** from `results/profiler/` — evaluators never hand-copy.
- Added a `results/` folder + `collect` stage (copies report/sweep/pareto/jsonl
  and the shared per-model profiles).

## No-sudo, self-contained execution (for evaluators)
- Runs as `--user $(id -u):$(id -g)` with `-e USER/HOME` — **no root, no chown**.
- `HF_HOME` **always** resolves to the persistent repo-local `hf_cache/` — the
  driver ignores the image's baked, root-owned, `--rm`-ephemeral `/cache`
  entirely (a deliberate `HF_HOME`/`--hf-home` is still honored). Identical
  behavior as root or non-root; downloads persist.
- lm_eval's external-`--include_path` crash is fixed **in-process** via
  `micr/_lmeval_shim` (sitecustomize on `PYTHONPATH`) — site-packages untouched.
- **Single local model cache**: `$SANDHI_MODELS_DIR` else `$HF_HOME/models`
  (removed the `unified_models`/`hf_cache` split); missing weights download there.
- (`docker/` — a `chmod 0777 /cache` derived-image + patch — was **removed** as
  obsolete: nothing ever uses the baked `/cache` (driver *and* standalone
  `run_eval.py` both use the repo-local `hf_cache`), so the pushed base image
  `oytunkuday/merge-tools:reference` needs no cache fix.)

## De-hardcoding & cleanup
- Removed all hardcoded usernames/machine paths; registry `local_path`s are
  relative dir-names.
- Removed `experiments/` (moved `top_k` → `micr/top_k_experiment.py`) and the
  unused `clustering_algorithm`.
- `clustering.py` resolves models via the single cache root (no hardcoded
  `/scratch/.../unified_models`).

## Evaluation correctness
- UltraMedical **medqa is 1-shot everywhere** — profiler split configs AND the
  finaleval full-set variant (`eval_splits.py`, `eval_harness.py`, configs).
- `has_weights()` counts `.bin` shards too (deepseek-math ships `.bin`-only), so
  a fresh HF download is detected correctly.
- Fixed stage1/stage2 **tensor overlap** in emitted merge specs (last-write-wins
  dedup in `scripts/formatter.py`).

## Operating-point spec output (`B/C.json` + `B/C.jsonl`)
- Each operating point now emits **both** `{point}.json` (a single pretty-printed
  JSON array of all merge groups) and `{point}.jsonl` (one group per line), same
  content.
- **`self_attn.*` component naming** (HF state-dict path) for attention; `mlp.*`
  unchanged.
- **Single-model (touched-but-unmerged) groups are included** — the full
  per-tensor recipe, not just `>=2` merges (`formatter.build_spec(keep_singletons=True)`).
  The internal merges-only spec still drives the savings model.
- The parallel scaling-factor sidecar (`{point}.scaling_factors.npz`) is re-keyed
  to the same `self_attn.*` names so it still joins the spec 1:1.
- `collect` now copies `*.json` into `results/`. Existing runs were backfilled.

## Merge policy
- Per-runset **scaling** (`RUNSET_SCALING`): ON for different-pretrain pairs
  (qwen2.5 coder/math, deepseek coder/math), OFF for same-base sets (llama5, the
  qwen-32B trio).
- Joint **cross-family** run-set `llama5_deepseek2` (5 Llama + 2 DeepSeek
  clustered together with std-shift alignment, scaler on) so shape-compatible
  cross-family tensors can merge and self-select via MICR.

## Known issues / staged fixes
- **`--ulimit nofile=524288:524288`** is required for the 7B profilers: two
  vLLM-spawning evals in one container exhaust the default 1024 FD soft limit,
  which disables persistent-eval and then thrashes. Raising it keeps
  persistent-eval on (one reused engine). Add it to `dock()`.
- **32B finaleval replay is a CLI-wiring gap (small fix), not missing logic.**
  The replay engine is shared: `run_single_target_pipeline` (in `run_eval.py`)
  implements the full replay path (`save_variant_dir`, `replay_mode`,
  `micr_replay.json`), and `run_eval_32b.py` delegates to it — but
  `run_eval_32b.py`'s own argparse doesn't define/forward the 4 replay flags
  (`--replay_steps_csv`, `--replay_cutoff`, `--replay_cutoff_mode`,
  `--save_variant_dir`), so finaleval's call argparse-errors first. Fix = add the
  4 `add_argument`s + pass them through. Until then, fig5a/fig5d get the full
  Pareto (savings + M-split accuracy); only the 32B full-set numbers are blocked.
- scaling-sidecar `makedirs` — benign `FileNotFoundError` log noise during
  scaler-run finaleval (self-heals; the `.npz` still lands). Staged, not applied.

## Per-model runtime (profiling vs MICR), H200-class GPUs

Profiling evaluates one perturbed variant per (layer, group) cell on split P;
MICR merge-and-evaluates one accepted step at a time on split M. Profiling
generally dominates for generative-eval tasks and for 32B; MICR time scales with
the number of accepted steps.

| model | task | profiling | MICR | MICR steps |
|---|---|---:|---:|---:|
| **DeepSeek 7B** | | | | |
| deepseek-coder-7b-instruct-v1.5 | humaneval | 59.5m | 21.5m | 16 |
| deepseek-math-7b-instruct | gsm8k-cot | 56.7m | 22.5m | 16 |
| **Llama 8B** | | | | |
| Llama-3.1-8B-Instruct-multi-truth-judge | truthfulqa_mc2 | 28.8m | 35.5m | 53 |
| Llama-3.1-8B-UltraMedical | medqa (1-shot) | 26.8m | 18.0m | 39 |
| Llama-3.1-Hawkish-8B | mmlu_econometrics | 5.8m | 8.6m | 26 |
| Llama-SafetyGuard-Content-Binary | sst2 | 6.3m | 15.5m | 49 |
| calme-2.3-legalkit-8b | mmlu_professional_law | 22.0m | 24.5m | 43 |
| **Qwen 32B** | | | | |
| Light-IF-32B | tinyMMLU | 217.4m (3.6h) | 85.9m | 130 |
| MedGo | medqa | 189.5m (3.2h) | 82.9m | 128 |
| T-pro-it-2.0 | m_mmlu_ru | 128.0m (2.1h) | 107.9m | 130 |

Notes:
- **8B**: profiling and MICR are roughly balanced; light tasks (sst2,
  econometrics) profile in ~6 min, generative/MC tasks (truthfulqa, medqa,
  mmlu-law) in 20–30 min.
- **32B**: profiling dominates (2–3.6 h; 128 cells of slow 32B inference) vs
  ~1.5 h MICR.
- **Reused profiles** (e.g. the joint `llama5_deepseek2` run) skip profiling
  entirely — MICR only (9.5–37.7 min per model).
