# SANDHI — SOSP 2026 Artifact

Artifact for the SOSP 2026 paper:

> **SANDHI: Fine-Grained Merging for Memory Efficient Multi-Model Serving**
> <!-- TODO(user): author list -->

SANDHI selectively merges fine-grained components (per-layer attention/MLP
projections) across a pool of fine-tuned models, then serves the pool with the
merged tensors deduplicated in GPU memory. The paper's two headline claims:

- **Figure 5 (merging):** 26.7%–49.8% memory reduction across pools of 2–12
  models while *increasing* average task accuracy by up to 1.2% (per-task up to
  1.9%) over independent serving; LoRA and full-model merging baselines are
  restricted to same-family pools.
- **Figure 6 (serving):** the freed memory expands KV-cache capacity, delaying
  saturation — in the 9-model scenario, up to 2.93× throughput and 1052× P95
  TTFT improvement for DeepSeek-7B (480 KB/token KV), 2.11× / 564× for
  Qwen-2.5-7B (56 KB/token).

**Badges claimed: Available, Functional, Results Reproduced.** The final
camera-ready version of this artifact will be archived on Zenodo with a DOI;
this GitHub repository is the evaluation-time copy.

## Artifact structure

The artifact has two components, matching the paper's pipeline split. The
hand-off between them is the *merge spec* (which tensors are shared): the
merging pipeline emits it, the serving harness consumes it.

| component | reproduces | entry point | reference environment |
|---|---|---|---|
| [`merging/`](merging/) | Figure 5 (accuracy + memory Pareto, operating points) and the *model sets* behind every Figure 6 pool | `merging/scripts/run_figures.py` — one command per figure set | Docker Hub `oytunkuday/merge-tools:reference` (digest `sha256:3fe1edb3…`) |
| [`serving/`](serving/) | Figure 6 (throughput + P95 TTFT, baseline vs SANDHI serving) | `serving/sandhi_scripts/run_all.sh` — one command per deployment scenario | Docker Hub `nandanmeda1999/sandhi-inference:latest` |

Each component has its own detailed README:
**[`merging/README.md`](merging/README.md)** (with
`merging/REPRODUCE_DOCKER.md` as the evaluator runbook) and
**[`serving/README.md`](serving/README.md)**.

### Provenance

This repository is a self-contained packaging of two upstream repositories:

- `merging/` — [oytunkuday/merge_tools_artifacts](https://github.com/oytunkuday/merge_tools_artifacts) `@ main`,
  plus the recorded `clustering/candidates/` reference data (required by the
  memory-model validation check) and an env-override fix in
  `micr/run_eval_lora.py`.
- `serving/sandhi_scripts/` — the `sandhi_scripts/` directory of the SANDHI
  vLLM fork [nandanmeda1999/vllm_merged_model](https://github.com/nandanmeda1999/vllm_merged_model)
  `@ users/nmeda6/shared-components` (commit `a90cc51e9e19`), which also
  contains the SANDHI serving-stack source (prebuilt in the Docker image).

## Requirements

- **Hardware:** NVIDIA GPUs.
  - Merging: 1 GPU (≥40 GB) suffices for the Llama/Qwen2.5/DeepSeek 7–8B
    pools; the Llama pool profiling stage can use up to 5 GPUs in parallel;
    the Qwen3-32B pool (fig5a) wants ~5×80 GB (or one ≥80 GB GPU with the
    documented reduced-utilization settings).
  - Serving: 1–2 GPUs; constrained-memory scenarios are *emulated* by a
    ballast allocator, so any sufficiently large GPU works.
- **Software:** Docker with the NVIDIA container runtime. Everything else is
  inside the two reference images.
- **Storage/network:** ~50 GB image pulls; Hugging Face model downloads
  (~15 GB per 7–8B model, up to ~140 GB HF cache for the largest pools). All
  models used by the shipped experiments are ungated (no HF token needed),
  except the optional legacy LoRA-eval path noted under *Known limitations*.

## Kick-the-tires (≈15 minutes, no GPU compute)

1. **Merging pipeline dry-run** (validates environment, code paths, and set
   composition without computing anything):

   ```bash
   docker pull oytunkuday/merge-tools:reference
   docker tag  oytunkuday/merge-tools:reference merge-tools:reference
   CODE=$PWD/merging
   docker run --rm --gpus all --shm-size=32g --ulimit nofile=524288:524288 \
     --user "$(id -u):$(id -g)" -e USER="$(id -un)" -e HOME=/tmp \
     -v $CODE:/workspace/merge_tools -w /workspace/merge_tools \
     merge-tools:reference \
     python scripts/run_figures.py --run-name smoke --sets 5b \
       --stages clustering --dry-run --gpus 0
   ```

2. **Memory-model validation check** (pure CPU/Python; recomputes the recorded
   7-Llama reference and must report ~34.5% savings at the reference cutoffs):

   ```bash
   docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
     -v $CODE:/workspace/merge_tools -w /workspace/merge_tools \
     merge-tools:reference \
     python scripts/build_operating_points.py --pool 7_llamas --validate
   ```

3. **Serving image smoke** (image boots and the SANDHI-extended vLLM is
   present):

   ```bash
   docker pull nandanmeda1999/sandhi-inference:latest
   docker tag  nandanmeda1999/sandhi-inference:latest sandhi:latest
   docker run --rm --entrypoint /bin/bash sandhi:latest -lc "vllm --help | head -5"
   ```

## Reproducing the paper's claims

### Figure 5 — merging (accuracy + memory)

Follow **`merging/REPRODUCE_DOCKER.md`** (the step-by-step evaluator runbook).
Summary: one command per figure set,

```bash
python scripts/run_figures.py --run-name r1 --sets 5b --gpus auto   # via the dock() wrapper
```

with sets `5a` (3× Qwen3-32B), `5b` (5× Llama-8B), `5c` (+2 DeepSeek, 7
models), `5d` (12 models), and `6a`–`6e` (the Figure 6 pools' model sets).
Shipped per-model profiles (`merging/results/profiler/`) let you skip the
profiling stage (`--profiles reuse`); `--profiles redo` reruns clean-room.
Outputs land in `merging/runs/<run>/analysis/<set>/`: `report.csv`
(memory savings % + per-model accuracy drops), `pareto.png`, `sweep.csv`, and
the operating-point specs `B/C.json(l)`. The Figure 5 bar charts render from
the shipped aggregate tables via `merging/plots/` (see `merging/plots/README.md`),
and the No-merge / Full-merge / LoRA comparison bars reproduce per
`merging/BASELINES.md`. Scope and caveats (what reproduces end-to-end vs from
shipped tables, run-to-run variance on generative-eval pools) are documented
up front in `merging/README.md`.

### Figure 6 — serving (throughput + P95 TTFT)

Follow **`serving/README.md`**. Summary: one command per deployment scenario,

```bash
bash run_all.sh --config <scenario>_config.sh --run-base-dir /vllm-workspace/<out>
```

which benchmarks the pool in baseline mode and SANDHI mode and renders the
throughput / P95-TTFT comparison plots. Ready-to-use merge specs for the
llama5, DeepSeek, and Qwen2.5 pools ship in `serving/specs/`; specs for any
pool can be regenerated from the merging pipeline's operating-point output
(`analysis/<set>/C.json`) — that hand-off is documented in
`serving/README.md § Merge specs`.

## Known limitations / evaluator notes

- **Fig 5 rendered figures** aggregate the SANDHI runs with externally-computed
  baselines; they reproduce from the shipped tables in `merging/plots/data/`
  (regenerable per `BASELINES.md`), not in one shot from a fresh pipeline run.
- **Generative-eval pools** (`deepseek2`, `qwen25_2`) have run-to-run cutoff
  variance; `merging/README.md § Reproducibility` explains, and shipped
  results are provided.
- **Legacy LoRA eval script** (`merging/micr/run_eval_lora.py`): the LoRA
  *baseline* reproduces per `merging/BASELINES.md` with public adapters
  (`anjohn0077/NEXS-lora-adapters`); the legacy script defaults to a
  cluster-internal adapter path — override with `MICR_LORA_ADAPTER_ROOT`, and
  note its base model `meta-llama/Llama-3.1-8B` is license-gated on Hugging
  Face (accept the Meta license + `HF_TOKEN`).
- **7-/9-model serving specs**: the exact spec files from the paper's serving
  runs ship for the single-family pools; the cross-family 7-/9-model specs are
  regenerated via the documented merging→serving hand-off.
- Figure 6's *model sets* come from `merging/` (sets `6a`–`6e`); the serving
  *measurements* come from `serving/`.

## License

MIT (see [LICENSE](LICENSE)). `serving/sandhi_scripts/` originates from a
vLLM fork and the SANDHI serving stack inherits vLLM's Apache-2.0 license in
its own repository; the scripts are included here under the artifact's MIT
license with attribution above.

## Contact

<!-- TODO(user): corresponding author name + email for the AE period -->
