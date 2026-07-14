# SANDHI

Code and data for the SOSP 2026 paper **"SANDHI: Fine-Grained Merging for
Memory Efficient Multi-Model Serving"**.

SANDHI serves a pool of fine-tuned LLMs in less GPU memory by selectively
merging fine-grained components (per-layer attention/MLP projections) across
models and deduplicating the merged tensors at serving time. Across pools of
2–12 models this frees 26.7%–49.8% of weight memory while slightly *improving*
average task accuracy, and the freed memory expands KV-cache capacity — in the
9-model deployment, up to 2.93× higher throughput and 1052× lower P95 TTFT for
DeepSeek-7B compared with serving the models independently.

## Repository layout

```
merging/   the SANDHI merging pipeline: gaussian profiling → component
           clustering → MICR merge-and-evaluate → memory/accuracy analysis.
           Produces the Figure 5 results and, for every pool, the Pareto
           frontier and operating-point merge specs.
serving/   the serving benchmark harness: runs each deployment pool in
           baseline mode and SANDHI mode (shared-layer dedup) under load and
           plots throughput and P95 TTFT — the Figure 6 experiments.
           serving/specs/ holds the merge specs consumed by the servers.
```

The two parts connect through the **merge spec** — a JSON list of tensor
groups to deduplicate. The merging pipeline emits it
(`runs/<run>/analysis/<set>/C.json`); the serving harness consumes it
(`SHARED_SPEC` in each scenario config). See `serving/README.md § Merge specs`.

Each part has its own README: [`merging/README.md`](merging/README.md)
(runbook: [`merging/REPRODUCE_DOCKER.md`](merging/REPRODUCE_DOCKER.md)) and
[`serving/README.md`](serving/README.md).

Both parts run inside prebuilt Docker images — no local builds needed:

| part | image |
|---|---|
| merging | `oytunkuday/merge-tools:reference` |
| serving | `nandanmeda1999/sandhi-inference:latest` |

`serving/sandhi_scripts/` comes from the SANDHI vLLM fork
([nandanmeda1999/vllm_merged_model](https://github.com/nandanmeda1999/vllm_merged_model),
branch `users/nmeda6/shared-components`, commit `a90cc51e9e19`), which
implements the serving stack itself; `merging/` mirrors
[oytunkuday/merge_tools_artifacts](https://github.com/oytunkuday/merge_tools_artifacts).

## Requirements

- NVIDIA GPUs and Docker with the NVIDIA container runtime.
  - Merging: one ≥40 GB GPU covers the 7–8B pools (Llama profiling can spread
    over up to 5 GPUs); the Qwen3-32B pool wants ~5×80 GB or one ≥80 GB GPU
    with the reduced-utilization settings described in `merging/README.md`.
  - Serving: 1–2 GPUs; the paper's constrained-memory deployments are emulated
    by a ballast allocator, so larger GPUs are fine.
- Disk and network: ~50 GB of image pulls plus Hugging Face model downloads
  (~15 GB per 7–8B model; up to ~140 GB of HF cache for the largest pool).
  All models used by the shipped experiments are ungated.

## Quick start

Pull the images:

```bash
docker pull oytunkuday/merge-tools:reference
docker tag  oytunkuday/merge-tools:reference merge-tools:reference
docker pull nandanmeda1999/sandhi-inference:latest
docker tag  nandanmeda1999/sandhi-inference:latest sandhi:latest
```

Sanity-check the merging pipeline without computing anything (dry run + the
recorded 7-Llama memory-model check, which should report ~34.5% savings):

```bash
CODE=$PWD/merging
docker run --rm --gpus all --shm-size=32g --ulimit nofile=524288:524288 \
  --user "$(id -u):$(id -g)" -e USER="$(id -un)" -e HOME=/tmp \
  -v $CODE:/workspace/merge_tools -w /workspace/merge_tools \
  merge-tools:reference \
  python scripts/run_figures.py --run-name smoke --sets 5b \
    --stages clustering --dry-run --gpus 0

docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v $CODE:/workspace/merge_tools -w /workspace/merge_tools \
  merge-tools:reference \
  python scripts/build_operating_points.py --pool 7_llamas --validate
```

## Reproducing the results

**Figure 5 (merging):** one command per figure set — see
`merging/REPRODUCE_DOCKER.md` for the full runbook.

```bash
python scripts/run_figures.py --run-name r1 --sets 5b --gpus auto
```

Sets: `5a` (3× Qwen3-32B), `5b` (5× Llama-8B), `5c` (7 models), `5d` (12
models), and `6a`–`6e` (the Figure 6 pools). Shipped per-model profiles let you
skip the profiling stage (`--profiles reuse`); `--profiles redo` reruns from
scratch. Results land in `merging/runs/<run>/analysis/<set>/`: `report.csv`
(savings % and per-model accuracy deltas), `pareto.png`, `sweep.csv`, and the
operating-point specs. The Figure 5 bar charts render via `merging/plots/`,
with the No-merge / Full-merge / LoRA baselines covered in
`merging/BASELINES.md`.

**Figure 6 (serving):** one command per deployment scenario — see
`serving/README.md`.

```bash
bash run_all.sh --config <scenario>_config.sh --run-base-dir /vllm-workspace/<out>
```

This benchmarks the pool in baseline and SANDHI modes across a request-rate
sweep and renders the throughput / P95-TTFT comparison plots. Ready-made merge
specs for the llama5, DeepSeek, and Qwen2.5 pools are in `serving/specs/`.

## Notes and caveats

- The rendered Figure 5 bar charts aggregate SANDHI runs with
  externally-computed baselines and reproduce from the shipped tables in
  `merging/plots/data/` (regenerable per `merging/BASELINES.md`).
- Pools gated on generative evals (`deepseek2`, `qwen25_2`) have some
  run-to-run cutoff variance; see `merging/README.md § Reproducibility`.
- The legacy LoRA eval script (`merging/micr/run_eval_lora.py`) defaults to a
  cluster-internal adapter path — set `MICR_LORA_ADAPTER_ROOT`, and note its
  base model `meta-llama/Llama-3.1-8B` is license-gated on Hugging Face.
- Cross-family (7- and 9-model) serving specs are regenerated from the merging
  pipeline's operating points as described in `serving/README.md § Merge specs`.

## License

MIT (see [LICENSE](LICENSE)). `serving/sandhi_scripts/` originates from a vLLM
(Apache-2.0) fork; it is included here with attribution above.
