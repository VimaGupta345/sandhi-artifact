# SANDHI

Code for the SOSP 2026 paper **"SANDHI: Fine-Grained Merging for Memory
Efficient Multi-Model Serving"**.

SANDHI serves a pool of fine-tuned LLMs in less GPU memory by selectively
merging per-layer attention/MLP projections across models and deduplicating
the merged tensors at serving time. Across pools of 2–12 models this frees
26.7%–49.8% of weight memory while slightly improving average task accuracy,
and the freed memory expands KV-cache capacity — up to 2.93× higher throughput
and 1052× lower P95 TTFT than independent serving.

## Layout

```
merging/   gaussian profiling → component clustering → MICR merge-and-evaluate
           → memory/accuracy analysis (Figure 5, per-pool merge specs)
serving/   load benchmarks per deployment pool, baseline vs SANDHI mode;
           throughput and P95 TTFT (Figure 6)
```

The merging pipeline emits a merge spec (`analysis/<set>/C.json`); the serving
harness consumes it (`SHARED_SPEC`, prebuilt copies in `serving/specs/`).
See [`merging/README.md`](merging/README.md) and
[`serving/README.md`](serving/README.md).

Both parts run in prebuilt Docker images:

| part | image |
|---|---|
| merging | `oytunkuday/merge-tools:reference` |
| serving | `nandanmeda1999/sandhi-inference:latest` |

## Setup

We use NVIDIA H200 GPUs (144 GB HBM) unless stated otherwise. Machines are
equipped with AMD EPYC processors, connected via NVLink where applicable, and
run Ubuntu 22.04 with CUDA 12.6. Tensor-parallel degree is set per experiment
according to the compute and memory requirements of multi-model serving.

**Online serving.** We measure throughput and time to first token at steady
state, using representative workloads with varying input and output lengths —
a mean of one thousand tokens per request and a prefill-to-decode ratio of
1:10, reflecting real-world production deployments.

**Metrics.** Memory savings are the percentage reduction in GPU memory
footprint relative to the total HBM used across GPUs. Accuracy uses
task-appropriate metrics from EleutherAI's evaluation harness. Throughput is
tokens per second; TTFT is measured with vLLM (v0.11.0).

**Baselines.** (1) Independent serving — each fine-tuned model loaded
separately; (2) Multi-SLERP — merges all layers without SANDHI's selective
strategy; (3) LoRA — adapters applied to the corresponding layers at runtime.

## Quick start

```bash
docker pull oytunkuday/merge-tools:reference
docker tag  oytunkuday/merge-tools:reference merge-tools:reference
docker pull nandanmeda1999/sandhi-inference:latest
docker tag  nandanmeda1999/sandhi-inference:latest sandhi:latest
```

Sanity-check the merging pipeline (dry run, then the recorded 7-Llama memory
check — expected ~34.5% savings):

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

**Figure 5 (merging)** — one command per figure set; runbook in
[`merging/REPRODUCE_DOCKER.md`](merging/REPRODUCE_DOCKER.md):

```bash
python scripts/run_figures.py --run-name r1 --sets 5b --gpus auto
```

Sets: `5a`–`5d` and `6a`–`6e`. Results land in
`merging/runs/<run>/analysis/<set>/` (`report.csv`, `pareto.png`, operating
points). Baselines: `merging/BASELINES.md`.

**Figure 6 (serving)** — one command per deployment scenario; see
[`serving/README.md`](serving/README.md):

```bash
bash run_all.sh --config <scenario>_config.sh --run-base-dir /vllm-workspace/<out>
```

## License

MIT (see [LICENSE](LICENSE)). `serving/sandhi_scripts/` originates from a vLLM
(Apache-2.0) fork; see `serving/README.md`.
