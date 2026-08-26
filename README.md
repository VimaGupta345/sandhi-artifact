# SANDHI

Artifact for the SOSP 2026 paper:

> **SANDHI: Fine-Grained Merging for Memory Efficient Multi-Model Serving**
> Vima Gupta, Oytun Kuday Duran, Nandan Suresh Meda, Ikhyun An (Georgia
> Institute of Technology), Ganesh Ananthanarayanan (Microsoft), and Anand
> Iyer (Georgia Institute of Technology).
> *Proceedings of the 31st ACM Symposium on Operating Systems Principles
> (SOSP '26).*

Changes made in response to artifact evaluation are summarized in
[CHANGES.md](CHANGES.md); citation metadata is in
[CITATION.cff](CITATION.cff).

SANDHI serves a pool of fine-tuned LLMs in less GPU memory by selectively
merging per-layer attention/MLP projections across models and deduplicating
the merged tensors at serving time. Across pools of 2–12 models this frees
26.7%–48.2% of weight memory while preserving full-dataset task accuracy
within a ≤2% budget, and the freed memory expands KV-cache capacity — up to
2.93× higher throughput and 1052× lower P95 TTFT than independent serving.
Per-pool operating points and their measured accuracy are documented in
[`merging/README.md`](merging/README.md).

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

We use NVIDIA H200 GPUs (141 GB HBM) unless stated otherwise. Machines are
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
tokens per second; TTFT is measured with vLLM's serving benchmark. Exact
versions: the serving image ships the SANDHI vLLM build
`0.11.1.dev14+gf0dd2fcb6` (a fork of vLLM v0.11); the merging environment pins
`vllm==0.11.2` for evaluation.

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
# inside the merging container (dock() wrapper from the runbook)
python scripts/run_figures.py --run-name r1 --sets 5b --gpus auto
```

Sets: `5a`–`5d` and `6a`–`6f`. Results land in
`merging/runs/<run>/analysis/<set>/` (`report.csv`, `pareto.png`, operating
points). Baselines: `merging/BASELINES.md`.

**Figure 6 (serving)** — one command per deployment scenario; full details in
[`serving/README.md`](serving/README.md):

```bash
# start the serving container (1 GPU for single-pool scenarios, 2 for cross-family)
docker run --rm -it --runtime nvidia --name sandhi_eval --gpus '"device=0,1"' \
    --ipc=host -p 8000:8000 -v /path/to/hf_cache:/root/.cache/huggingface \
    --entrypoint /bin/bash sandhi:latest

# from the host: copy in the harness and the merge specs
docker cp serving/sandhi_scripts/ sandhi_eval:/vllm-workspace/
docker cp serving/specs/. sandhi_eval:/vllm-workspace/sandhi_scripts/

# inside the container: one command per scenario
cd /vllm-workspace/sandhi_scripts
bash run_all.sh --config llama5_config.sh       --run-base-dir /vllm-workspace/llama5_out
bash run_all.sh --config ds2_40gb_config.sh     --run-base-dir /vllm-workspace/ds2_out
bash run_all.sh --config llama-qwen_config.sh   --run-base-dir /vllm-workspace/llama_qwen_out
bash run_all.sh --config llama-qwen-ds_config.sh --run-base-dir /vllm-workspace/llama_qwen_ds_out
# (qwen2_40gb_config.sh — Figure 6b — is under investigation; see serving/README.md)

# render the paper-style panels from any results directory (host; pandas+matplotlib)
cd serving/paper_plots
python parse_bench_logs.py --results-root ../results   # or your run's results dir
python performance_plots.py                            # -> figures/*.pdf
```

These commands are self-contained: models download automatically, and the
sandhi arm deduplicates the original checkpoints according to the spec — no
merged weights need to be built first. The recorded reference runs in
`serving/results/` (`*_variants`) additionally serve the materialized merged
weights, built with `merging/GENERATE_VARIANTS.md` and enabled via
`MODELS_SANDHI` (see `serving/README.md` § Serving the exact merged weights);
both configurations agree within a few percent.

Figures 7–11 and the §5.9 ablations are out of scope for this artifact.

## Dependencies

- **Python packages** — installable with pip. `merging/requirements.txt` is a
  pip-freeze lockfile of the reference environment (Python 3.12; install with
  `pip install --no-deps -r requirements.txt`); `merging/requirements-native.txt`
  is the minimal normally-installable set. The Docker images below ship both
  environments preinstalled, so no manual installation is needed.
- **OS-level dependencies** (CUDA 12.6 stack, vLLM builds, and
  [kvcached](https://github.com/ovg-project/kvcached) v0.1.3 — the elastic
  KV-cache allocator both serving arms run with) are contained in the two
  prebuilt Docker images, pinned by digest in `merging/README.md` and
  `serving/README.md` — pull them as shown there; no image build is needed or
  expected. All recorded reference results in this repository were produced
  with these exact images. Their sources are
  [`ikhyunAn/merge-tools-docker`](https://github.com/ikhyunAn/merge-tools-docker)
  (merging) and the SANDHI vLLM fork
  ([`nandanmeda1999/vllm_merged_model`](https://github.com/nandanmeda1999/vllm_merged_model))
  (serving). Host requirements: Docker with the NVIDIA container runtime and
  NVIDIA GPUs.
- **Exotic dependencies** are downloaded and built automatically:
  model weights and datasets resolve from Hugging Face into a repo-local cache
  on first use; `tinyBenchmarks` installs from its git URL (noted in
  `requirements.txt`); `mergekit` (Figure 5 full-merge baseline only) installs
  with `pip install mergekit` (`merging/BASELINES.md`).
- **Gated dependencies** — every model used by the paper's scenarios is
  ungated. The only gated repo is `meta-llama/Llama-3.1-8B`, needed solely as
  the base for the LoRA baseline: accept the Meta Llama 3.1 license on Hugging
  Face and authenticate with `HF_TOKEN` (`merging/BASELINES.md`). Without
  access, the shipped precomputed LoRA results (`merging/plots/data/lora/`)
  reproduce that baseline's figures.

## License

MIT (see [LICENSE](LICENSE)). `serving/sandhi_scripts/` originates from a vLLM
(Apache-2.0) fork; see `serving/README.md`.
