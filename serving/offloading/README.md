# CPU-offloading experiments — supplementary recorded data

Recorded runs and launcher for the paper's offloading comparison (Figures 8
and 9). These figures are out of scope for this artifact's claims; the logs
are provided as-is.

- **Eliminating offloading** (2× Llama-3.1-8B on a 1×A100-40GB
  budget): with independent serving the weights don't fit alongside the 2 GB
  per-model KV budget, so the baseline must offload 2 GB to CPU; SANDHI's
  dedup fits entirely in GPU memory.
  - `results_llama_kv2gb_offload2gb/` — baseline (`--cpu-offload-gb 2`)
  - `results_llama_kv2gb_no-offload/` — SANDHI-equivalent (no offload)
- **Reducing offload volume** (3× Qwen3-32B on a 2×A100-80GB
  budget, 32 GB KV): both configurations must offload, but SANDHI reduces the
  transferred volume.
  - `results_qwen3_kv32gb_offload14gb/` — baseline (`--cpu-offload-gb 14`)
  - `results_qwen3_kv32gb_offload4.39gb/` — SANDHI (`--cpu-offload-gb 4.39`,
    the volume left after fig5a's dedup)
  - `results_qwen3_kv32gb_no-offload/` — no-offload reference

Each run dir holds the raw `vllm serve` log (`server_<scenario>.log`, with the
full `non-default args` config line) and the `vllm bench serve` sweep log
(`bench_<scenario>.log`, full percentile tables per request rate: median/P95/
P99 TTFT, output token throughput). `plots/` has the rendered llama-pair panels.

`run_offload_experiment.sh` re-runs any configuration; the recorded parameters
are baked in as defaults (max_model_len 4096, max_num_seqs 250, 100 prompts,
random 100-in/900-out, `--ignore-eos`). Rates: llama {2,5,7,10,15,20}, qwen3
{1,2,5,10}.

The recorded runs used stock vLLM 0.11.0 on H200 hosts, with the A100
memory budgets set via `--kv-cache-memory-bytes`. The offload penalty is visible
directly in the bench logs (e.g. Qwen3 @ 14 GB offload, RPS 1: median TTFT
732 ms vs sub-100 ms unoffloaded).

Note: the recorded llama runs served `fin-llama3.1-8b`, not the pair named in
the paper's Figure 9 description. It is a Llama-3.1-8B fine-tune, so the
weight footprint and offload volume are identical.
