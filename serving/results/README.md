# Recorded Figure 6 runs (reference results)

Complete recorded runs of all five deployment scenarios, produced with this
repository's harness, configs, and specs exactly as shipped: NVIDIA H200
(141.6 GiB), image `nandanmeda1999/sandhi-inference@sha256:3e5c7960…`,
2026-08-19. Each scenario dir holds the full server logs for both arms
(`logs/servers/{baseline,sandhi}/`), the raw benchmark sweeps
(`logs/benchmarks/`), and the rendered comparison plots (`results/plots/`).

Models were served from local directories named after each model (see the
model-name matching caveat in `../README.md`); the sandhi arm used the shipped
`Cpm` specs from `../specs/`.

## Summary (best-rate improvement, sandhi over baseline)

| scenario | benched model | throughput | P95 TTFT | paper (§5.3 / Fig 6) |
|---|---|---|---|---|
| `ds2` (2× DeepSeek, 40 GB) | deepseek-coder-7b | 2.06× (810→1668 tok/s) | up to 22× | diverges earliest (480 KB/token) |
| `qwen2` (2× Qwen2.5, 40 GB) | Qwen2.5-Coder-7B | ~1.0× | ~1.0× | diverges latest (56 KB/token; spec frees only 0.55 GB) |
| `llama5` (5× Llama, 80 GB) | Llama-3.1-UltraMedical | 7.1× (1.4k→10.0k) | up to 2466× | 1.1× / 197× (less constrained baseline) |
| `llama_qwen` (7 models, 2×80 GB) | Llama / Qwen | 3.3× / 2.4× | up to 1250× / ~500× | saturation extended 15→25+ / 30→40+ RPS |
| `llama_qwen_ds` (9 models) | DeepSeek / Llama / Qwen | **2.89× / 2.17× / 1.66×** | **991× / 615× / 135×** | **2.93× / 2.11× / 1.72×; ~1000× / ~500× / ~300×** |

Per-rate tables are in the benchmark logs; the plots show the full sweeps.
The divergence ordering (DeepSeek → Llama → Qwen, by per-token KV size) and
the 9-model headline ratios reproduce the paper's claims closely. Absolute
numbers depend on the emulated deployment budget (ballast) — see the scenario
table in `../README.md`.
