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

Reference measurements from the `*_variants` runs — the sandhi arm serves the
materialized merged weights (see `../README.md` § Serving the exact merged
weights):

| scenario | benched model | throughput | P95 TTFT | paper (§5.3 / Fig 6) |
|---|---|---|---|---|
| `ds2` (2× DeepSeek, 40 GB) | deepseek-coder-7b | 1.97× (845→1662 tok/s) | up to 43× | diverges earliest (480 KB/token) |
| `qwen2` (2× Qwen2.5, 40 GB) | Qwen2.5-Coder-7B | ~1.0× | ~1.1× | — |
| `llama5` (5× Llama, 80 GB) | Llama-3.1-UltraMedical | 7.1× (1.4k→10.2k) | up to 2534× | exceeds the paper's 1.1× / 197× |
| `llama_qwen` (7 models, 2×80 GB) | Llama / Qwen | 3.4× / 2.5× | up to 1329× / 443× | saturation extended 15→25+ / 30→40+ RPS |
| `llama_qwen_ds` (9 models) | DeepSeek / Llama / Qwen | **2.94× / 2.14× / 1.69×** | **1011× / 640× / 294×** | **2.93× / 2.11× / 1.72×; 1052× / 564× / ~300×** |

The recorded 5-Llama run exceeds the paper's printed improvement
(7.1× / 2534× vs 1.1× / 197×); the improvement magnitude depends on the
configured memory budget (see the scenario table in `../README.md`), and the
recorded configuration is fully specified in `llama5_config.sh`.

The owner-tensor runs (same scenarios without the `_variants` suffix) agree
within a few percent (e.g. 9-model DeepSeek 2.89× vs 2.94×), confirming the
ratios depend on the sharing structure, not tensor values. Per-rate tables
are in the benchmark logs; the plots show the full sweeps. The divergence
ordering (DeepSeek → Llama → Qwen, by per-token KV size) and the 9-model
headline ratios reproduce the paper's claims closely. Absolute numbers depend
on the configured deployment budget (ballast) — see the scenario table in
`../README.md`.
