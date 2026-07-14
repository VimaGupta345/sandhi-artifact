# Figure pools composed from available base runs

Additive composition of the 5 atoms (only `5llama+ds` is joint / non-additive). Savings = (Σ atom freed)/(Σ atom size).

## Detected base runs (atoms)

- ✅ **5llama** — 78.4 GB — `runs/llama5/analysis/llama5/report.csv`
- ✅ **2ds** — 27.0 GB — `runs/deepseek2/analysis/deepseek2/report.csv`
- ✅ **2qwen** — 29.8 GB — `runs/qwen25_2/analysis/qwen25_2/report.csv`
- ✅ **3qwen32b** — 192.0 GB — `runs/qwen32b3/analysis/fig5a/report.csv`
- ✅ **5llama+ds** — 105.4 GB — `runs/llama5_deepseek2/analysis/llama5_deepseek2/report.csv`

## Figure pools

| figure | atoms | status | pool GB | Bpm ≤1% | Cpm ≤2% | Kpm knee | paper |
|---|---|---|---|---|---|---|---|
| **fig5a (3, qwen32b)** | 3qwen32b | ✅ | 192 | 25.5% | 45.6% | 45.6% | 49.8% |
| **fig5b (5, llama)** | 5llama | ✅ | 78 | 43.2% | 43.3% | 43.2% | 35.2% |
| **fig5c (7, llama+ds)** | 5llama+ds | ✅ | 105 | 33.1% | 33.5% | 33.1% | 26.7% |
| **fig5d (12, all)** | 5llama+ds+2qwen+3qwen32b | ✅ | 327 | 25.8% | 37.7% | 37.6% | 38.0% |
| **fig6a (2, ds)** | 2ds | ✅ | 27 | 4.0% | 4.4% | 3.8% | — |
| **fig6b (2, qwen)** | 2qwen | ✅ | 30 | 1.7% | 1.8% | 1.8% | — |
| **fig6c (7, llama+ds)** | 5llama+ds | ✅ | 105 | 33.1% | 33.5% | 33.1% | — |
| **fig6d (5, llama)** | 5llama | ✅ | 78 | 43.2% | 43.3% | 43.2% | — |
| **fig6e (9, llama+ds+qwen)** | 5llama+ds+2qwen | ✅ | 135 | 26.2% | 26.5% | 26.2% | — |
