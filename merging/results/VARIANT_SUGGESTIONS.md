# Automated variant suggestions — Figure 5 sets

Four auto-computed operating points per set (from each set's `report.csv`). **Kpm (knee) is the recommended balance.** Savings = distinct-tensor memory reduction; drop = MICR M-split per-model accuracy drop (confirm on full set before publishing).

## fig5a — 3x Qwen3-32B  ·  paper mem 49.8%

| variant | rule | savings | worst drop |
|---|---|---|---|
| **Bpm** | max savings, per-model ≤1% | 25.5% | +0.40 |
| **Kpm** | knee — savings/accuracy sweet spot (recommended) | 45.6% | +1.87 |
| **Cpm** | max savings, per-model ≤2% | 45.6% | +1.87 |
| **P** | closest to paper's reported memory | 49.8% | +3.74 |

_Kpm per-model cutoffs: Light=c129, T=c126, MedGo=c52_

## fig5b — 5x Llama-3.1-8B  ·  paper mem 35.2%

| variant | rule | savings | worst drop |
|---|---|---|---|
| **Bpm** | max savings, per-model ≤1% | 43.2% | +0.65 |
| **Kpm** | knee — savings/accuracy sweet spot (recommended) | 43.2% | +0.65 |
| **Cpm** | max savings, per-model ≤2% | 43.3% | +1.75 |
| **P** | closest to paper's reported memory | 35.3% | +1.25 |

_Kpm per-model cutoffs: Llama=c52, Llama=c52, Llama=c52, calme=c52, Llama=c24_

## llama5_deepseek2 — 5 Llama + 2 DeepSeek (cross)  ·  paper mem 26.7%

| variant | rule | savings | worst drop |
|---|---|---|---|
| **Bpm** | max savings, per-model ≤1% | 33.1% | +0.65 |
| **Kpm** | knee — savings/accuracy sweet spot (recommended) | 33.1% | +0.65 |
| **Cpm** | max savings, per-model ≤2% | 33.5% | +1.80 |
| **P** | closest to paper's reported memory | 27.1% | +5.60 |

_Kpm per-model cutoffs: Llama=c40, Llama=c55, Llama=c55, calme=c55, Llama=c22, deepseek=c18, deepseek=c14_

## qwen25_2 — 2x Qwen2.5-7B (within)  ·  (in pool-12)

| variant | rule | savings | worst drop |
|---|---|---|---|
| **Bpm** | max savings, per-model ≤1% | 1.7% | +0.00 |
| **Kpm** | knee — savings/accuracy sweet spot (recommended) | 1.8% | +1.21 |
| **Cpm** | max savings, per-model ≤2% | 1.8% | +1.21 |

_Kpm per-model cutoffs: Qwen2.5=c11, Qwen2.5=c13_

## deepseek2 — 2x DeepSeek-7B (within)  ·  (in pool-12)

| variant | rule | savings | worst drop |
|---|---|---|---|
| **Bpm** | max savings, per-model ≤1% | 4.0% | +0.70 |
| **Kpm** | knee — savings/accuracy sweet spot (recommended) | 3.8% | +0.40 |
| **Cpm** | max savings, per-model ≤2% | 4.4% | +1.50 |

_Kpm per-model cutoffs: deepseek=c10, deepseek=c15_

