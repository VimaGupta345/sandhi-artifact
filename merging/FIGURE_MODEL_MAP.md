# Figure → Models mapping (SANDHI reproduction)

Generated from `scripts/run_figures.py` SETS/RUNSETS + `models_download/hf_repos.json`.

## Figure 5 — merge figures (full profile→MICR→analysis pipeline)

| Figure | #models | paper mem% | Models (task) |
|---|---|---|---|
| **fig5a** | 3 | 49.8 | Light-IF (tinyMMLU), T-pro (m_mmlu_ru), MedGo (medqa_4options) |
| **fig5b** | 5 | 35.2 | UltraMed (medqa_4options), multi-truth (truthfulqa_mc2), SafetyGuard (sst2), calme-legal (mmlu_professional_law), Hawkish (mmlu_econometrics) |
| **fig5c** | 7 | 26.7 | UltraMed (medqa_4options), multi-truth (truthfulqa_mc2), SafetyGuard (sst2), calme-legal (mmlu_professional_law), Hawkish (mmlu_econometrics), ds-coder (humaneval), ds-math (gsm8k-cot) |
| **fig5d** | 12 | 38.0 | UltraMed (medqa_4options), multi-truth (truthfulqa_mc2), SafetyGuard (sst2), calme-legal (mmlu_professional_law), Hawkish (mmlu_econometrics), Qwen-Coder (humaneval), Qwen-Math (gsm8k-cot), Light-IF (tinyMMLU), T-pro (m_mmlu_ru), MedGo (medqa_4options), ds-coder (humaneval), ds-math (gsm8k-cot) |

## Figure 6 — serving / deployment pools

_Compositions of existing run-sets (no new profiling/MICR). The pipeline emits the Pareto frontier + operating-point specs; cutoff choice + serving are downstream/out of scope._

| Figure | #models | Composition | Models |
|---|---|---|---|
| **fig6a** | 2 | deepseek(reuse) | ds-coder, ds-math |
| **fig6b** | 2 | qwen25_2 | Qwen-Coder, Qwen-Math |
| **fig6c** | 7 | llama5+deepseek(reuse) | UltraMed, multi-truth, SafetyGuard, calme-legal, Hawkish, ds-coder, ds-math |
| **fig6d** | 5 | llama5 | UltraMed, multi-truth, SafetyGuard, calme-legal, Hawkish |
| **fig6e** | 9 | llama5+qwen25_2+deepseek(reuse) | UltraMed, multi-truth, SafetyGuard, calme-legal, Hawkish, Qwen-Coder, Qwen-Math, ds-coder, ds-math |

## Atomic run-sets

| Run-set | Models (task) |
|---|---|
| `llama5` | UltraMed (medqa_4options), multi-truth (truthfulqa_mc2), SafetyGuard (sst2), calme-legal (mmlu_professional_law), Hawkish (mmlu_econometrics) |
| `qwen25_2` | Qwen-Coder (humaneval), Qwen-Math (gsm8k-cot) |
| `qwen32b3` | Light-IF (tinyMMLU), T-pro (m_mmlu_ru), MedGo (medqa_4options) |
| `deepseek2` | ds-math (gsm8k-cot), ds-coder (humaneval) |
