# Figure 5 (accuracy + memory-savings) plots

Regenerates the paper's Figure 5 panels: per-configuration grouped-bar accuracy
(SANDHI vs No-merge / Full-merge / LoRA) plus the memory-savings panel.

## Run (in the reference container — no extra deps)
```bash
CODE=/path/to/artifacts_worktree
docker run --rm --user "$(id -u):$(id -g)" -e USER="$(id -un)" -e HOME=/tmp \
  -v "$CODE":/workspace/merge_tools -w /workspace/merge_tools/plots/source \
  merge-tools:reference python accuracy_memory_figures.py
```
Outputs land in `plots/figures/*.pdf` (600 DPI, Type-42 fonts). The DeepSeek
full-merge baseline was converted from `.xlsx` to `.csv`, so **no `openpyxl`**
(or any dep beyond the image's pandas/matplotlib) is required.

## Config → figure map
| script config | paper panel | pipeline set |
|---|---|---|
| `3-Qwen-32B` | **Fig 5a** | fig5a (`qwen32b3`) |
| `5-llama` | **Fig 5b** | fig5b (`llama5`) |
| `7-5-llama-2-DS` | **Fig 5c** | fig5c (`llama5`+`deepseek2`) |
| `12-model` | **Fig 5d** | fig5d (all 12) |
| `2-DS`, `2-Qwen-7B`, `7-5-llama-2-qwen`, `9-model` | Fig 6 pools | fig6a/6b/6e … |

The script emits every config in one pass; the four above are the Figure 5
panels.

## Files
- `source/accuracy_memory_figures.py` — the plotting script (canonical, per the
  upstream `sandhi-plots` INSTRUCTIONS).
- `source/sandhi_colors.py`, `source/paper.mplstyle` — colors + publication style.
- `data/all_models_final.csv` — SANDHI accuracy deltas + memory saved per
  (config, model); `data/memory_savings.csv`, `data/vllm-no-merge.csv` — memory
  and no-merge baselines.
- `data/full_merge/*.csv` — Full-merge (multi-slerp) baselines; `data/lora/*.csv`
  — LoRA-adapter baselines.

## Provenance
The `data/` tables are the paper's aggregated accuracy/memory numbers. The
pipeline (`scripts/run_figures.py`) produces the underlying per-set values —
each set's `analysis/<set>/report.csv` (savings % + per-model drops) is the
source for the corresponding rows in `all_models_final.csv` / `memory_savings.csv`.
So the pipeline generates the numbers; this script renders them into the paper
figure.
