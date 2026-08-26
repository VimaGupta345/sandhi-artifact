# Figure 5 (accuracy + memory-savings) plots

Regenerates the paper's Figure 5 panels: per-configuration grouped-bar accuracy
(SANDHI vs No-merge / Full-merge / LoRA) plus the memory-savings panel.

## Run (in the reference container — no extra deps)
```bash
CODE=/path/to/this/repo
docker run --rm --user "$(id -u):$(id -g)" -e USER="$(id -un)" -e HOME=/tmp \
  -v "$CODE":/workspace/merge_tools -w /workspace/merge_tools/plots/source \
  merge-tools:reference python accuracy_memory_figures.py
```
Outputs land in `plots/figures/*.pdf` (600 DPI, Type-42 fonts). No
dependencies beyond the image's pandas/matplotlib are required.

## Config → figure map
| script config | paper panel | pipeline set |
|---|---|---|
| `3-Qwen-32B` | **Fig 5a** | fig5a (`qwen32b3`) |
| `5-llama` | **Fig 5b** | fig5b (`llama5`) |
| `7-5-llama-2-DS` | **Fig 5c** | fig5c (`llama5`+`deepseek2`) |
| `12-model` | **Fig 5d** | fig5d (all 12) |

The script renders exactly these four panels (plus the legend) — the paper's
only accuracy/memory bar charts. The Figure 6 pools have no bar-chart panel:
their memory numbers are the pipeline's (`results/FIGURE_COMPOSITIONS.md` and
`results/fig6*/report.csv`), and their serving plots are rendered by the
harness in `../../serving/` (recorded in `../../serving/results/`).

## Files
- `source/accuracy_memory_figures.py` — the plotting script.
- `source/sandhi_colors.py`, `source/paper.mplstyle` — colors + publication style.
- `data/all_models_final.csv` — SANDHI accuracy deltas + memory saved per
  (config, model); `data/memory_savings.csv`, `data/vllm-no-merge.csv` — memory
  and no-merge baselines.
- `data/full_merge/*.csv` — Full-merge (multi-slerp) baselines; `data/lora/*.csv`
  — LoRA-adapter baselines.

## Data sources and operating points
These figures reproduce the paper's *rendered* Figure 5, so the two tables
deliberately sit at the points the paper reports:

- `data/memory_savings.csv` holds the paper's reported memory numbers at the
  paper's operating point and denominators (e.g. 5-llama 26.4 GB / 35.2%,
  3-Qwen-32B 92.6 GB / 49.8% of the paper's 186 GB total).
- `data/all_models_final.csv` holds the artifact runs' per-model accuracy
  deltas (regenerated from each set's `analysis/<set>/report.csv` by
  `scripts/build_plot_data.py`), measured at the artifact's chosen operating
  point — which can free *more* memory than the paper's bar (e.g. 5-llama
  33.9 GB at Cpm vs the paper's 26.4 GB).

The artifact's own memory numbers at every operating point are in each set's
`report.csv` and `results/FIGURE_COMPOSITIONS.md`; the denominator
reconciliation for Fig 5a is in `../results/fig5a/README.md`.
