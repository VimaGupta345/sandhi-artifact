# Paper-style Figure 6 plots

Renders the paper's Figure 6 panels — throughput (top) and P95 TTFT (bottom)
vs request rate, baseline vs sandhi, with per-rate improvement annotations —
from the recorded serving results, in the paper's own style
(`performance_plots.py` is the paper's plotting script; `paper.mplstyle` and
`sandhi_colors.py` are its style assets).

Two commands, run from this directory (needs only pandas + matplotlib — both
in the reference images, or any Python environment with them):

```bash
# 1. parse the recorded benchmark logs into one CSV
python parse_bench_logs.py                # reads ../results/*_variants (reference runs)

# 2. render the paper-style panels
python performance_plots.py               # -> figures/*.pdf
```

`parse_bench_logs.py` emits `all_results.csv` with one row per (scenario, arm,
request rate): output token throughput and median/P95/P99 TTFT, extracted from
the vLLM serving-benchmark logs in `../results/<scenario>/logs/benchmarks/`.
Scenario names follow the paper's convention (e.g. `ds_2ds_40gb_before` /
`_after` for the baseline / sandhi arm). Options: `--runs owner` parses the
owner-tensor runs instead of the merged-weights reference runs;
`--results-root` points at a different results directory (e.g. a fresh run of
`../sandhi_scripts/run_all.sh`).

`performance_plots.py` renders one throughput and one TTFT panel per
deployment scenario (all five scenarios; the multi-model scenarios get one
panel pair per benchmarked family) plus the shared legend, into `figures/`.
Scenarios absent from the CSV are skipped with a message. Each panel prints
its per-rate and average improvement multipliers to stdout — these are the
numbers quoted in §5.3 (e.g. the 9-model DeepSeek panel's top-rate ratios).

The harness also auto-renders simpler comparison plots per run
(`../sandhi_scripts/parse_and_plot_results.py`, invoked by `run_all.sh`);
this directory is for reproducing the panels as they appear in the paper.
