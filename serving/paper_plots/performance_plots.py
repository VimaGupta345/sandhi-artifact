import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from sandhi_colors import systems

# =============================================================================
# Configuration
# =============================================================================

import os

OUT_DIR = "figures"
STYLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper.mplstyle")

FONT_SIZE = 22
TICK_SIZE = 20
ANNOT_SIZE = 18
LEGEND_SIZE = 24

# Performance plot colors - consistent with sandhi_colors.py
perf_systems = {
    "baseline": {
        "name": "Baseline",
        "color": '#3182bd',
        "marker": 's',
        "hatch": "//"
    },
    "sandhi": {
        "name": "Sandhi",
        "color": '#de2d26',
        "marker": 'o',
        "hatch": ""
    }
}
BAR_EDGE_COLOR = "#000000"
BAR_LINEWIDTH = 0.5
GRID_COLOR = "lightgrey"

# Bar geometry
BAR_WIDTH = 0.3


def create_ttft_plot(df, scenario_filter, title, figname):
    """
    Create a TTFT bar plot comparing before/after merging.
    Shows P95 TTFT vs Request Rate (log scale).
    """
    after_data = df[df['scenario'].str.contains(scenario_filter + '_after', na=False) |
                    df['scenario'].str.contains(scenario_filter + '_after0', na=False)]
    before_data = df[df['scenario'].str.contains(scenario_filter + '_before', na=False)]

    after_data = after_data.sort_values('Request Rate (RPS)')
    before_data = before_data.sort_values('Request Rate (RPS)')

    merged = pd.merge(
        before_data[['Request Rate (RPS)', 'P95 TTFT (ms)']],
        after_data[['Request Rate (RPS)', 'P95 TTFT (ms)']],
        on='Request Rate (RPS)',
        suffixes=('_baseline', '_sandhi')
    )
    merged['multiplier'] = (merged['P95 TTFT (ms)_baseline'] / merged['P95 TTFT (ms)_sandhi'])

    print(f"\n{title} - P95 TTFT Multiplier (Baseline/Sandhi, >1 means Sandhi is better):")
    for _, row in merged.iterrows():
        print(f"  RPS {row['Request Rate (RPS)']:.0f}: {row['multiplier']:.2f}x")
    print(f"  Average: {merged['multiplier'].mean():.2f}x")

    with plt.style.context(STYLE_PATH):
        fig, ax = plt.subplots(figsize=(7, 4.5), layout='constrained')

        rps = merged['Request Rate (RPS)']
        x = np.arange(len(rps))

        baseline_bars = ax.bar(
            x - BAR_WIDTH/2,
            merged['P95 TTFT (ms)_baseline'],
            width=BAR_WIDTH,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_LINEWIDTH,
            label='Baseline',
            color=perf_systems['baseline']['color'],
            hatch=perf_systems['baseline']['hatch']
        )

        sandhi_bars = ax.bar(
            x + BAR_WIDTH/2,
            merged['P95 TTFT (ms)_sandhi'],
            width=BAR_WIDTH,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_LINEWIDTH,
            label='Sandhi',
            color=perf_systems['sandhi']['color'],
            hatch=perf_systems['sandhi']['hatch']
        )

        ax.set_yscale('log')

        # Compute ylim with headroom for rotated annotations on log scale
        ymax = max(
            merged['P95 TTFT (ms)_baseline'].max(),
            merged['P95 TTFT (ms)_sandhi'].max()
        )
        ymin = min(
            merged['P95 TTFT (ms)_baseline'].min(),
            merged['P95 TTFT (ms)_sandhi'].min()
        )
        # ~1.5 decades of headroom in log space for rotated text
        log_top = np.log10(ymax) + 0.4 * (np.log10(ymax) - np.log10(ymin))
        ax.set_ylim(bottom=ymin * 0.7, top=10 ** log_top)

        # Multiplier annotation above baseline bars
        for i, bar in enumerate(baseline_bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height * 1.15,
                f"{merged['multiplier'].iloc[i]:.1f}x",
                ha='center',
                va='bottom',
                fontsize=ANNOT_SIZE,
                fontweight='bold',
                rotation=90,
            )

        ax.set_xlabel('Request Rate (RPS)', fontsize=FONT_SIZE)
        ax.set_ylabel('P95 TTFT (ms)', fontsize=FONT_SIZE)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(v)}" for v in rps])

        ax.tick_params(axis='both', labelsize=TICK_SIZE)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color=GRID_COLOR, linestyle='dashed', alpha=0.7)

        fig.savefig(f'{OUT_DIR}/{figname}.pdf', bbox_inches='tight')
        print(f"Saved {OUT_DIR}/{figname}.pdf")
        plt.close(fig)


def create_throughput_plot(df, scenario_filter, title, figname):
    """
    Create a throughput bar plot comparing before/after merging.
    Shows Output Token Throughput vs Request Rate (linear scale).
    """
    after_data = df[df['scenario'].str.contains(scenario_filter + '_after', na=False) |
                    df['scenario'].str.contains(scenario_filter + '_after0', na=False)]
    before_data = df[df['scenario'].str.contains(scenario_filter + '_before', na=False)]

    after_data = after_data.sort_values('Request Rate (RPS)')
    before_data = before_data.sort_values('Request Rate (RPS)')

    merged = pd.merge(
        before_data[['Request Rate (RPS)', 'Output Token Throughput (tok/s)']],
        after_data[['Request Rate (RPS)', 'Output Token Throughput (tok/s)']],
        on='Request Rate (RPS)',
        suffixes=('_baseline', '_sandhi')
    )
    merged['multiplier'] = merged['Output Token Throughput (tok/s)_sandhi'] / merged['Output Token Throughput (tok/s)_baseline']
    print(f"\n{title} - Throughput Multiplier (Sandhi/Baseline, >1 means Sandhi is better):")
    for _, row in merged.iterrows():
        print(f"  RPS {row['Request Rate (RPS)']:.0f}: {row['multiplier']:.2f}x")
    print(f"  Average: {merged['multiplier'].mean():.2f}x")

    with plt.style.context(STYLE_PATH):
        fig, ax = plt.subplots(figsize=(7, 4.5), layout='constrained')

        rps = merged['Request Rate (RPS)']
        x = np.arange(len(rps))

        baseline_bars = ax.bar(
            x - BAR_WIDTH/2,
            merged['Output Token Throughput (tok/s)_baseline'],
            width=BAR_WIDTH,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_LINEWIDTH,
            label='Baseline',
            color=perf_systems['baseline']['color'],
            hatch=perf_systems['baseline']['hatch']
        )

        sandhi_bars = ax.bar(
            x + BAR_WIDTH/2,
            merged['Output Token Throughput (tok/s)_sandhi'],
            width=BAR_WIDTH,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_LINEWIDTH,
            label='Sandhi',
            color=perf_systems['sandhi']['color'],
            hatch=perf_systems['sandhi']['hatch']
        )

        # 35% headroom for rotated annotations on linear scale
        ymax = max(
            merged['Output Token Throughput (tok/s)_baseline'].max(),
            merged['Output Token Throughput (tok/s)_sandhi'].max()
        )
        ax.set_ylim(bottom=0, top=ymax * 1.35)

        # Multiplier annotation above sandhi bars
        for i, bar in enumerate(sandhi_bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height * 1.02,
                f"{merged['multiplier'].iloc[i]:.2f}x",
                ha='center',
                va='bottom',
                fontsize=ANNOT_SIZE,
                fontweight='bold',
                rotation=90,
            )

        ax.set_xlabel("")
        ax.set_ylabel('Throughput (tok/s)', fontsize=FONT_SIZE)

        ax.set_xticks(x)
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax.tick_params(axis='y', labelsize=TICK_SIZE)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color=GRID_COLOR, linestyle='dashed', alpha=0.7)

        fig.savefig(f'{OUT_DIR}/{figname}.pdf', bbox_inches='tight')
        print(f"Saved {OUT_DIR}/{figname}.pdf")
        plt.close(fig)


def create_legend():
    """Create a standalone legend figure."""
    with plt.style.context(STYLE_PATH):
        fig, ax = plt.subplots(figsize=(10, 1))
        ax.axis('off')

        legend_elements = [
            Patch(
                facecolor=perf_systems['baseline']['color'],
                edgecolor=BAR_EDGE_COLOR,
                hatch=perf_systems['baseline']['hatch'],
                label=perf_systems['baseline']['name'],
            ),
            Patch(
                facecolor=perf_systems['sandhi']['color'],
                edgecolor=BAR_EDGE_COLOR,
                hatch=perf_systems['sandhi']['hatch'],
                label=perf_systems['sandhi']['name'],
            )
        ]

        ax.legend(handles=legend_elements, loc='center',
                  ncols=2, fontsize=LEGEND_SIZE, frameon=False)

        fig.savefig(f'{OUT_DIR}/perf_legend.pdf', bbox_inches='tight')
        print(f"Saved {OUT_DIR}/perf_legend.pdf")
        plt.close(fig)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Render the paper-style Figure 6 panels (throughput + P95 "
                    "TTFT vs request rate, baseline vs sandhi) from a results "
                    "table produced by results_to_table.py.")
    ap.add_argument("--table", default="all_results.csv",
                    help="Parsed results CSV from parse_bench_logs.py. "
                         "Default: all_results.csv")
    ap.add_argument("--out-dir", default=OUT_DIR,
                    help=f"Output directory for PDFs. Default: {OUT_DIR}")
    args = ap.parse_args()

    if not os.path.exists(args.table):
        ap.error(f"results CSV not found: {args.table} — generate it first: "
                 "python parse_bench_logs.py")
    OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(args.table)

    create_legend()

    experiments = [
        ('ds_2ds_40gb', '2-DS (A100 40GB)', '2ds'),
        ('qwen_2qwen_40gb', '2-Qwen (A100 40GB)', '2qwen'),
        ('llama_5llama_141gb', '5-Llama (H200 141GB)', '5llama'),
        ('llama_qwen-llama_141gb_tp2', '7-model Llama (2 H200 141GB)', '7model_llama'),
        ('qwen_qwen-llama_141gb_tp2', '7-model Qwen (2 H200 141GB)', '7model_qwen'),
        ('ds_ds-llama-qwen_141gb_tp2', '9-model DS (2 H200 141GB)', '9model_ds'),
        ('llama_ds-llama-qwen_141gb_tp2', '9-model Llama (2 H200 141GB)', '9model_llama'),
        ('qwen_ds-llama-qwen_141gb_tp2', '9-model Qwen (2 H200 141GB)', '9model_qwen'),
    ]

    for scenario_filter, title, figname in experiments:
        if not df['scenario'].str.contains(scenario_filter, na=False).any():
            print(f"[skip] no rows for {scenario_filter} in {args.table}")
            continue
        create_ttft_plot(df, scenario_filter, title, f'{figname}_ttft')
        create_throughput_plot(df, scenario_filter, title, f'{figname}_throughput')

    print("\nPerformance plots generated!")
