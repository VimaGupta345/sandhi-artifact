#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RPS_PATTERN = re.compile(r"Request rate configured \(RPS\):\s+([\d.]+)")
RPS_FALLBACK_PATTERN = re.compile(r"RPS=([\d.]+)")
P95_TTFT_PATTERN = re.compile(r"P95 TTFT \(ms\):\s+([\d.]+)")
MEAN_TTFT_PATTERN = re.compile(r"Mean TTFT \(ms\):\s+([\d.]+)")
TOKEN_THROUGHPUT_PATTERN = re.compile(r"Output token throughput \(tok/s\):\s+([\d.]+)")
MEAN_ITL_PATTERN = re.compile(r"Mean ITL \(ms\):\s+([\d.]+)")
P95_ITL_PATTERN = re.compile(r"P95 ITL \(ms\):\s+([\d.]+)")
PORT_PATTERN = re.compile(r"Port:\s+(\d+)")
MODEL_PATTERN = re.compile(r"Model:\s+(.+)")
TARGET_PATTERN = re.compile(r"Target:\s+(.+)")

SYSTEMS = {
    "baseline": {"label": "Baseline", "color": "#3182bd", "hatch": "//", "marker": "s"},
    "sandhi": {"label": "Sandhi", "color": "#de2d26", "hatch": "", "marker": "o"},
}

BAR_WIDTH = 0.36
BAR_EDGE_COLOR = "black"
BAR_LINEWIDTH = 1.0
GRID_COLOR = "lightgrey"
FONT_SIZE = 14
TICK_SIZE = 12
ANNOT_SIZE = 11
LEGEND_SIZE = 12


def parse_log(log_path: Path):
    records = []
    current_port = None
    current_model = None
    current_rps = None
    current_target = None
    current_metrics = {}
    mode_name = log_path.stem.split("__", 1)[0]

    def flush_current():
        nonlocal current_rps, current_metrics
        if current_port is None or current_model is None or current_rps is None:
            current_metrics = {}
            return
        if "p95_ttft_ms" not in current_metrics or "token_throughput_tok_s" not in current_metrics:
            current_metrics = {}
            return
        record = {
            "mode": mode_name,
            "port": current_port,
            "model": current_model,
            "target": current_target,
            "rps": current_rps,
            "p95_ttft_ms": current_metrics.get("p95_ttft_ms"),
            "mean_ttft_ms": current_metrics.get("mean_ttft_ms"),
            "token_throughput_tok_s": current_metrics.get("token_throughput_tok_s"),
            "mean_itl_ms": current_metrics.get("mean_itl_ms"),
            "p95_itl_ms": current_metrics.get("p95_itl_ms"),
        }
        records.append(record)
        current_metrics = {}

    for raw_line in log_path.read_text().splitlines():
        line = raw_line.strip()

        port_match = PORT_PATTERN.search(line)
        if port_match:
            flush_current()
            current_port = int(port_match.group(1))
            continue

        model_match = MODEL_PATTERN.search(line)
        if model_match:
            current_model = model_match.group(1).strip()
            continue

        target_match = TARGET_PATTERN.search(line)
        if target_match:
            current_target = target_match.group(1).strip()
            continue

        rps_match = RPS_PATTERN.search(line) or RPS_FALLBACK_PATTERN.search(line)
        if rps_match:
            flush_current()
            current_rps = float(rps_match.group(1))
            continue

        p95_ttft_match = P95_TTFT_PATTERN.search(line)
        if p95_ttft_match:
            current_metrics["p95_ttft_ms"] = float(p95_ttft_match.group(1))
            continue

        mean_ttft_match = MEAN_TTFT_PATTERN.search(line)
        if mean_ttft_match:
            current_metrics["mean_ttft_ms"] = float(mean_ttft_match.group(1))
            continue

        throughput_match = TOKEN_THROUGHPUT_PATTERN.search(line)
        if throughput_match:
            current_metrics["token_throughput_tok_s"] = float(throughput_match.group(1))
            continue

        mean_itl_match = MEAN_ITL_PATTERN.search(line)
        if mean_itl_match:
            current_metrics["mean_itl_ms"] = float(mean_itl_match.group(1))
            continue

        p95_itl_match = P95_ITL_PATTERN.search(line)
        if p95_itl_match:
            current_metrics["p95_itl_ms"] = float(p95_itl_match.group(1))
            continue

    flush_current()
    return records


def slugify(value: str):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def build_mode_lookup(records, metric_key: str):
    return {
        mode: sorted(
            [record for record in records if record["mode"] == mode and record.get(metric_key) is not None],
            key=lambda record: record["rps"],
        )
        for mode in SYSTEMS
    }


def create_ttft_plot(records, title: str, output_path: Path):
    mode_records = build_mode_lookup(records, "p95_ttft_ms")
    baseline_records = mode_records["baseline"]
    sandhi_records = mode_records["sandhi"]
    if not baseline_records or not sandhi_records:
        return

    rps = [record["rps"] for record in baseline_records]
    x = np.arange(len(rps))
    baseline_values = [record["p95_ttft_ms"] for record in baseline_records]
    sandhi_values = [record["p95_ttft_ms"] for record in sandhi_records]
    multipliers = [base / sandhi for base, sandhi in zip(baseline_values, sandhi_values)]

    print(f"\n{title} - P95 TTFT Multiplier (Baseline/Sandhi, >1 means Sandhi is better):")
    for rps_value, multiplier in zip(rps, multipliers):
        print(f"  RPS {rps_value:.0f}: {multiplier:.2f}x")
    print(f"  Average: {sum(multipliers) / len(multipliers):.2f}x")

    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")

    ax.plot(
        x,
        baseline_values,
        marker=SYSTEMS["baseline"]["marker"],
        markersize=10,
        linewidth=2.5,
        color=SYSTEMS["baseline"]["color"],
        label="Baseline",
    )
    ax.plot(
        x,
        sandhi_values,
        marker=SYSTEMS["sandhi"]["marker"],
        markersize=10,
        linewidth=2.5,
        color=SYSTEMS["sandhi"]["color"],
        label="Sandhi",
    )

    ax.set_yscale("log")
    ymax = max(max(baseline_values), max(sandhi_values))
    ymin = min(min(baseline_values), min(sandhi_values))
    log_top = np.log10(ymax) + 0.4 * (np.log10(ymax) - np.log10(ymin))
    ax.set_ylim(bottom=ymin * 0.7, top=10 ** log_top)

    arrow_props = {
        "arrowstyle": "<->",
        "color": "gray",
        "linestyle": "-",
        "linewidth": 1.5,
        "mutation_scale": 12,
        "shrinkA": 0,
        "shrinkB": 0,
    }

    for i, (y_base, y_sandhi, multiplier) in enumerate(zip(baseline_values, sandhi_values, multipliers)):
        ax.annotate(
            "",
            xytext=(x[i], y_sandhi),
            xy=(x[i], y_base),
            arrowprops=arrow_props,
        )

        text_y = np.sqrt(y_base * y_sandhi)
        ax.annotate(
            f"{multiplier:.1f}x",
            xy=(x[i], text_y),
            xytext=(2, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=ANNOT_SIZE,
            fontweight="bold",
            rotation=90,
        )

    ax.set_xlabel("Request Rate (RPS)", fontsize=FONT_SIZE)
    ax.set_ylabel("P95 TTFT (ms)", fontsize=FONT_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}" for v in rps])
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID_COLOR, linestyle="dashed", alpha=0.7)
    ax.legend(fontsize=LEGEND_SIZE, frameon=False)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def create_throughput_plot(records, title: str, output_path: Path):
    mode_records = build_mode_lookup(records, "token_throughput_tok_s")
    baseline_records = mode_records["baseline"]
    sandhi_records = mode_records["sandhi"]
    if not baseline_records or not sandhi_records:
        return

    rps = [record["rps"] for record in baseline_records]
    x = np.arange(len(rps))
    baseline_values = [record["token_throughput_tok_s"] for record in baseline_records]
    sandhi_values = [record["token_throughput_tok_s"] for record in sandhi_records]
    multipliers = [sandhi / base for base, sandhi in zip(baseline_values, sandhi_values)]

    print(f"\n{title} - Throughput Multiplier (Sandhi/Baseline, >1 means Sandhi is better):")
    for rps_value, multiplier in zip(rps, multipliers):
        print(f"  RPS {rps_value:.0f}: {multiplier:.2f}x")
    print(f"  Average: {sum(multipliers) / len(multipliers):.2f}x")

    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")

    ax.bar(
        x - BAR_WIDTH / 2,
        baseline_values,
        width=BAR_WIDTH,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_LINEWIDTH,
        color=SYSTEMS["baseline"]["color"],
        hatch=SYSTEMS["baseline"]["hatch"],
        label="Baseline",
    )
    sandhi_bars = ax.bar(
        x + BAR_WIDTH / 2,
        sandhi_values,
        width=BAR_WIDTH,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_LINEWIDTH,
        color=SYSTEMS["sandhi"]["color"],
        hatch=SYSTEMS["sandhi"]["hatch"],
        label="Sandhi",
    )

    ymax = max(max(baseline_values), max(sandhi_values))
    ax.set_ylim(bottom=0, top=ymax * 1.35)

    for i, bar in enumerate(sandhi_bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height * 1.02,
            f"{multipliers[i]:.2f}x",
            ha="center",
            va="bottom",
            fontsize=ANNOT_SIZE,
            fontweight="bold",
            rotation=90,
        )

    ax.set_xlabel("Request Rate (RPS)", fontsize=FONT_SIZE)
    ax.set_ylabel("Throughput (tok/s)", fontsize=FONT_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}" for v in rps])
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID_COLOR, linestyle="dashed", alpha=0.7)
    ax.legend(fontsize=LEGEND_SIZE, frameon=False)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-log-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    bench_log_dir = Path(args.bench_log_dir)
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    for log_path in sorted(bench_log_dir.glob("*.log")):
        all_records.extend(parse_log(log_path))

    if not all_records:
        raise SystemExit(f"No benchmark results found in {bench_log_dir}")

    all_records.sort(key=lambda record: (record["model"], record["mode"], record["rps"]))
    models = sorted({record["model"] for record in all_records})
    for model in models:
        model_records = [record for record in all_records if record["model"] == model]
        model_slug = slugify(model)
        create_throughput_plot(model_records, model, plots_dir / f"{model_slug}_throughput_vs_rps.png")
        create_ttft_plot(model_records, model, plots_dir / f"{model_slug}_p95_ttft_vs_rps.png")

    print(f"Wrote plots to {plots_dir}")


if __name__ == "__main__":
    main()
