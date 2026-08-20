#!/usr/bin/env python3
"""Parse recorded serving benchmark logs into the CSV the paper-style
Figure 6 renderer (performance_plots.py) consumes.

Walks each scenario's benchmark logs (baseline and sandhi arms), extracts the
per-request-rate metrics printed by vLLM's serving benchmark, and emits one CSV
row per (scenario, arm, request rate):

    scenario,Request Rate (RPS),Output Token Throughput (tok/s),
    Median TTFT (ms),P95 TTFT (ms),P99 TTFT (ms)

Scenario names follow the paper's convention (<model>_<pool>_<budget>[_tp2]
with a _before/_after suffix for the baseline/sandhi arm), so the renderer's
filters match without further mapping. If a benchmark repeats a request rate
(a rerun within the same log), the last occurrence wins.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

# results dir -> [(benchmark-log substring, paper scenario prefix)]
SCENARIOS = {
    "ds2": [("deepseek", "ds_2ds_40gb")],
    "qwen2": [("qwen2_5", "qwen_2qwen_40gb")],
    "llama5": [("llama", "llama_5llama_141gb")],
    "llama_qwen": [
        ("llama", "llama_qwen-llama_141gb_tp2"),
        ("qwen2_5", "qwen_qwen-llama_141gb_tp2"),
    ],
    "llama_qwen_ds": [
        ("deepseek", "ds_ds-llama-qwen_141gb_tp2"),
        ("llama", "llama_ds-llama-qwen_141gb_tp2"),
        ("qwen2_5", "qwen_ds-llama-qwen_141gb_tp2"),
    ],
}

ARMS = {"baseline": "before", "sandhi": "after"}

METRICS = {
    "Output Token Throughput (tok/s)": r"Output token throughput \(tok/s\):\s*([\d.]+)",
    "Median TTFT (ms)": r"Median TTFT \(ms\):\s*([\d.]+)",
    "P95 TTFT (ms)": r"P95 TTFT \(ms\):\s*([\d.]+)",
    "P99 TTFT (ms)": r"P99 TTFT \(ms\):\s*([\d.]+)",
}


def parse_log(path: Path):
    """Return {rate: {metric: value}} for one benchmark log (last block wins)."""
    text = path.read_text(errors="ignore")
    blocks = re.split(r"Traffic request rate:\s*([\d.]+)", text)[1:]
    rates = {}
    for rate, block in zip(blocks[0::2], blocks[1::2]):
        row = {}
        for col, pat in METRICS.items():
            m = re.search(pat, block)
            if m:
                row[col] = float(m.group(1))
        if len(row) == len(METRICS):
            rates[float(rate)] = row
    return rates


def main():
    ap = argparse.ArgumentParser(
        description="Parse recorded serving benchmark logs into the CSV that "
                    "performance_plots.py consumes.")
    ap.add_argument("--results-root", default="../results", type=Path,
                    help="Directory holding the recorded scenario dirs "
                         "(default: ../results)")
    ap.add_argument("--runs", choices=["variants", "owner"], default="variants",
                    help="Which recorded runs to use: 'variants' = the "
                         "reference runs serving the materialized merged "
                         "weights (default), 'owner' = the owner-tensor runs.")
    ap.add_argument("--out", default="all_results.csv",
                    help="Output CSV path (default: all_results.csv)")
    args = ap.parse_args()

    if not args.results_root.is_dir():
        ap.error(f"results root not found: {args.results_root}")

    rows, missing = [], []
    for base, targets in SCENARIOS.items():
        scen_dir = args.results_root / (f"{base}_variants" if args.runs == "variants" else base)
        bench_dir = scen_dir / "logs" / "benchmarks"
        if not bench_dir.is_dir():
            missing.append(str(scen_dir))
            continue
        for arm, suffix in ARMS.items():
            for log in sorted(bench_dir.glob(f"{arm}__*.log")):
                match = next((prefix for sub, prefix in targets
                              if sub in log.name and
                              ("deepseek" not in log.name or sub == "deepseek")), None)
                if match is None:
                    continue
                for rate, metrics in sorted(parse_log(log).items()):
                    rows.append({"scenario": f"{match}_{suffix}",
                                 "Request Rate (RPS)": rate, **metrics})

    if not rows:
        sys.exit(f"no benchmark data found under {args.results_root} "
                 f"(--runs {args.runs}); expected <scenario>/logs/benchmarks/"
                 "{baseline,sandhi}__*.log")

    cols = ["scenario", "Request Rate (RPS)", *METRICS]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {len(rows)} rows for "
          f"{len({r['scenario'] for r in rows})} scenarios -> {args.out}")
    for m in missing:
        print(f"[warn] scenario dir missing, skipped: {m}")


if __name__ == "__main__":
    main()
