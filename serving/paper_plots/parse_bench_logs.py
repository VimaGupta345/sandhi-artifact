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
    "llama5": [("llama", "llama_5llama_80gb")],
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


def scenario_key(dir_name):
    """Map a results-dir name to a SCENARIOS key (e.g. `llama5_out` -> `llama5`,
    `ds2_variants` -> `ds2`). Longest match wins; None if nothing matches."""
    base = dir_name[: -len("_variants")] if dir_name.endswith("_variants") else dir_name
    if base in SCENARIOS:
        return base
    hits = [k for k in SCENARIOS if k in base]
    return max(hits, key=len) if hits else None

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

    def bench_dir(d):
        return d / "logs" / "benchmarks"

    # Collect scenario dirs: either the root IS a single run dir (a fresh
    # `run_all.sh --run-base-dir` output), or it holds one subdir per scenario
    # (the shipped `../results` layout). When both an owner dir and its
    # `_variants` sibling exist, --runs picks which one; an unpaired dir (a
    # user's own run) is always included.
    if bench_dir(args.results_root).is_dir():
        scen_dirs = [args.results_root]
    else:
        subs = [d for d in sorted(args.results_root.iterdir()) if bench_dir(d).is_dir()]
        names = {d.name for d in subs}
        scen_dirs = []
        for d in subs:
            if d.name.endswith("_variants"):
                if args.runs == "variants":
                    scen_dirs.append(d)
            elif f"{d.name}_variants" in names:
                if args.runs == "owner":
                    scen_dirs.append(d)
            else:
                scen_dirs.append(d)

    order = list(SCENARIOS)

    def sort_key(d):
        key = scenario_key(d.name)
        return (order.index(key) if key else len(order), d.name)

    scen_dirs.sort(key=sort_key)

    rows, unknown = [], []
    for scen_dir in scen_dirs:
        key = scenario_key(scen_dir.name)
        if key is None:
            unknown.append(scen_dir.name)
            continue
        targets = SCENARIOS[key]
        for arm, suffix in ARMS.items():
            for log in sorted(bench_dir(scen_dir).glob(f"{arm}__*.log")):
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
                 f"(--runs {args.runs}); expected logs/benchmarks/"
                 "{baseline,sandhi}__*.log in the root itself or in one "
                 "subdir per scenario (dir names must contain a scenario "
                 f"key: {', '.join(SCENARIOS)})")

    cols = ["scenario", "Request Rate (RPS)", *METRICS]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {len(rows)} rows for "
          f"{len({r['scenario'] for r in rows})} scenarios -> {args.out}")
    for name in unknown:
        print(f"[warn] dir matches no known scenario, skipped: {name}")


if __name__ == "__main__":
    main()
