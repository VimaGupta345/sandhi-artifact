#!/usr/bin/env python3
"""Automated variant suggestion for the Figure-5 sets.

Reads each set's analysis `report.csv` (produced by build_operating_points.py
--per-model --paper-mem) and emits, per set, the four auto-computed operating
points as suggested variants:

  Bpm  -- most memory savings within a 1% per-model accuracy drop
  Cpm  -- most memory savings within a 2% per-model accuracy drop
  Kpm  -- the savings/accuracy KNEE (recommended balance; <=2% envelope)
  P    -- the point whose savings is closest to the paper's reported number
          (only for sets with a paper figure)

Each per-model point carries its own {model: cutoff} (a fragile member no longer
caps a robust one). Savings come from the distinct-tensor model in
build_operating_points.py; accuracy is the MICR M-split drop (full-set eval is a
separate confirmation step). No eval and no GPU -- pure read of report.csv, so
reviewers reproduce it from the shipped CSVs or after a rerun.

Usage: python scripts/suggest_variants.py [--results DIR] [--out FILE]
"""
import argparse
import csv
import os

# figure set -> (results subdir, human label, paper memory %)
SETS = [
    ("fig5a", "3x Qwen3-32B", 49.8),
    ("fig5b", "5x Llama-3.1-8B", 35.2),
    ("llama5_deepseek2", "5 Llama + 2 DeepSeek (cross)", 26.7),
    ("qwen25_2", "2x Qwen2.5-7B (within)", None),
    ("deepseek2", "2x DeepSeek-7B (within)", None),
]
ORDER = ["Bpm", "Kpm", "Cpm", "P"]
RULE = {
    "Bpm": "max savings, per-model ≤1%",
    "Kpm": "knee — savings/accuracy sweet spot (recommended)",
    "Cpm": "max savings, per-model ≤2%",
    "P": "closest to paper's reported memory",
}


def load(path):
    with open(path, newline="") as f:
        return {r["point"]: r for r in csv.DictReader(f)}


def worst_drop(r):
    return max(float(v) for k, v in r.items() if k.endswith("__drop"))


def per_model_cuts(r):
    return {k[:-8]: r[k] for k in r if k.endswith("__cutoff")}


def render(results_dir):
    out = ["# Automated variant suggestions — Figure 5 sets", "",
           "Four auto-computed operating points per set (from each set's "
           "`report.csv`). **Kpm (knee) is the recommended balance.** Savings = "
           "distinct-tensor memory reduction; drop = MICR M-split per-model "
           "accuracy drop (confirm on full set before publishing).", ""]
    for setname, label, paper in SETS:
        rep = os.path.join(results_dir, setname, "report.csv")
        if not os.path.exists(rep):
            out += [f"## {setname} — {label}", "_(no report.csv)_", ""]
            continue
        R = load(rep)
        out.append(f"## {setname} — {label}"
                   + (f"  ·  paper mem {paper}%" if paper else "  ·  (in pool-12)"))
        out.append("")
        out.append("| variant | rule | savings | worst drop |")
        out.append("|---|---|---|---|")
        for pt in ORDER:
            if pt not in R:
                continue
            r = R[pt]
            out.append(f"| **{pt}** | {RULE[pt]} | "
                       f"{float(r['savings_pct']):.1f}% | {worst_drop(r):+.2f} |")
        # per-model cutoffs for the recommended knee
        if "Kpm" in R:
            cuts = per_model_cuts(R["Kpm"])
            cs = ", ".join(f"{m.split('-')[0][:12]}=c{v}" for m, v in cuts.items())
            out += ["", f"_Kpm per-model cutoffs: {cs}_", ""]
    return "\n".join(out)


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(here, "results"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    md = render(a.results)
    out = a.out or os.path.join(a.results, "VARIANT_SUGGESTIONS.md")
    with open(out, "w") as f:
        f.write(md + "\n")
    print(md)
    print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
