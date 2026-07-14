#!/usr/bin/env python3
"""Detect which base runs (atoms) exist and compose every figure pool from them.

Only FIVE base runs are ever needed:

  5llama, 2ds, 2qwen, 3qwen32b   -- within-family (each merged internally)
  5llama+ds                      -- the ONE joint cross-family run

Every figure pool is an ADDITIVE composition of these atoms:

    savings = (Σ atom freed) / (Σ atom size)

because the only cross-family merge that is shape-possible is llama<->deepseek
(both hidden=4096); Qwen2.5 (3584) and Qwen3-32B (5120) share with nothing, so
their freed bytes simply add over the summed denominator. The joint `5llama+ds`
run carries the only non-additive (cross-family) sharing.

This script auto-detects each atom by matching a report.csv's member columns,
then reports which figure pools are computable and their composed operating
points (Bpm=<=1%, Cpm=<=2%, Kpm=knee). Pure read of report.csv -- no eval, no
GPU -- so reviewers reproduce it from the shipped CSVs or after a rerun.
"""
import csv
import glob
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

L5 = {"Llama-3.1-8B-UltraMedical", "Llama-3.1-8B-Instruct-multi-truth-judge",
      "Llama-SafetyGuard-Content-Binary", "calme-2.3-legalkit-8b",
      "Llama-3.1-Hawkish-8B"}
DS2 = {"deepseek-math-7b-instruct", "deepseek-coder-7b-instruct-v1.5"}
QW2 = {"Qwen2.5-Coder-7B-Instruct", "Qwen2.5-Math-7B-Instruct"}
QW32 = {"Light-IF-32B", "T-pro-it-2.0", "MedGo"}

# atom -> exact member set that identifies its report.csv
ATOMS = {"5llama": L5, "2ds": DS2, "2qwen": QW2, "3qwen32b": QW32,
         "5llama+ds": L5 | DS2}

# figure pool -> (atom list, paper memory %). Cross-family pools use the JOINT
# 5llama+ds atom (never 5llama + 2ds separately -- that would drop the
# llama<->deepseek sharing and double nothing, since the joint atom IS the 7).
FIGURES = {
    "fig5a (3, qwen32b)":        (["3qwen32b"], 49.8),
    "fig5b (5, llama)":          (["5llama"], 35.2),
    "fig5c (7, llama+ds)":       (["5llama+ds"], 26.7),
    "fig5d (12, all)":           (["5llama+ds", "2qwen", "3qwen32b"], 38.0),
    "fig6a (2, ds)":             (["2ds"], None),
    "fig6b (2, qwen)":           (["2qwen"], None),
    "fig6c (7, llama+ds)":       (["5llama+ds"], None),
    "fig6d (5, llama)":          (["5llama"], None),
    "fig6e (9, llama+ds+qwen)":  (["5llama+ds", "2qwen"], None),
}
POINTS = ["Bpm", "Cpm", "Kpm"]


def detect_atoms(runs_glob):
    """Scan report.csv files; map each atom to a report whose member columns
    match it exactly, PREFERRING one that carries the per-model points (Bpm/Cpm/
    Kpm) -- so a stale pre-per-model run never shadows the current one."""
    found = {}  # atom -> (report_path, has_per_model)
    for rep in sorted(glob.glob(runs_glob)):
        try:
            rows = list(csv.DictReader(open(rep)))
        except OSError:
            continue
        if not rows:
            continue
        members = frozenset(c[:-len("__baseline")] for c in rows[0]
                            if c.endswith("__baseline"))
        has_pm = any(r["point"] == "Cpm" for r in rows)
        for atom, mset in ATOMS.items():
            if members != frozenset(mset):
                continue
            if atom not in found or (has_pm and not found[atom][1]):
                found[atom] = (rep, has_pm)
    return {atom: path for atom, (path, _) in found.items()}


def atom_report(path):
    R = {r["point"]: r for r in csv.DictReader(open(path))}
    size = next((float(r["savings_mb"]) / (float(r["savings_pct"]) / 100)
                 for r in R.values() if float(r["savings_pct"]) > 0), 0.0)
    return R, size


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "FIGURE_COMPOSITIONS.md"))
    a = ap.parse_args()

    found = detect_atoms(os.path.join(a.runs, "*", "analysis", "*", "report.csv"))
    reports = {atom: atom_report(p) for atom, p in found.items()}

    out = ["# Figure pools composed from available base runs", "",
           "Additive composition of the 5 atoms (only `5llama+ds` is joint / "
           "non-additive). Savings = (Σ atom freed)/(Σ atom size).", "",
           "## Detected base runs (atoms)", ""]
    for atom in ATOMS:
        if atom in found:
            _, sz = reports[atom]
            out.append(f"- ✅ **{atom}** — {sz/1024:.1f} GB — `{os.path.relpath(found[atom], HERE)}`")
        else:
            out.append(f"- ❌ **{atom}** — NOT FOUND (run it to unlock its figures)")
    out += ["", "## Figure pools", "",
            "| figure | atoms | status | pool GB | Bpm ≤1% | Cpm ≤2% | Kpm knee | paper |",
            "|---|---|---|---|---|---|---|---|"]
    for fig, (atoms, paper) in FIGURES.items():
        missing = [x for x in atoms if x not in found]
        if missing:
            out.append(f"| {fig} | {'+'.join(atoms)} | ⛔ missing {','.join(missing)} "
                       f"| — | — | — | — | {paper or '—'} |")
            continue
        size = sum(reports[x][1] for x in atoms)
        cells = []
        for pt in POINTS:
            freed = sum(float(reports[x][0][pt]["savings_mb"])
                        for x in atoms if pt in reports[x][0])
            cells.append(f"{100*freed/size:.1f}%")
        pcell = f"{paper}%" if paper else "—"
        out.append(f"| **{fig}** | {'+'.join(atoms)} | ✅ | {size/1024:.0f} "
                   f"| {cells[0]} | {cells[1]} | {cells[2]} | {pcell} |")
    md = "\n".join(out)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write(md + "\n")
    print(md)
    print(f"\n[wrote] {a.out}")


if __name__ == "__main__":
    main()
