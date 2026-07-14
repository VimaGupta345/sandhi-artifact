#!/usr/bin/env python3
"""ETL: our pipeline results -> the Figure-5 plot's data CSVs.

Reads each set's `report.csv` (M-split = the eval subset) and writes the CSVs
that `plots/source/accuracy_memory_figures.py` consumes, so the plot renders OUR
numbers instead of the paper's shipped tables. All accuracy is on the **M-split**
(the same eval subset the SANDHI operating points were selected on), so SANDHI,
No-merge, Full-merge and LoRA are mutually comparable.

Produces so far (the bars we have data for):
  - all_models_final.csv : SANDHI per-model accuracy delta + memory saved
  - vllm-no-merge.csv     : No-merge (unmerged specialist) accuracy, our benchmarks

Full-merge (full_merge/*.csv) and LoRA (lora/*.csv) are written by the baseline
eval step (BASELINES.md) once those run on the M-split; this script leaves the
existing files untouched if their inputs are absent.

Point selected for the SANDHI bar: default `Cpm` (per-model <=2%); override with
--point.
"""
import argparse
import csv
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# our model label -> plot 'Model' key (DOMAIN_MAPS in the plot script)
MODEL_KEY = {
    "Llama-3.1-8B-UltraMedical": "Medical",
    "Llama-3.1-Hawkish-8B": "Finance",
    "calme-2.3-legalkit-8b": "Legal",
    "Llama-SafetyGuard-Content-Binary": "Safety",
    "Llama-3.1-8B-Instruct-multi-truth-judge": "Truth",
    "Qwen2.5-Math-7B-Instruct": "Math",
    "Qwen2.5-Coder-7B-Instruct": "Coder",
    "Light-IF-32B": "Light-IF 32B",
    "MedGo": "MedGo",
    "T-pro-it-2.0": "T-pro-it 2.0",
    "deepseek-math-7b-instruct": "DS-Math",
    "deepseek-coder-7b-instruct-v1.5": "DS-Coder",
}
# our benchmark per model (for vllm-no-merge.csv 'benchmark' column)
BENCH = {
    "Medical": "medqa_4options", "Finance": "mmlu_econometrics", "Legal": "mmlu_professional_law",
    "Safety": "sst2", "Truth": "truthfulqa_mc2", "Math": "gsm8k-cot", "Coder": "humaneval",
    "Light-IF 32B": "tinyMMLU", "MedGo": "medqa_4options", "T-pro-it 2.0": "m_mmlu_ru",
    "DS-Math": "gsm8k-cot", "DS-Coder": "humaneval",
}

# plot Set name -> (report.csv path relative to repo, is_composed)
# single-run sets read one report.csv; composed sets (fig5d/fig6e) reuse the
# atoms' reports (compose_figures.py logic) -- both handled by _set_rows().
# The four Figure-5 panels (a-d) only. Panel (d) 12-model is the additive
# composition of the atom reports (cross-family already inside the 7-atom).
SETS = {
    "3-Qwen-32B":     ["runs/qwen32b3/analysis/fig5a/report.csv"],              # (a)
    "5-llama":        ["runs/llama5/analysis/llama5/report.csv"],              # (b)
    "7-5-llama-2-DS": ["runs/llama5_deepseek2/analysis/llama5_deepseek2/report.csv"],  # (c)
    "12-model":       ["runs/llama5_deepseek2/analysis/llama5_deepseek2/report.csv",   # (d)
                       "runs/qwen25_2/analysis/qwen25_2/report.csv",
                       "runs/qwen32b3/analysis/fig5a/report.csv"],
}
MIB_TO_GB = 1.0 / 1024


def _load(path):
    with open(os.path.join(HERE, path)) as f:
        return {r["point"]: r for r in csv.DictReader(f)}


def _models(report):  # model labels present in a report
    return [c[:-len("__drop")] for c in report["A"] if c.endswith("__drop")]


def _set_rows(paths, point):
    """(per-model {key: delta}, total memory saved GB) for a set, summing freed
    over its atom reports (additive; only the joint 7-atom carries cross-family)."""
    deltas, saved_mb = {}, 0.0
    for p in paths:
        R = _load(p)
        pt = R.get(point) or R["C"]
        for m in _models(R):
            if m in MODEL_KEY:
                deltas[MODEL_KEY[m]] = -float(pt[f"{m}__drop"])   # delta = -drop
        saved_mb += float(pt["savings_mb"])
    return deltas, saved_mb * MIB_TO_GB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", default="Cpm", help="operating point for the SANDHI bar")
    ap.add_argument("--data", default=os.path.join(HERE, "plots", "data"))
    a = ap.parse_args()

    # --- all_models_final.csv (SANDHI) ---
    amf = os.path.join(a.data, "all_models_final.csv")
    with open(amf, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Set", "Model", "Accuracy_Delta_Percent", "Memory_Saved_GB"])
        for setname, paths in SETS.items():
            deltas, saved_gb = _set_rows(paths, a.point)
            for key, d in deltas.items():
                w.writerow([setname, key, round(d, 2), round(saved_gb, 2)])
    print(f"[wrote] {amf}  (SANDHI, point={a.point}, M-split)")

    # --- vllm-no-merge.csv (No-merge = report.csv point A, our benchmarks) ---
    nm = {}
    for paths in SETS.values():
        for p in paths:
            R = _load(p)
            for m in _models(R):
                if m in MODEL_KEY:
                    key = MODEL_KEY[m]
                    nm[key] = float(R["A"][f"{m}__score"])
    vnm = os.path.join(a.data, "vllm-no-merge.csv")
    fam = {"Medical": "llama3.1", "Finance": "llama3.1", "Legal": "llama3.1", "Safety": "llama3.1",
           "Truth": "llama3.1", "Math": "qwen2.5", "Coder": "qwen2.5",
           "Light-IF 32B": "qwen3", "MedGo": "qwen3", "T-pro-it 2.0": "qwen3",
           "DS-Math": "deepseek", "DS-Coder": "deepseek"}
    dom = {"Medical": "medical", "Finance": "financial", "Legal": "legal", "Safety": "toxicity",
           "Truth": "truthfulness", "Math": "math", "Coder": "coding", "Light-IF 32B": "IF",
           "MedGo": "medical", "T-pro-it 2.0": "Russian", "DS-Math": "math", "DS-Coder": "coding"}
    with open(vnm, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain", "benchmark", "base_model", "base_score", "source"])
        for key, score in nm.items():
            w.writerow([dom[key], BENCH[key], key, round(score, 2), fam[key]])
    print(f"[wrote] {vnm}  (No-merge, M-split, our benchmarks)")
    print("NOTE: Full-merge (full_merge/*.csv) + LoRA (lora/*.csv) come from the "
          "M-split baseline evals (BASELINES.md); not overwritten here.")


if __name__ == "__main__":
    main()
