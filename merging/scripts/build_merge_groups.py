"""
Emit merge-group JSON/JSONL that matches the actually-applied MICR merges for a run.

Each group is one merge operation that was ACCEPTED and (optionally) within a per-model
cutoff: a list of {"model", "layer", "component"} giving the tensors averaged together,
each at its own source layer (stage-2 merges are cross-layer). Groups are deduplicated by
(component, frozenset of (model, layer)), like scripts/formatter.py.

Why not use formatter.py: its acceptance test keys on (block=mlp|attn, layer) collapsed
across stages, and its proposals drop stage identity during dedup. For a run with stage-2
cross-layer merges that mis-attributes membership -- verified against the reproduced 7-llama
variants, formatter's cutoff spec had 52 entries no reproduction uses and 9 it was missing.
This script instead walks the same accepted blocks the replay applies, so its flattened
member set is exactly the set of contributor tensors the variants were built from.

Cutoffs are 1-based CSV line numbers (step_idx = line - 3); see cutoff-index-conventions.
"""
from __future__ import annotations
import argparse, csv, json, os
from pathlib import Path

ATTN = ["attn_q", "attn_k", "attn_v", "attn_o"]
MLP = ["mlp_gate", "mlp_up", "mlp_down"]
GROUP = {**{c: "attn" for c in ATTN}, **{c: "mlp" for c in MLP}}
OUT = {"attn_q": "attn.q_proj", "attn_k": "attn.k_proj", "attn_v": "attn.v_proj",
       "attn_o": "attn.o_proj", "mlp_gate": "mlp.gate_proj", "mlp_up": "mlp.up_proj",
       "mlp_down": "mlp.down_proj"}


def load_ops(ops_dir: Path, target: str):
    rows = []
    for stage, name in ((1, f"ops_step1_{target}.csv"), (2, f"ops_step2_{target}.csv")):
        p = ops_dir / name
        if p.exists():
            for r in csv.DictReader(open(p)):
                if r.get("op") == "merge":
                    rows.append({"stage": stage, "component": r["component"],
                                 "layer": int(r["layer"]), "models": r["models"]})
    return rows


def plan_blocks(rows):
    blocks, i, n = [], 0, len(rows)
    while i < n:
        st, ly = rows[i]["stage"], rows[i]["layer"]
        j = i
        while j < n and rows[j]["stage"] == st and rows[j]["layer"] == ly:
            j += 1
        k = i
        while k < j:
            g = GROUP[rows[k]["component"]]
            idx = []
            while k < j and GROUP[rows[k]["component"]] == g:
                idx.append(k); k += 1
            blocks.append({"stage": st, "layer": ly, "group": g, "idx": idx})
        i = j
    return blocks


def parse_models(s, default_layer):
    out = []
    for tok in str(s).split(","):
        p = tok.strip().split(":")
        if p and p[0].strip():
            out.append((p[0].strip(),
                        int(p[1]) if len(p) > 1 and p[1].strip().isdigit() else default_layer))
    return out


def groups_for_run(ops_dir, steps_csv, target, cutoff_line):
    """Applied (accepted, within-cutoff) merge groups for one target run."""
    rows = load_ops(ops_dir, target)
    blocks = plan_blocks(rows)
    steps = [r for r in csv.DictReader(open(steps_csv)) if int(r["step_idx"]) >= 0]
    if len(blocks) != len(steps):
        raise SystemExit(f"{target}: {len(blocks)} planned blocks != {len(steps)} recorded steps")
    cutoff = None if cutoff_line is None else cutoff_line - 3
    out = []
    for b, s in zip(blocks, steps):
        if s["decision"] != "accepted":
            continue
        if cutoff is not None and int(s["step_idx"]) > cutoff:
            continue
        for comp in (ATTN if b["group"] == "attn" else MLP):
            for i in b["idx"]:
                r = rows[i]
                if r["component"] != comp:
                    continue
                parts = parse_models(r["models"], b["layer"])
                if not any(m == target for m, _ in parts):
                    continue
                out.append((OUT[comp], parts))
    return out


def build(models_cfg, ops_dir, steps_dir, steps_files, use_cutoff):
    seen, spec = set(), []
    for m in models_cfg:
        steps_csv = steps_dir / (steps_files.get(m["label"]) if steps_files else f"{m['label']}_steps.csv")
        for comp_out, parts in groups_for_run(ops_dir, steps_csv, m["label"],
                                              m["cutoff"] if use_cutoff else None):
            canon = (comp_out, frozenset(parts))
            if canon in seen:
                continue
            seen.add(canon)
            spec.append([{"model": lbl, "layer": lyr, "component": comp_out} for lbl, lyr in parts])
    return spec


# The 7_llamas_standardized run. Cutoffs are CSV line numbers.
DEFAULT = {
    "models": [
        {"label": "Llama-3.1-Hawkish-8B", "cutoff": 38},
        {"label": "calme-2.3-legalkit-8b", "cutoff": 62},
        {"label": "Llama-SafetyGuard-Content-Binary", "cutoff": 69},
        {"label": "Llama-3.1-8B-Instruct-multi-truth-judge", "cutoff": 59},
        {"label": "Llama-3.1-8B-UltraMedical", "cutoff": 54},
    ],
    "ops_dir": "clustering/candidates/7_llamas_standardized",
    "steps_dir": "micr/results/7_llamas_standardized",
    "steps_files": {
        "Llama-3.1-Hawkish-8B": "hawkish/steps.csv",
        "calme-2.3-legalkit-8b": "legalkit/steps.csv",
        "Llama-SafetyGuard-Content-Binary": "safetyguard/steps.csv",
        "Llama-3.1-8B-Instruct-multi-truth-judge": "truthjudge/steps.csv",
        "Llama-3.1-8B-UltraMedical": "ultramedical/steps.csv",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--format", choices=["json", "jsonl", "both"], default="both")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    ops_dir = (root / DEFAULT["ops_dir"]).resolve()
    steps_dir = (root / DEFAULT["steps_dir"]).resolve()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for use_cutoff, stem in ((False, "merged_spec_all_accepted"), (True, "merged_spec_up_to_cutoff")):
        spec = build(DEFAULT["models"], ops_dir, steps_dir, DEFAULT["steps_files"], use_cutoff)
        if args.format in ("json", "both"):
            json.dump(spec, open(out_dir / f"{stem}.json", "w"), indent=2)
        if args.format in ("jsonl", "both"):
            with open(out_dir / f"{stem}.jsonl", "w") as f:
                for g in spec:
                    f.write(json.dumps(g) + "\n")
        members = sum(len(g) for g in spec)
        print(f"  {stem}: {len(spec)} groups, {members} members")


if __name__ == "__main__":
    main()
