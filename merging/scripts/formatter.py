"""
Produce merged-layer spec JSON from ops_step_csvs_{family}.

Merge proposals (with per-model layer numbers) come from ops_step1_*.csv and ops_step2_*.csv.
Acceptance and cutoff use only result/*_steps.csv.

Cutoff semantics:
- By default, cutoffs are interpreted as 1-based *file line numbers* in the _steps.csv
  (header is line 1, first data row is line 2).
- Optionally, cutoffs can be interpreted as `step_idx` (when present) via --cutoff-mode step_idx.

Output format: list of "merged layers"; each merged layer is a list of
  {"model": "<name>", "layer": <int>, "component": "<attn|mlp>.<sub>"}
with each model's own layer number (layers can differ across models in one merge).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

# ops_step component (e.g. attn_q) -> output component name (attn.q_proj)
OPS_COMPONENT_TO_OUTPUT = {
    "attn_q": "attn.q_proj",
    "attn_k": "attn.k_proj",
    "attn_v": "attn.v_proj",
    "attn_o": "attn.o_proj",
    "mlp_gate": "mlp.gate_proj",
    "mlp_up": "mlp.up_proj",
    "mlp_down": "mlp.down_proj",
}

FAMILY_CONFIG = {
    "llama": {
        "models": [
            "fin-llama3.1-8b",
            "calme-2.3-legalkit-8b",
            "Llama-SafetyGuard-Content-Binary",
            "Llama-3-8B-UltraMedical",
            "Llama-3.1-8B-Instruct-multi-truth-judge",
        ],
        "cutoffs": {
            "fin-llama3.1-8b": 60,
            "calme-2.3-legalkit-8b": 52,
            "Llama-SafetyGuard-Content-Binary": 49,
            "Llama-3-8B-UltraMedical": 31,
            "Llama-3.1-8B-Instruct-multi-truth-judge": 47,
        },
        "dir": "ops_step_csvs_llama",
    },
    "deepseek": {
        "models": [
            "deepseek-math-7b-instruct",
            "deepseek-coder-7b-instruct-v1.5",
        ],
        "cutoffs": {
            "deepseek-coder-7b-instruct-v1.5": 20,
            "deepseek-math-7b-instruct": 23,
        },
        "dir": "ops_step_csvs_deepseek",
    },
    "9model": {
        "models": [
            "fin-llama3.1-8b",
            "calme-2.3-legalkit-8b",
            "Llama-SafetyGuard-Content-Binary",
            "Llama-3-8B-UltraMedical",
            "Llama-3.1-8B-Instruct-multi-truth-judge",
            "deepseek-math-7b-instruct",
            "deepseek-coder-7b-instruct-v1.5",
            "Qwen2.5-Math-7B-Instruct",
            "Qwen2.5-Coder-7B-Instruct",
        ],
        "cutoffs": {
            # Reuse existing cutoffs for shared models
            "fin-llama3.1-8b": 60,
            "calme-2.3-legalkit-8b": 52,
            "Llama-SafetyGuard-Content-Binary": 49,
            "Llama-3-8B-UltraMedical": 31,
            "Llama-3.1-8B-Instruct-multi-truth-judge": 47,
            "deepseek-coder-7b-instruct-v1.5": 20,
            "deepseek-math-7b-instruct": 23,
            # For Qwen models, default to a very high cutoff
            # so that, by default, all accepted steps are included.
            "Qwen2.5-Math-7B-Instruct": 10**9,
            "Qwen2.5-Coder-7B-Instruct": 10**9,
        },
        # Point to the 9-model CSVs used for visualization/merging.
        "dir": "../visualization/csvs_for_merging/ops_step_csvs_9model",
    },
    "qwen32b": {
        "models": [
            "T-pro-it-2.0",
            "MedGo",
            "Light-IF-32B",
        ],
        "cutoffs": {
            # From component_memory_analysis MODEL_STEP_CUTOFFS
            "T-pro-it-2.0": 118,
            "MedGo": 124,
            "Light-IF-32B": 122,
        },
        "dir": "ops_step_csvs_qwen32b",
    },
    "7_llamas_standardized": {
        "models": [
            "Llama-3.1-Hawkish-8B",
            "calme-2.3-legalkit-8b",
            "Llama-SafetyGuard-Content-Binary",
            "Llama-3.1-8B-Instruct-multi-truth-judge",
            "Llama-3.1-8B-UltraMedical",
        ],
        # 1-based CSV *line* numbers -- this script's default convention. The first data row
        # of a steps.csv is the baseline (step_idx == -1), so step_idx == line - 3.
        # E.g. SafetyGuard line 70 -> step_idx 67, the row scoring 90.25.
        # plot_steps.apply_step_cutoff reads these same numbers as step_idx, i.e. 3 steps
        # too late; treat its cutoffs as line numbers.
        "cutoffs": {
            "Llama-3.1-Hawkish-8B": 38,                       # step_idx 35, score 56.14
            "calme-2.3-legalkit-8b": 62,                      # step_idx 59, score 50.07
            "Llama-SafetyGuard-Content-Binary": 69,           # step_idx 66, score 90.60
            "Llama-3.1-8B-Instruct-multi-truth-judge": 59,    # step_idx 56, score 74.75
            "Llama-3.1-8B-UltraMedical": 54,                  # step_idx 51, score 78.31
        },
        "cutoff_mode": "line",
        "dir": "../clustering/candidates/7_llamas_standardized",
        # This run stores results as <short_name>/steps.csv rather than <model>_steps.csv.
        "steps_dir": "../micr/results/7_llamas_standardized",
        "steps_files": {
            "Llama-3.1-Hawkish-8B": "hawkish/steps.csv",
            "calme-2.3-legalkit-8b": "legalkit/steps.csv",
            "Llama-SafetyGuard-Content-Binary": "safetyguard/steps.csv",
            "Llama-3.1-8B-Instruct-multi-truth-judge": "truthjudge/steps.csv",
            "Llama-3.1-8B-UltraMedical": "ultramedical/steps.csv",
        },
    },
}


def _steps_path(result_dir: Path, model: str, steps_files: dict[str, str] | None = None) -> Path:
    if steps_files and model in steps_files:
        return result_dir / steps_files[model]
    return result_dir / f"{model}_steps.csv"


def load_merge_proposals(
    ops_dir: Path, models: list[str]
) -> list[tuple[str, list[tuple[str, int]]]]:
    """
    Load all merge proposals from ops_step1_*.csv and ops_step2_*.csv in ops_dir.
    Returns list of (component_output, [(model, layer), ...]) with only specified MODELS, deduplicated.
    """
    seen: set[tuple[str, frozenset[tuple[str, int]]]] = set()
    result: list[tuple[str, list[tuple[str, int]]]] = []
    # Sort to ensure deterministic order of processing
    paths = sorted(ops_dir.glob("ops_step*.csv"))
    if not paths:
        print(f"Warning: No ops_step*.csv files found in {ops_dir}")
        
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    comp_raw = row.get("component", "").strip()
                    models_str = row.get("models", "").strip().strip('"')
                except (KeyError, ValueError):
                    continue
                comp_out = OPS_COMPONENT_TO_OUTPUT.get(comp_raw)
                if not comp_out or not models_str:
                    continue
                # Parse "M1:L1,M2:L2,..."
                entries = []
                for part in re.split(r",\s*", models_str):
                    if ":" not in part:
                        continue
                    model_name, layer_str = part.split(":", 1)
                    model_name = model_name.strip()
                    if model_name not in models:
                        continue
                    try:
                        layer = int(layer_str.strip())
                    except ValueError:
                        continue
                    entries.append((model_name, layer))
                if not entries:
                    continue
                # Dedupe (model, layer) within this proposal
                entries = list(dict.fromkeys(entries))  # preserve order, unique (model, layer)
                key = (comp_out, frozenset(entries))
                if key in seen:
                    continue
                seen.add(key)
                result.append((comp_out, entries))
    return result


def load_model_steps(
    result_dir: Path, model: str, steps_files: dict[str, str] | None = None
) -> tuple[dict[tuple[str, int], str], list[tuple[tuple[str, int], str, int, int | None]]]:
    """
    Load one model's result/<model>_steps.csv (not ops_step1/2). component in CSV is "attn" or "mlp".
    Returns decisions (component, layer) -> accepted/rejected and list of
    (key, decision, file_line_no, step_idx_or_none).
    """
    path = _steps_path(result_dir, model, steps_files)
    if not path.exists():
        return {}, []
    decisions: dict[tuple[str, int], str] = {}
    rows: list[tuple[tuple[str, int], str, int, int | None]] = []
    with open(path, newline="", encoding="utf-8") as f:
        lines = f.readlines()
    reader = csv.DictReader(lines)
    for line_no, row in enumerate(reader, start=2):
        try:
            comp = row["component"].strip()
            layer = int(row["layer"])
            decision = row["decision"].strip().lower()
        except (KeyError, ValueError):
            continue
        step_idx: int | None = None
        step_idx_raw = row.get("step_idx")
        if step_idx_raw is not None and str(step_idx_raw).strip() != "":
            try:
                step_idx = int(str(step_idx_raw).strip())
            except ValueError:
                step_idx = None
        key = (comp, layer)
        rows.append((key, decision, line_no, step_idx))
        if decision == "accepted":
            decisions[key] = "accepted"
        elif key not in decisions:
            decisions[key] = decision
    return decisions, rows


def load_all_steps(
    result_dir: Path, models: list[str], steps_files: dict[str, str] | None = None
) -> tuple[
    dict[str, dict[tuple[str, int], str]],
    dict[str, list[tuple[tuple[str, int], str, int, int | None]]],
]:
    all_decisions: dict[str, dict[tuple[str, int], str]] = {}
    all_rows: dict[str, list[tuple[tuple[str, int], str, int, int | None]]] = {}
    for model in models:
        dec, row_list = load_model_steps(result_dir, model, steps_files)
        all_decisions[model] = dec
        all_rows[model] = row_list
    return all_decisions, all_rows


def comp_output_to_block(comp_out: str) -> str:
    """attn.q_proj -> attn, mlp.gate_proj -> mlp."""
    return comp_out.split(".")[0]


def build_spec(
    proposals: list[tuple[str, list[tuple[str, int]]]],
    all_decisions: dict[str, dict[tuple[str, int], str]],
    all_rows: dict[str, list[tuple[tuple[str, int], str, int, int | None]]],
    models: list[str],
    cutoffs: dict[str, int],
    use_cutoff: bool,
    cutoff_mode: str,
    keep_singletons: bool = False,
) -> list[list[dict]]:
    """
    For each merge proposal (component, [(model, layer), ...]), keep only (model, layer)
    where that model accepted (attn|mlp, layer) in result/<model>_steps.csv and (if use_cutoff)
    at/before the model cutoff according to cutoff_mode:
      - "line": 1-based file line numbers in _steps.csv (header is line 1)
      - "step_idx": numeric `step_idx` from _steps.csv (if missing, the row never counts toward cutoff)
    """
    effective_cutoffs = cutoffs if use_cutoff else {m: 1 << 30 for m in models}
    accepted_within_cutoff: dict[str, set[tuple[str, int]]] = {}
    if use_cutoff:
        for model in models:
            accepted_within_cutoff[model] = set()
            cutoff = effective_cutoffs.get(model, 0)
            for (key, decision, line_no, step_idx) in all_rows.get(model, []):
                if decision != "accepted":
                    continue
                if cutoff_mode == "step_idx":
                    if step_idx is not None and step_idx <= cutoff:
                        accepted_within_cutoff[model].add(key)
                else:
                    if line_no <= cutoff:
                        accepted_within_cutoff[model].add(key)
    result: list[list[dict]] = []
    seen: set[tuple[str, frozenset[tuple[str, int]]]] = set()
    for comp_out, entries in proposals:
        block = comp_output_to_block(comp_out)
        kept = []
        for model, layer in entries:
            key = (block, layer)
            if all_decisions.get(model, {}).get(key) != "accepted":
                continue
            if use_cutoff and key not in accepted_within_cutoff.get(model, set()):
                continue
            kept.append({"model": model, "layer": layer, "component": comp_out})
        if not kept:
            continue
        # Deduplicate: same (component, set of (model, layer)) can come from different ops_step rows
        canonical = (comp_out, frozenset((d["model"], d["layer"]) for d in kept))
        if canonical in seen:
            continue
        seen.add(canonical)
        result.append(kept)

    # Tensor-level dedup, consistent with the savings model's last-write-wins
    # (build_operating_points.recipes_at overwrites an earlier merge of a slot
    # with a later one -> the LAST accepted write within the cutoff wins). A
    # single (model, layer, component) tensor merged in BOTH a stage-1
    # same-layer group and a stage-2 cross-layer group (proposals are ordered
    # stage 1 then stage 2) would otherwise be emitted in multiple blocks --
    # ill-defined for a downstream merge and inconsistent with the accounted
    # grouping. Keep each tensor only in its LAST (latest-proposal) block; drop
    # any block that collapses to a single model (a group of one is not a merge,
    # so that tensor stays unmerged -- which matches how recipes_at prices it).
    final_of: dict[tuple[str, int, str], int] = {}
    for bi, blk in enumerate(result):
        for d in blk:
            final_of[(d["model"], d["layer"], d["component"])] = bi
    deduped: list[list[dict]] = []
    for bi, blk in enumerate(result):
        kept_blk = [d for d in blk
                    if final_of[(d["model"], d["layer"], d["component"])] == bi]
        # Default: drop blocks that collapse to a single model (a group of one is
        # not a merge). keep_singletons=True keeps them -- used for the full
        # per-tensor recipe emitted to the .json/.jsonl specs (touched-but-
        # unmerged tensors), while the savings model stays merges-only.
        if len(kept_blk) >= (1 if keep_singletons else 2):
            deduped.append(kept_blk)
    return deduped


def drop_subset_merged_layers(spec: list[list[dict]]) -> list[list[dict]]:
    """
    Drop merged layers that are fully covered by another merged layer of the same
    component.

    "Covered" means the smaller layer's exact set of (model, layer) pairs is a
    proper subset of the larger one's (i.e., the larger has the same per-model
    layer assignments for all models in the smaller, plus extra models).
    """
    # Group by component: comp -> [(index, set of (model, layer))]
    by_comp: dict[str, list[tuple[int, frozenset[tuple[str, int]]]]] = {}
    for i, block in enumerate(spec):
        comp = block[0]["component"]
        pairs = frozenset((d["model"], d["layer"]) for d in block)
        by_comp.setdefault(comp, []).append((i, pairs))
    # Drop any block whose (model, layer) set is a proper subset of another's (same component)
    drop: set[int] = set()
    for blocks in by_comp.values():
        for i, pairs_i in blocks:
            for j, pairs_j in blocks:
                if i != j and pairs_i < pairs_j:
                    drop.add(i)
                    break
    return [block for i, block in enumerate(spec) if i not in drop]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate merged model specs from ops_step CSVs.")
    parser.add_argument(
        "--family",
        choices=sorted(FAMILY_CONFIG),
        default="llama",
        help="Model family to process (default: llama)",
    )
    parser.add_argument(
        "--cutoff-mode",
        choices=["line", "step_idx"],
        default=None,
        help="Interpret cutoffs as CSV line numbers or as step_idx values. Defaults to the "
             "family's own cutoff_mode, else 'line'. NOTE: plot_steps.py cutoffs are step_idx; "
             "line == step_idx + 3 when the first data row is the baseline.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "jsonl", "both"],
        default="json",
        help="jsonl writes one merged layer per line (each line is one element of the JSON list).",
    )
    parser.add_argument("--out-dir", default=None, help="Where to write the spec files.")
    args = parser.parse_args()

    config = FAMILY_CONFIG[args.family]
    models = config["models"]
    cutoffs = config["cutoffs"]
    ops_dir_name = config["dir"]
    cutoff_mode = args.cutoff_mode or config.get("cutoff_mode", "line")
    steps_files = config.get("steps_files")

    base_dir = Path(__file__).resolve().parent
    ops_dir = (base_dir / ops_dir_name).resolve()
    if config.get("steps_dir"):
        result_dir = (base_dir / config["steps_dir"]).resolve()
    else:
        result_dir = ops_dir / "result"

    # Check if directories exist
    if not ops_dir.exists():
        print(f"Error: Directory {ops_dir} does not exist.")
        return
    if not result_dir.exists():
        print(f"Error: Directory {result_dir} does not exist.")
        return

    print(f"Processing family: {args.family}")
    print(f"Models: {models}")
    print(f"Ops directory: {ops_dir}")
    print(f"Steps directory: {result_dir}")
    print(f"Cutoff mode: {cutoff_mode}  cutoffs: {cutoffs}")

    for m in models:
        sp = _steps_path(result_dir, m, steps_files)
        if not sp.exists():
            print(f"Error: steps CSV missing for {m}: {sp}")
            return

    proposals = load_merge_proposals(ops_dir, models)
    all_decisions, all_rows = load_all_steps(result_dir, models, steps_files)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else ops_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    def emit(spec: list[list[dict]], stem: str) -> None:
        if args.format in ("json", "both"):
            p = out_dir / f"{stem}.json"
            with open(p, "w", encoding="utf-8") as f:
                json.dump(spec, f, indent=2)
            print(f"Wrote {len(spec)} merged layers -> {p}")
        if args.format in ("jsonl", "both"):
            p = out_dir / f"{stem}.jsonl"
            with open(p, "w", encoding="utf-8") as f:
                for group in spec:
                    f.write(json.dumps(group) + "\n")
            print(f"Wrote {len(spec)} merged layers -> {p}")

    spec_all = drop_subset_merged_layers(
        build_spec(proposals, all_decisions, all_rows, models, cutoffs,
                   use_cutoff=False, cutoff_mode=cutoff_mode)
    )
    emit(spec_all, "merged_spec_all_accepted")

    spec_cutoff = drop_subset_merged_layers(
        build_spec(proposals, all_decisions, all_rows, models, cutoffs,
                   use_cutoff=True, cutoff_mode=cutoff_mode)
    )
    emit(spec_cutoff, "merged_spec_up_to_cutoff")


if __name__ == "__main__":
    main()
