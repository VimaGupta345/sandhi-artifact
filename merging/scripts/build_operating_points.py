"""
Derive memory/accuracy operating points for a merged model *pool* and emit one
JSONL merge-spec per point (one merged group per line).

Concept
-------
For a pool of N models we walk the per-model MICR step logs (accept/reject +
eval score) in their recorded order. A larger "cutoff" (step_idx threshold)
accepts more merges -> more memory savings, but lower accuracy. Each operating
point is the *deepest* cutoff (max savings) whose per-model accuracy drop stays
within an accuracy budget, measured against the paper-reported baselines.

  A = unmerged reference (0 savings, 0 drop) -- no file emitted.
  B = max savings with every model within `budget_b` points of its baseline.
  C = max savings with every model within `budget_c` points of its baseline.

Merge plan + acceptance reuse `formatter.py` (load_merge_proposals / build_spec /
drop_subset_merged_layers) -- the emitted merge specs (jsonl) are unchanged.

Memory savings model (corrected, distinct-tensor over full model size):
  NUMERATOR   Per (layer, component) slot, group the pool's models by their
              FINAL merge recipe at that slot: the frozenset of
              (contributor_label, contributor_layer) written by the last
              accepted step within the cutoff (stage-aware: a stage-2 accept
              overwrites a stage-1 one). Models from different families can
              never share a recipe (contributor labels differ), so
              cross-family groups never form. Freed MiB = sum over recipe
              groups with k >= 2 models of (k - 1) * component_size_MiB.
  DENOMINATOR The FULL on-disk model size per family (attn+mlp projections +
              embed_tokens + lm_head + norms, measured from safetensors),
              summed over the pool's members. See FAMILY_PROFILES.
This replaces the earlier participant-count / attn+mlp-only model: counting
(len(group)-1) per proposal group over-counts freed copies when several
models converge onto the SAME final tensor via different proposal rows, and
the attn+mlp-only denominator overstates savings as a fraction of what is
actually stored on disk.

Usage:
  python scripts/build_operating_points.py
  python scripts/build_operating_points.py --pool 7_llamas --sweep
  python scripts/build_operating_points.py --pool 7_llamas --validate
  python scripts/build_operating_points.py --pool 7_llamas+deepseek --sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

# --- reuse the existing merge-spec builder ---------------------------------
import sys

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent  # merge_tools/
sys.path.insert(0, str(_SCRIPT_DIR))
# Register the in-repo `merge_tools` package keyed to this file's location
# (dirname-independent; a same-named sibling checkout can never shadow it).
import types as _types
if "merge_tools" not in sys.modules:
    _mt_pkg = _types.ModuleType("merge_tools")
    _mt_pkg.__path__ = [str(_REPO_ROOT)]
    sys.modules["merge_tools"] = _mt_pkg
import formatter as F  # noqa: E402

# --- shared baselines (Agent B's module) -----------------------------------
# Signature: get_baseline(label, task=None) -> Optional[float]. We prefer this
# source and fall back to the per-pool literal baselines below if it returns
# None (or the module is unavailable), keeping the existing numbers intact.
try:  # pragma: no cover - depends on sibling module being present at runtime
    from merge_tools.baselines import get_baseline as _get_baseline  # noqa: E402
except Exception:  # pragma: no cover - additive fallback keeps old behavior
    def _get_baseline(label, task=None):
        return None


def resolve_baselines(cfg: dict) -> dict:
    """Baseline accuracy per model label.

    Prefer merge_tools.baselines.get_baseline(label); fall back to the literal
    per-pool baseline when it returns None. The literal fallback reproduces the
    exact previously-hardcoded values, so drop arithmetic is unchanged.
    """
    literal = cfg.get("baselines") or {}
    resolved = {}
    for label in cfg["models"]:
        val = None
        try:
            val = _get_baseline(label)
        except Exception:
            val = None
        if val is None:
            val = literal.get(label)
        # None is legal here: build_pool prefers the run's own steps.csv
        # baseline row (run_base, measured on the same eval split as every
        # score in the sweep) over whatever resolves here; central/literal
        # values are only the fallback for CSVs without a baseline row.
        resolved[label] = val
    return resolved


def _short(label: str) -> str:
    """Compact display name for report tables (no-dash labels, e.g. MedGo,
    would IndexError under the old split('-')[1])."""
    parts = label.split("-")
    return parts[1] if len(parts) > 1 else label


def _anchor_baselines(models, baselines: dict, run_base: dict) -> dict:
    """Anchor drop arithmetic to the run's own steps.csv baseline row.

    run_base is measured on the same eval split (M) as every score in the
    sweep -- the only self-consistent anchor for split-mode runs. Central/
    literal baselines are the fallback for recorded runs whose CSVs lack a
    baseline row. Mutates and returns `baselines` (callers pass a copy when
    they need to keep the un-anchored profiling view).
    """
    for label in models:
        if run_base.get(label) is not None:
            baselines[label] = run_base[label]
        elif baselines.get(label) is None:
            raise SystemExit(f"no baseline for {label}: its steps.csv has no "
                             f"baseline row and the pool has no literal/central "
                             f"baseline")
    return baselines

# --- corrected memory model: per-family sizes -------------------------------
# `components`: per-layer tensor sizes in MiB (bf16), matching
# visualization/plot_steps.py:MEMORY_PROFILES for llama-8b / deepseek-7b.
# `full_model_mib`: FULL model size on disk -- measured from the safetensors
# shards (attn+mlp projections + embed_tokens + lm_head + norms). This is the
# per-member denominator of the savings percentage.
FAMILY_PROFILES = {
    "llama-8b": {
        "components": {
            "q_proj": 32.0, "k_proj": 8.0, "v_proj": 8.0, "o_proj": 32.0,
            "gate_proj": 112.0, "up_proj": 112.0, "down_proj": 112.0,
        },
        "full_model_mib": 15316.5,  # measured from safetensors
        "num_layers": 32,
    },
    "deepseek-7b": {
        "components": {
            "q_proj": 32.0, "k_proj": 32.0, "v_proj": 32.0, "o_proj": 32.0,
            "gate_proj": 86.0, "up_proj": 86.0, "down_proj": 86.0,
        },
        "full_model_mib": 13180.5,  # measured from safetensors
        "num_layers": 30,
    },
    "qwen2.5-7b": {
        "components": {
            "q_proj": 24.5, "k_proj": 3.5, "v_proj": 3.5, "o_proj": 24.5,
            "gate_proj": 129.5, "up_proj": 129.5, "down_proj": 129.5,
        },
        "full_model_mib": 14525.6,  # measured from safetensors
        "num_layers": 28,
    },
}

_MEM_SRC = ("distinct-tensor recipes over ops_step CSVs; sizes/full-model MiB "
            "from FAMILY_PROFILES (measured from safetensors)")

# Internal math stays in MiB (measured byte-exact from safetensors; binary units
# match nvidia-smi/du). All USER-FACING outputs convert to decimal MB (user
# decision 2026-07-11): value_MB = value_MiB * MIB_TO_MB. Percentages are
# unit-invariant.
MIB_TO_MB = 1.048576

# ops_step CSV component names -> merge-spec output names, per block.
_OPS_ATTN = ("attn_q", "attn_k", "attn_v", "attn_o")
_OPS_MLP = ("mlp_gate", "mlp_up", "mlp_down")
_OPS_GROUP = {**{c: "attn" for c in _OPS_ATTN}, **{c: "mlp" for c in _OPS_MLP}}
_OPS_TO_OUTPUT = {
    "attn_q": "q_proj", "attn_k": "k_proj", "attn_v": "v_proj",
    "attn_o": "o_proj",
    "mlp_gate": "gate_proj", "mlp_up": "up_proj", "mlp_down": "down_proj",
}
# output name -> jsonl/sidecar spec component name ("q_proj" -> "attn.q_proj"),
# the vocabulary of formatter.OPS_COMPONENT_TO_OUTPUT, run_eval's sidecar keys
# and the B/C.jsonl entries.
_OUTPUT_TO_SPEC = {out: f"{_OPS_GROUP[ops]}.{out}" for ops, out in _OPS_TO_OUTPUT.items()}


# --- scaling-factor sidecars (row_affine_v1; written by micr/run_eval.py) ----
# Per-run file next to steps.csv: keys "L{layer}.{spec_component}.{mu|sigma
# |achieved_mu|achieved_sigma}" + a JSON "__metadata__" byte array. Scaled
# merge groups share the UNSCALED canonical tensor at serving time; each
# member re-applies its per-row factors at load. Members/groups absent from a
# sidecar are identity (unscaled).
SIDECAR_BASENAME = "scaling_factors.npz"


def _load_scaling_sidecar(member: dict):
    """Load one member's run sidecar.

    Returns (slots, meta) where slots maps (layer:int, spec_component:str) ->
    {"mu","sigma"[,"achieved_mu","achieved_sigma"],"step_idx","cause"}, or None
    when the run has no sidecar (unscaled run, or a recorded run that predates
    factor capture).
    """
    path = member["steps_csv"].parent / SIDECAR_BASENAME
    if not path.exists():
        return None
    import numpy as np
    data = np.load(path)
    meta = {}
    if "__metadata__" in data.files:
        try:
            meta = json.loads(bytes(bytearray(data["__metadata__"])).decode("utf-8"))
        except Exception:
            meta = {}
    slots_meta = meta.get("slots", {})
    slots: dict = {}
    for name in data.files:
        if not name.endswith(".mu") or name.endswith(".achieved_mu"):
            continue
        prefix = name[: -len(".mu")]
        try:
            l_part, comp = prefix.split(".", 1)
            layer = int(l_part[1:])
        except (ValueError, IndexError):
            continue
        rec = {"mu": data[name], "sigma": data[prefix + ".sigma"]}
        if prefix + ".achieved_mu" in data.files:
            rec["achieved_mu"] = data[prefix + ".achieved_mu"]
            rec["achieved_sigma"] = data[prefix + ".achieved_sigma"]
        sm = slots_meta.get(prefix, {})
        rec["step_idx"] = sm.get("step_idx")
        rec["cause"] = sm.get("achieved_cause")
        slots[(layer, comp)] = rec
    return slots, meta


def _row_scaled(step_row: dict) -> bool:
    """steps.csv 'scaled' column (absent in pre-feature CSVs -> False)."""
    return str(step_row.get("scaled", "")).strip().lower() == "true"


def _load_ops_rows(ops_dir: Path, label: str) -> list[dict]:
    """One model's merge proposals from ops_step1_/ops_step2_<label>.csv, in
    file order, tagged with their stage."""
    out: list[dict] = []
    for stage, name in ((1, f"ops_step1_{label}.csv"), (2, f"ops_step2_{label}.csv")):
        path = ops_dir / name
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("op") != "merge":
                    continue
                out.append({"stage": stage, "component": r["component"],
                            "layer": int(r["layer"]), "models": r["models"]})
    return out


def _ops_blocks(ops_rows: list[dict]) -> list[dict]:
    """Group contiguous ops rows into (stage, layer, attn|mlp) blocks.

    One block corresponds to one MICR step (steps.csv logs one attn/mlp
    decision per step); `idx` are the indices of the block's ops rows.
    """
    blocks: list[dict] = []
    i, n = 0, len(ops_rows)
    while i < n:
        stage, layer = ops_rows[i]["stage"], ops_rows[i]["layer"]
        j = i
        while j < n and ops_rows[j]["stage"] == stage and ops_rows[j]["layer"] == layer:
            j += 1
        k = i
        while k < j:
            grp = _OPS_GROUP[ops_rows[k]["component"]]
            idx = []
            while k < j and _OPS_GROUP[ops_rows[k]["component"]] == grp:
                idx.append(k)
                k += 1
            blocks.append({"layer": layer, "group": grp, "idx": idx})
        i = j
    return blocks


def _parse_recipe(models_str, default_layer: int) -> list[tuple[str, int]]:
    """Parse an ops row's 'models' field into (contributor_label, layer) pairs;
    entries without an explicit ':layer' default to the row's layer."""
    parts = []
    for p in str(models_str).split(","):
        if not p.strip():
            continue
        name = p.strip().split(":")[0].strip()
        if ":" in p and p.split(":")[1].strip().isdigit():
            layer = int(p.split(":")[1])
        else:
            layer = default_layer
        parts.append((name, layer))
    return parts


def _load_step_rows(steps_csv: Path) -> list[dict]:
    """steps.csv rows with step_idx >= 0 (skips the baseline row), in order.
    These align 1:1 with the model's ops blocks."""
    with open(steps_csv, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if int(r["step_idx"]) >= 0]


def recipes_at(members: list[dict], cutoffs: dict[str, int],
               scaled_out: dict | None = None) -> dict:
    """Final merge recipe per (label, layer, component) at the given per-model
    step_idx cutoffs.

    Walks each member's ops blocks zipped with its steps.csv rows (same order:
    one steps row per block, stage 1 then stage 2). For each ACCEPTED step
    within the cutoff, every sub-component ops row that includes the model
    itself writes recipe = frozenset of (contributor_label, contributor_layer)
    for that slot. Later writes overwrite earlier ones, so the LAST accepted
    write within the cutoff wins (stage-aware).

    scaled_out: optional dict filled with (label, layer, component) -> bool,
    whether the final accepted write at the slot (within the cutoff) used
    row-affine scaling (steps.csv 'scaled' column; pre-feature CSVs -> False).
    """
    rec: dict = {}
    for m in members:
        lbl = m["label"]
        cutoff = cutoffs[lbl]
        for blk, step in zip(m["blocks"], m["step_rows"]):
            if step["decision"] != "accepted" or int(step["step_idx"]) > cutoff:
                continue
            comps = _OPS_ATTN if blk["group"] == "attn" else _OPS_MLP
            for comp in comps:
                for i in blk["idx"]:
                    row = m["ops_rows"][i]
                    if row["component"] != comp:
                        continue
                    parts = _parse_recipe(row["models"], blk["layer"])
                    if not any(name == lbl for name, _ in parts):
                        continue
                    rec[(lbl, blk["layer"], _OPS_TO_OUTPUT[comp])] = frozenset(parts)
                    if scaled_out is not None:
                        scaled_out[(lbl, blk["layer"], _OPS_TO_OUTPUT[comp])] = (
                            _row_scaled(step))
    return rec


def freed_distinct_mib(members: list[dict], cutoffs: dict[str, int]) -> float:
    """DISTINCT-TENSOR freed MiB for the pool at the given per-model cutoffs.

    Per (layer, component) slot, models sharing the same final recipe store one
    tensor instead of k, freeing (k-1) * component_size. Models with no
    accepted write at a slot keep their own tensor (unique sentinel identity).
    Cross-family recipes can never match (contributor labels are
    family-specific), so every k>=2 group is single-family and priced with
    that family's component size.

    Honest accounting for SCALED shared slots: the members of a scaled k>=2
    group share the UNSCALED canonical tensor, not the baked bytes, and each
    member must keep its per-row factor vectors resident to reconstruct its
    own distribution at load (fp32 mu+sigma, plus achieved_mu/achieved_sigma
    where shipped). Those sidecar bytes are subtracted from the freed total --
    computed exactly from the stored vector sizes. Unscaled slots, runs
    without a sidecar (all recorded runs), and pre-feature steps.csv files
    (no 'scaled' column) subtract exactly zero, so historical numbers --
    including VALIDATION_REFERENCE -- are unchanged.
    """
    scaled_at: dict = {}
    rec = recipes_at(members, cutoffs, scaled_out=scaled_at)
    sidecars = {m["label"]: m.get("scaling_sidecar") for m in members}
    slots: dict = defaultdict(list)  # (layer, comp) -> [(family, identity, label)]
    for m in members:
        fam = FAMILY_PROFILES[m["family"]]
        for layer in range(fam["num_layers"]):
            for comp in fam["components"]:
                r = rec.get((m["label"], layer, comp))
                ident = r if r is not None else ("__own__", m["label"], layer, comp)
                slots[(layer, comp)].append((m["family"], ident, m["label"]))
    freed = 0.0
    for (layer, comp), entries in slots.items():
        groups: dict = defaultdict(list)
        for family, ident, label in entries:
            groups[ident].append((family, label))
        for ident, mems in groups.items():
            if isinstance(ident, frozenset) and len(mems) >= 2:
                # same recipe -> same contributors -> same family
                freed += (len(mems) - 1) * FAMILY_PROFILES[mems[0][0]]["components"][comp]
                for _family, label in mems:
                    if not scaled_at.get((label, layer, comp)):
                        continue
                    sc = sidecars.get(label)
                    rec_s = sc[0].get((layer, _OUTPUT_TO_SPEC[comp])) if sc else None
                    if rec_s is None:
                        continue  # factors not on disk (recorded run): nothing shipped
                    nbytes = rec_s["mu"].nbytes + rec_s["sigma"].nbytes
                    if ("achieved_mu" in rec_s and rec_s.get("step_idx") is not None
                            and int(rec_s["step_idx"]) <= cutoffs[label]):
                        nbytes += (rec_s["achieved_mu"].nbytes
                                   + rec_s["achieved_sigma"].nbytes)
                    freed -= nbytes / (1 << 20)
    return freed


# --- pool configurations ----------------------------------------------------
POOLS = {
    "deepseek": {
        # label -> result subdir under run_dir/
        "models": {
            "deepseek-coder-7b-instruct-v1.5": "deepseek_coder",
            "deepseek-math-7b-instruct": "deepseek_math",
        },
        # MICR run that has BOTH models (profiling run); source of accept/scores.
        "run_dir": _REPO_ROOT / "micr/results/6_deepseek_standardized",
        # ops_step CSVs (merge proposals at sub-component granularity).
        "ops_dir": _REPO_ROOT / "clustering/candidates/6_deepseek_standardized",
        # Paper-reported baseline accuracies (HARDCODED_BASELINE_SCORES).
        "baselines": {
            "deepseek-coder-7b-instruct-v1.5": 65.85,
            "deepseek-math-7b-instruct": 81.4,
        },
        "memory_profile": "deepseek-7b",
        "num_layers": 30,
        # Operating points. kind="mem_ge": smallest cutoff whose pool savings %
        #   is >= target (best accuracy at that savings level); optional range
        #   [lo, hi] just records/checks the intended band.
        # kind="acc": deepest cutoff whose worst per-model accuracy drop <= budget.
        "points": [
            # NOTE: savings % is now the corrected distinct-tensor / full-size
            # metric, so these targets select different cutoffs than under the
            # old participant-count / attn+mlp model (B band 3-4% and C ~6.7%
            # were tuned on the old metric; mem_near still picks the cutoff
            # whose corrected savings is closest to the target).
            {"name": "B", "kind": "mem_near", "target": 3.5, "range": [3.0, 4.0]},
            {"name": "C", "kind": "mem_near", "target": 6.7},
        ],
        "out_dir": _SCRIPT_DIR / "operating_points_deepseek",
    },
    "7_llamas": {
        # 5 Llama task-specialist models sharing one MICR profiling run.
        "models": {
            "Llama-3.1-Hawkish-8B": "hawkish",
            "calme-2.3-legalkit-8b": "legalkit",
            "Llama-SafetyGuard-Content-Binary": "safetyguard",
            "Llama-3.1-8B-Instruct-multi-truth-judge": "truthjudge",
            "Llama-3.1-8B-UltraMedical": "ultramedical",
        },
        "run_dir": _REPO_ROOT / "micr/results/7_llamas_standardized",
        "ops_dir": _REPO_ROOT / "clustering/candidates/7_llamas_standardized",
        # Literal fallback baselines (used only if merge_tools.baselines returns
        # None); these mirror each model's MICR run baseline score.
        "baselines": {
            "Llama-3.1-Hawkish-8B": 54.39,
            "calme-2.3-legalkit-8b": 50.2,
            "Llama-SafetyGuard-Content-Binary": 88.99,
            "Llama-3.1-8B-Instruct-multi-truth-judge": 73.25,
            "Llama-3.1-8B-UltraMedical": 77.21,
        },
        "memory_profile": "llama-8b",
        "num_layers": 32,
        "points": [
            {"name": "B", "kind": "mem_near", "target": 3.5, "range": [3.0, 4.0]},
            {"name": "C", "kind": "mem_near", "target": 6.7},
        ],
        "out_dir": _SCRIPT_DIR / "operating_points_7_llamas",
    },
    # Composed pool: all 5 Llamas + both DeepSeeks share storage. Cross-family
    # recipes can never match, so freed memory is additive across the member
    # pools; the denominator is the sum of all 7 members' full model sizes.
    "7_llamas+deepseek": {
        "compose": ["7_llamas", "deepseek"],
        "points": [
            {"name": "B", "kind": "mem_near", "target": 3.5, "range": [3.0, 4.0]},
            {"name": "C", "kind": "mem_near", "target": 6.7},
        ],
        "out_dir": _SCRIPT_DIR / "operating_points_7_llamas_deepseek",
    },
}

# Verified reference: the corrected memory model must reproduce 34.5% (+/-0.1)
# for the 7_llamas pool at these per-model step_idx cutoffs.
VALIDATION_REFERENCE = {
    "7_llamas": {
        "cutoffs": {
            "Llama-3.1-8B-UltraMedical": 51,                 # Med
            "Llama-3.1-8B-Instruct-multi-truth-judge": 56,   # Truth
            "Llama-SafetyGuard-Content-Binary": 66,          # Safe
            "calme-2.3-legalkit-8b": 59,                     # Legal
            "Llama-3.1-Hawkish-8B": 35,                      # Fin
        },
        "expected_pct": 34.5,
        "tol": 0.1,
    },
}


def _coerce_pool_paths(cfg: dict) -> dict:
    """Turn string run_dir/ops_dir/out_dir in a user-supplied pool config into
    Paths (relative paths resolved against the repo root)."""
    def _p(v):
        p = Path(v)
        return p if p.is_absolute() else (_REPO_ROOT / p)
    for key in ("run_dir", "ops_dir", "out_dir"):
        if key in cfg and not isinstance(cfg[key], Path):
            cfg[key] = _p(cfg[key])
    return cfg


def load_pool_config(path: str) -> None:
    """Merge a JSON pool-definition file into POOLS (additive/override).

    The file maps pool_name -> pool_cfg with the same shape as the built-in
    POOLS entries. run_dir/ops_dir/out_dir may be given as strings.
    """
    if not path:
        return
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Optional top-level "_family_profiles": {family: {components, full_model_mib,
    # num_layers}} lets a config introduce families (e.g. qwen3-32b) without
    # editing this file. Merged before pools so leaf pools can reference them.
    for fam, prof in (data.pop("_family_profiles", None) or {}).items():
        FAMILY_PROFILES[fam] = prof
    for name, cfg in data.items():
        POOLS[name] = _coerce_pool_paths(dict(cfg))


def effective_pool(pool_name: str) -> dict:
    """Resolve a pool (possibly composed) into an effective config with a flat
    `members` list: [{label, family, steps_csv, ops_dir}, ...].

    A composed pool ({"compose": [subpool, ...]}) concatenates its subpools'
    members/models/baselines; each member keeps its own family, run and ops
    locations, so mixed-family pools work (recipes never match across
    families, and each family is priced with its own component sizes).
    """
    cfg = dict(POOLS[pool_name])
    if "compose" in cfg:
        members, models, baselines = [], {}, {}
        for sub in cfg["compose"]:
            sub_cfg = effective_pool(sub)
            members += sub_cfg["members"]
            models.update(sub_cfg["models"])
            baselines.update(sub_cfg.get("baselines") or {})
        cfg["members"], cfg["models"], cfg["baselines"] = members, models, baselines
        return cfg
    family = cfg["memory_profile"]
    if family not in FAMILY_PROFILES:
        raise KeyError(f"unknown family {family!r}; known: {', '.join(FAMILY_PROFILES)}")
    cfg["members"] = [
        {"label": label, "family": family,
         "steps_csv": cfg["run_dir"] / subdir / "steps.csv",
         "ops_dir": cfg["ops_dir"]}
        for label, subdir in cfg["models"].items()
    ]
    return cfg


def load_member_recipe_inputs(members: list[dict]) -> None:
    """Attach ops_rows/blocks/step_rows (the recipe-model inputs) to each member."""
    for m in members:
        m["ops_rows"] = _load_ops_rows(m["ops_dir"], m["label"])
        m["blocks"] = _ops_blocks(m["ops_rows"])
        m["step_rows"] = _load_step_rows(m["steps_csv"])
        # scaling-factor sidecar (None for unscaled / recorded runs)
        m["scaling_sidecar"] = _load_scaling_sidecar(m)


def load_pool_proposals(members: list[dict]) -> list:
    """formatter.load_merge_proposals per distinct ops_dir, concatenated.

    Proposals are per-family by construction (an ops_dir only names its own
    family's models), so concatenating lists is exact for composed pools.
    """
    proposals = []
    for ops_dir in dict.fromkeys(m["ops_dir"] for m in members):
        labels = [m["label"] for m in members if m["ops_dir"] == ops_dir]
        proposals += F.load_merge_proposals(ops_dir, labels)
    return proposals


def pool_full_size_mib(members: list[dict]) -> float:
    """Pool denominator: FULL on-disk size (measured) summed over members."""
    return sum(FAMILY_PROFILES[m["family"]]["full_model_mib"] for m in members)


def load_model_steps_from_run(steps_csv: Path):
    """Load one model's MICR steps.csv (component is 'attn'/'mlp', has scores).

    Returns (decisions, rows, baseline_score, score_by_step_idx) matching the
    structures formatter.build_spec expects.
    """
    decisions: dict[tuple[str, int], str] = {}
    rows: list[tuple[tuple[str, int], str, int, int | None]] = []
    baseline: float | None = None
    score_by_idx: dict[int, float] = {}
    with open(steps_csv, newline="", encoding="utf-8") as f:
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            decision = row["decision"].strip().lower()
            score = None
            try:
                score = float(row["score"])
            except (KeyError, ValueError, TypeError):
                pass
            if decision == "baseline":
                baseline = score
                continue
            try:
                layer = int(row["layer"])
                comp = row["component"].strip()
            except (KeyError, ValueError):
                continue
            try:
                step_idx: int | None = int(row["step_idx"])
            except (KeyError, ValueError, TypeError):
                step_idx = None
            key = (comp, layer)
            rows.append((key, decision, line_no, step_idx))
            if decision == "accepted":
                decisions[key] = "accepted"
                if step_idx is not None and score is not None:
                    score_by_idx[step_idx] = score
            elif key not in decisions:
                decisions[key] = decision
    return decisions, rows, baseline, score_by_idx


# --per-model: also emit per-model-cutoff twin points (Bpm, Cpm) + the knee
# (Kpm). Each model takes its OWN deepest cutoff within budget, so a fragile
# member (e.g. deepseek in a llama+ds pool) no longer caps a robust one (llama).
# Additive: the global B/C are untouched. Savings are still computed by
# freed_distinct_mib, which only credits a slot where >=2 members land on the
# same recipe.
PER_MODEL = False

# --paper-mem T: also emit a "paper-closest" point P = the (global) cutoff whose
# pool savings is closest to the paper's reported memory reduction T% for this
# figure. None => not emitted (sets with no paper number, e.g. within-pairs).
PAPER_MEM = None


def score_at_cutoff(rows, score_by_idx, baseline, cutoff):
    """Most recent accepted eval score at step_idx <= cutoff (else baseline)."""
    best = baseline
    for (_key, decision, _line, step_idx) in rows:
        if decision == "accepted" and step_idx is not None and step_idx <= cutoff:
            if step_idx in score_by_idx:
                best = score_by_idx[step_idx]
    return best


def build_pool(pool_name: str):
    cfg = effective_pool(pool_name)
    members = cfg["members"]
    models = list(cfg["models"])
    baselines = resolve_baselines(cfg)

    proposals = load_pool_proposals(members)
    load_member_recipe_inputs(members)

    dec, rows, run_base, score_idx = {}, {}, {}, {}
    for m in members:
        label = m["label"]
        dec[label], rows[label], run_base[label], score_idx[label] = (
            load_model_steps_from_run(m["steps_csv"])
        )

    # pool denominator: FULL on-disk model size (measured), all N members
    pool_total_mb = pool_full_size_mib(members)

    # Keep the un-anchored profiling view for the report (returned as the 4th
    # element, display only); anchor the working dict for all drop arithmetic.
    # (For the recorded 7_llamas pool the two are byte-equal, so --validate
    # arithmetic is unchanged.)
    profiling_view = dict(baselines)
    _anchor_baselines(models, baselines, run_base)

    def _cut_map(cutoff):
        # cutoff may be a single int (broadcast to every model) OR a per-model
        # dict {label: cutoff} (--per-model points).
        return dict(cutoff) if isinstance(cutoff, dict) else {m: cutoff for m in models}

    def spec_at(cutoff):
        spec = F.build_spec(
            proposals, dec, rows, models,
            _cut_map(cutoff), use_cutoff=True, cutoff_mode="step_idx",
        )
        return F.drop_subset_merged_layers(spec)

    def spec_at_full(cutoff):
        # Full per-tensor recipe INCLUDING single-model (touched-but-unmerged)
        # groups -- emitted to the {point}.json / {point}.jsonl specs. The
        # savings model uses the merges-only spec_at() above; this is output-only.
        spec = F.build_spec(
            proposals, dec, rows, models,
            _cut_map(cutoff), use_cutoff=True, cutoff_mode="step_idx",
            keep_singletons=True,
        )
        return F.drop_subset_merged_layers(spec)

    def freed_at(cutoff) -> float:
        # corrected distinct-tensor numerator
        return freed_distinct_mib(members, _cut_map(cutoff))

    def worst_drop(cutoff: int):
        return max(
            baselines[m] - score_at_cutoff(rows[m], score_idx[m], run_base[m], cutoff)
            for m in models
        )

    max_cutoff = max(
        max((si for (_k, _d, _l, si) in rows[m] if si is not None), default=0)
        for m in models
    )

    # savings % at every cutoff (monotonic non-decreasing in cutoff)
    save_pct = {c: 100 * freed_at(c) / pool_total_mb
                for c in range(0, max_cutoff + 1)}

    def deepest_within_budget(budget: float):
        best = None  # None => even cutoff 0 violates the budget
        for c in range(0, max_cutoff + 1):
            if worst_drop(c) <= budget:
                best = c
        return best

    def per_model_cutoffs(budget: float) -> dict:
        # For EACH model independently: the deepest cutoff at which THAT model's
        # own drop stays within budget (same drop reference as worst_drop). A
        # model whose drop exceeds budget even at cutoff 0 gets -1 (unmerged, so
        # recipes_at applies none of its merges). Non-monotone drops are handled
        # by taking the deepest safe cutoff, not the first violation.
        out = {}
        for m in models:
            best = -1
            for c in range(0, max_cutoff + 1):
                drop = baselines[m] - score_at_cutoff(
                    rows[m], score_idx[m], run_base[m], c)
                if drop <= budget:
                    best = c
            out[m] = best
        return out

    def per_model_knee(max_budget: float = 2.0, step: float = 0.05):
        # "Sweet spot" within max_budget: sweep the per-model accuracy budget and
        # take the Pareto knee of savings(budget) -- the elbow beyond which extra
        # savings taper off. savings(budget) is monotone non-decreasing (a larger
        # budget only lets each model merge >= as deep), so the knee is the
        # normalized point furthest above the chord (Kneedle). Returns the
        # per-model cutoff dict at that budget.
        n = int(round(max_budget / step))
        grid = [i * step for i in range(n + 1)]
        cuts_at = [per_model_cutoffs(b) for b in grid]
        sav = [100 * freed_at(c) / pool_total_mb for c in cuts_at]
        x0, x1 = grid[0], grid[-1]
        y0, y1 = min(sav), max(sav)
        if x1 <= x0 or y1 <= y0:
            return cuts_at[-1]
        def _above_chord(i):
            nx = (grid[i] - x0) / (x1 - x0)
            ny = (sav[i] - y0) / (y1 - y0)
            return ny - nx
        return cuts_at[max(range(len(grid)), key=_above_chord)]

    def smallest_cutoff_mem_ge(target: float) -> int:
        for c in range(0, max_cutoff + 1):
            if save_pct[c] >= target - 1e-9:
                return c
        return max_cutoff  # target unreachable -> deepest

    def cutoff_mem_near(target: float) -> int:
        # closest savings to target; tie-break toward fewer merges (smaller cutoff)
        return min(range(0, max_cutoff + 1),
                   key=lambda c: (abs(save_pct[c] - target), c))

    def _point_from_cutoff(name, crit, kind, cutoff, range_=None, warning=None):
        # cutoff: int (global) OR dict {label: cutoff} (per-model). Assembles the
        # point dict; spec_at/spec_at_full/freed_at accept either form.
        spec = [g for g in spec_at(cutoff) if len(g) >= 2]
        spec_full = _to_self_attn(spec_at_full(cutoff))
        scores = {m: score_at_cutoff(
            rows[m], score_idx[m], run_base[m],
            cutoff[m] if isinstance(cutoff, dict) else cutoff) for m in models}
        freed = freed_at(cutoff)
        pt = {
            "name": name, "criterion": crit, "kind": kind,
            "cutoff": cutoff, "spec": spec, "spec_full": spec_full,
            "savings_mb": freed,
            "savings_pct": 100 * freed / pool_total_mb,
            "scores": scores,
            # drops anchored to the MICR run baseline (same eval as `scores`)
            "drops": {m: run_base[m] - scores[m] for m in models},
            "range": range_,
        }
        if warning:
            pt["warning"] = warning
        return pt

    def build_point(pspec: dict) -> dict:
        budget_warning = None
        if pspec["kind"] == "acc":
            cutoff = deepest_within_budget(pspec["budget"])
            crit = f"acc drop <= {pspec['budget']}%"
            if cutoff is None:
                cutoff = 0
                budget_warning = (
                    f"NO cutoff satisfies acc budget {pspec['budget']}% "
                    f"(worst drop at cutoff 0 is {worst_drop(0):.2f}%); "
                    f"reporting cutoff 0, which VIOLATES the budget")
        elif pspec["kind"] == "mem_ge":
            cutoff = smallest_cutoff_mem_ge(pspec["target"])
            crit = f"mem savings >= {pspec['target']}%"
        elif pspec["kind"] == "mem_near":
            cutoff = cutoff_mem_near(pspec["target"])
            crit = f"mem savings ~ {pspec['target']}%"
        else:
            raise ValueError(f"unknown point kind: {pspec['kind']}")
        pt = _point_from_cutoff(pspec["name"], crit, pspec["kind"], cutoff,
                                pspec.get("range"), budget_warning)
        # sanity: flag if savings falls outside the declared band (granularity gap)
        if pt["range"]:
            lo, hi = pt["range"]
            if not (lo - 1e-6 <= pt["savings_pct"] <= hi + 1e-6):
                side = "below" if pt["savings_pct"] < lo else "above"
                pt["warning"] = (f"no cutoff lands in [{lo},{hi}]% (granularity gap); "
                                 f"nearest is {pt['savings_pct']:.2f}% ({side} the band)")
        return pt

    def build_point_permodel(pspec: dict) -> dict:
        # Per-model twin of an "acc" point: each model at its own deepest
        # in-budget cutoff. Decouples fragile members from robust ones.
        cuts = per_model_cutoffs(pspec["budget"])
        return _point_from_cutoff(
            pspec["name"], f"per-model acc drop <= {pspec['budget']}%",
            "acc_permodel", cuts)

    def build_point_knee(pspec: dict) -> dict:
        # Per-model "sweet spot": the savings/accuracy knee within a 2% envelope.
        cuts = per_model_knee(pspec.get("max_budget", 2.0))
        return _point_from_cutoff(
            pspec["name"], "per-model knee (savings/accuracy sweet spot, <=2%)",
            "acc_knee", cuts)

    points = [{
        "name": "A", "criterion": "unmerged reference", "kind": "ref",
        "cutoff": -1, "spec": [], "spec_full": [], "savings_mb": 0.0, "savings_pct": 0.0,
        "scores": {m: run_base[m] for m in models},
        "drops": {m: 0.0 for m in models}, "range": None,
    }]
    points += [build_point(p) for p in cfg["points"]]
    if PER_MODEL:
        for p in cfg["points"]:
            if p.get("kind") == "acc":
                points.append(build_point_permodel({**p, "name": p["name"] + "pm"}))
        # Point 3: per-model knee (savings/accuracy sweet spot within 2%).
        if any(p.get("kind") == "acc" for p in cfg["points"]):
            points.append(build_point_knee({"name": "Kpm"}))
    # Point 4: paper-closest -- the global cutoff whose savings is nearest the
    # paper's reported memory reduction for this figure (mem_near). Emitted only
    # when a paper target is supplied (--paper-mem); within-pairs have none.
    if PAPER_MEM is not None:
        points.append(build_point(
            {"name": "P", "kind": "mem_near", "target": PAPER_MEM}))
    # run_base = MICR baseline (drop reference, same eval as scores)
    # profiling_view = profiling/"paper" baseline (un-anchored), reference only
    return cfg, models, run_base, profiling_view, pool_total_mb, points


def _short(model: str) -> str:
    parts = model.split("-")
    return parts[1] if len(parts) > 1 else model


def _group_str(group) -> str:
    """Human-readable membership of one merged group.

    Same layer for all members -> 'L<layer>(<models>)'.
    Cross-layer merge          -> '<m1>:L<l1>+<m2>:L<l2>'.
    """
    layers = {e["layer"] for e in group}
    if len(layers) == 1:
        members = "+".join(sorted(_short(e["model"]) for e in group))
        return f"L{next(iter(layers))}({members})"
    return "+".join(f"{_short(e['model'])}:L{e['layer']}"
                    for e in sorted(group, key=lambda e: _short(e["model"])))


def _to_self_attn(spec):
    """Emitted-spec component naming: attn.q_proj -> self_attn.q_proj (the HF
    state-dict submodule path); mlp.* unchanged. Applied only to the
    {point}.json / {point}.jsonl output, not to the internal merges-only spec
    used for savings/report/sidecar lookups."""
    return [[{**e, "component": ("self_" + e["component"])
              if e["component"].startswith("attn.") else e["component"]}
             for e in g] for g in spec]


def merging_summary(spec):
    """Per component, the merged groups (>=2 models) only; these create the savings."""
    by = defaultdict(list)
    for group in spec:
        if len(group) >= 2:
            by[group[0]["component"]].append(_group_str(group))
    return {c: sorted(by[c]) for c in sorted(by)}


def _write_point_sidecar(members: list[dict], pt: dict, path: Path):
    """Per-point scaling-factor sidecar, parallel to {name}.jsonl.

    The jsonl itself stays BYTE-COMPATIBLE (the external serving module
    consumes it unchanged); factors ship in this separate npz. For each k>=2
    group in the point's spec, every member whose final accepted write at the
    slot (within the point's cutoff) was SCALED gets its per-row factors
    pulled from its run's scaling_factors.npz, keyed
    "{label}|L{layer}.{spec_component}.{mu|sigma|...}" -- joinable 1:1 against
    the jsonl's {"model","layer","component"} entries. Absence of a member/
    group from this file means IDENTITY (unscaled: use the stored tensor
    as-is). Achieved stats are shipped only when the run-final write they
    describe lies within this point's cutoff (an earlier cutoff may precede
    the write that diverged).

    Returns (n_slots, missing) -- missing lists scaled slots whose run has no
    sidecar on disk (recorded runs predating factor capture); those members
    fall back to identity and are flagged in the metadata.
    """
    import numpy as np
    by_label = {m["label"]: m for m in members}
    scaled_at: dict = {}
    _cut = pt["cutoff"]  # per-model points already carry a {label: cutoff} dict
    recipes_at(members,
               dict(_cut) if isinstance(_cut, dict)
               else {m["label"]: _cut for m in members},
               scaled_out=scaled_at)
    arrays: dict = {}
    slot_meta: dict = {}
    missing: list[str] = []
    for group in pt["spec"]:
        if len(group) < 2:
            continue
        for e in group:
            label, layer, comp_spec = e["model"], int(e["layer"]), e["component"]
            comp_out = comp_spec.split(".", 1)[1]  # "attn.q_proj" -> "q_proj"
            if not scaled_at.get((label, layer, comp_out)):
                continue  # unscaled at this cutoff -> identity, no entry
            sc = by_label[label].get("scaling_sidecar")
            rec = sc[0].get((layer, comp_spec)) if sc else None
            if rec is None:
                missing.append(f"{label}:L{layer}.{comp_spec}")
                continue
            # Key the point-sidecar with the SAME self_attn.* naming the emitted
            # .json/.jsonl now use, so factors still join 1:1 against the spec.
            # (comp_spec above stays attn.* for the run-sidecar lookup, which is
            # keyed by the run's original naming.)
            spec_comp = ("self_" + comp_spec) if comp_spec.startswith("attn.") else comp_spec
            prefix = f"{label}|L{layer}.{spec_comp}"
            arrays[prefix + ".mu"] = rec["mu"]
            arrays[prefix + ".sigma"] = rec["sigma"]
            sm: dict = {"step_idx": rec.get("step_idx")}
            _lcut = pt["cutoff"][label] if isinstance(pt["cutoff"], dict) else pt["cutoff"]
            if ("achieved_mu" in rec and rec.get("step_idx") is not None
                    and int(rec["step_idx"]) <= _lcut):
                arrays[prefix + ".achieved_mu"] = rec["achieved_mu"]
                arrays[prefix + ".achieved_sigma"] = rec["achieved_sigma"]
                sm["achieved_cause"] = rec.get("cause")
            slot_meta[prefix] = sm
    if not arrays and not missing:
        return 0, []
    meta = {
        "format": "micr_point_scaling_factors",
        "version": 1,
        "transform": "row_affine_v1",
        "point": pt["name"],
        "cutoff": pt["cutoff"],
        # Keep in sync with micr/run_eval.py scaling_sidecar_metadata().
        "reconstruction": (
            "member = (X - rowmean(X)) * (sigma / rowstd(X)) + mu, per output "
            "row (stats over dim 1, float32, unbiased=False), then cast to "
            "bfloat16. X is ANY stored representation of the merged slot -- "
            "the unscaled canonical average or another member's baked scaled "
            "tensor -- with rowmean/rowstd computed from X's stored bytes at "
            "load. Where achieved_mu/achieved_sigma are present, use them "
            "instead to reproduce the exact tensor MICR validated."
        ),
        "identity_rule": ("a (model, layer, component) absent from this file "
                          "is UNSCALED: use the stored merged tensor as-is"),
        "keys": "{model}|L{layer}.{component}.{mu|sigma|achieved_mu|achieved_sigma}",
        "slots": slot_meta,
        "missing_factors": missing,
    }
    arrays["__metadata__"] = np.frombuffer(
        json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8)
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, str(path))
    return len(slot_meta), missing


def write_outputs(pool_name: str):
    cfg, models, run_base, profiling_baselines, pool_total_mb, points = build_pool(pool_name)
    out_dir = cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSONL specs for B and C (one merged group per line) + parallel per-point
    # scaling-factor sidecars (jsonl bytes untouched; see _write_point_sidecar)
    point_sidecars: dict = {}
    for pt in points:
        if pt["name"] == "A":
            continue
        # Emitted spec in BOTH serializations of the SAME data (self_attn.*
        # naming): {point}.jsonl (one group per line) and {point}.json (single
        # pretty-printed array). Single-model groups are DROPPED -- a group of
        # one is not a shared tensor (it's the model's own), so the spec lists
        # only the actual merge groups (>=2 models).
        groups = [g for g in pt["spec_full"] if len(g) >= 2]
        with open(out_dir / f"{pt['name']}.jsonl", "w", encoding="utf-8") as f:
            for group in groups:
                f.write(json.dumps(group) + "\n")
        with open(out_dir / f"{pt['name']}.json", "w", encoding="utf-8") as f:
            json.dump(groups, f, indent=2)
        n_slots, missing = _write_point_sidecar(
            cfg["members"], pt, out_dir / f"{pt['name']}.scaling_factors.npz")
        if n_slots or missing:
            point_sidecars[pt["name"]] = (n_slots, missing)
            if missing:
                print(f"[scaling] WARNING {pt['name']}: {len(missing)} scaled "
                      f"slot(s) have no factors on disk (run predates capture); "
                      f"those members fall back to identity: "
                      f"{', '.join(missing[:5])}"
                      + (" ..." if len(missing) > 5 else ""))

    # report.csv
    csv_path = out_dir / "report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["point", "criterion", "cutoff", "groups", "merging_groups",
                  "savings_mb", "savings_pct"]
        for m in models:
            header += [f"{m}__baseline", f"{m}__score", f"{m}__drop", f"{m}__cutoff"]
        header += ["mean_drop"]
        w.writerow(header)
        for pt in points:
            drops = [pt["drops"][m] for m in models]
            n_merge = sum(1 for g in pt["spec"] if len(g) >= 2)
            cut = pt["cutoff"]
            # per-model points carry a {label: cutoff} dict; the scalar "cutoff"
            # column then reads "pm" and each model's own cutoff lands in its
            # __cutoff column (so finaleval can replay each model at its own).
            cut_cell = "pm" if isinstance(cut, dict) else cut
            row = [pt["name"], pt["criterion"], cut_cell, len(pt["spec"]), n_merge,
                   round(pt["savings_mb"] * MIB_TO_MB, 1), round(pt["savings_pct"], 2)]
            for m in models:
                mcut = cut[m] if isinstance(cut, dict) else cut
                row += [run_base[m], round(pt["scores"][m], 2),
                        round(pt["drops"][m], 2), mcut]
            row += [round(sum(drops) / len(drops), 2)]
            w.writerow(row)

    # report.md
    md_path = out_dir / "report.md"
    lines = []
    lines.append(f"# Operating points — {pool_name} pool ({' + '.join(models)})")
    lines.append("")
    members = cfg["members"]
    for run_dir in dict.fromkeys(m["steps_csv"].parent.parent for m in members):
        lines.append(f"- Source run: `{run_dir.relative_to(_REPO_ROOT)}` (profiling)")
    for ops_dir in dict.fromkeys(m["ops_dir"] for m in members):
        lines.append(f"- Merge proposals: `{ops_dir.relative_to(_REPO_ROOT)}`")
    fams = sorted({m["family"] for m in members})
    lines.append(f"- Memory model: {_MEM_SRC}; families: {', '.join(fams)}")
    lines.append(f"- Pool denominator (FULL on-disk model size, measured from "
                 f"safetensors, {len(models)} models): {pool_total_mb * MIB_TO_MB:.1f} MB")
    lines.append("- **Drop reference = MICR run baseline** (same eval as the merged "
                 "scores): "
                 + ", ".join(f"{_short(m)}={run_base[m]}" for m in models))
    lines.append("- Profiling/noise baseline (HARDCODED_BASELINE_SCORES, *different eval* "
                 "— shown for reference, NOT used for drops): "
                 + ", ".join(f"{_short(m)}={profiling_baselines[m]}" for m in models))
    lines.append("- Selection: "
                 + "; ".join(f"{pt['name']} = {pt['criterion']}"
                            for pt in points if pt["name"] != "A")
                 + " (savings is monotonic in cutoff; mem_ge picks the smallest "
                 "cutoff reaching the target = best accuracy at that savings).")
    lines.append("- Single-model (non-merge) groups are excluded from the jsonl: a group "
                 "with one model is not a merge and saves nothing.")
    if point_sidecars:
        lines.append("- **Scaled merge groups share the UNSCALED canonical tensor, not the "
                     "baked bytes**: each member re-applies its per-row scaling factors at "
                     "load (row_affine_v1; see <point>.scaling_factors.npz next to the "
                     "jsonl -- a member/group absent from that file is identity/unscaled). "
                     "The savings figures above already subtract the factor-vector bytes. "
                     + "; ".join(f"{n}: {c} factored slot(s)"
                                 + (f", {len(miss)} missing (recorded run predates "
                                    f"factor capture; identity fallback)" if miss else "")
                                 for n, (c, miss) in sorted(point_sidecars.items())))
    for pt in points:
        if pt.get("warning"):
            lines.append(f"- ⚠️ {pt['name']}: {pt['warning']}")
    lines.append("")
    hdr = "| Point | Criterion | Save MB | Save % | " + \
          " | ".join(f"{_short(m)} score (drop)" for m in models) + " | Mean drop |"
    sep = "|" + "---|" * (4 + len(models) + 1)
    lines.append(hdr)
    lines.append(sep)
    for pt in points:
        cells = [pt["name"], pt["criterion"],
                 f"{pt['savings_mb'] * MIB_TO_MB:.0f}", f"{pt['savings_pct']:.1f}%"]
        for m in models:
            cells.append(f"{pt['scores'][m]:.2f} ({pt['drops'][m]:+.2f})")
        mean = sum(pt["drops"][m] for m in models) / len(models)
        cells.append(f"{mean:+.2f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    for pt in points:
        if pt["name"] == "A":
            continue
        merging = merging_summary(pt["spec"])
        n_merge = sum(len(v) for v in merging.values())
        lines.append(f"## {pt['name']} — exact merged components "
                     f"({n_merge} merge groups, {pt['savings_pct']:.1f}% savings)")
        lines.append(f"_Each line in {pt['name']}.jsonl is one merged tensor group. "
                     f"`L4(coder+math)` = both at layer 4; `coder:L20+math:L22` = cross-layer._")
        for comp, groups in merging.items():
            lines.append(f"- `{comp}`: {', '.join(groups)}")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return out_dir, points, models, run_base


# ---------------------------------------------------------------------------
# Full-sweep / Pareto-frontier tooling (additive; leaves A/B/C output intact).
# ---------------------------------------------------------------------------
def sweep_pool(pool_name: str, accuracy_metric: str = "worst"):
    """Sweep every cutoff 0..max_cutoff for a pool.

    Memory uses the corrected distinct-tensor model (freed_distinct_mib over
    the full-size denominator); accuracy reuses the EXACT score computation of
    the A/B/C sampler (score_at_cutoff). The only new quantities are
    aggregates (mean-per-model drop) and derived views (accuracy = -drop).

    Returns (cfg, models, pool_total_mb, table). Each table row is a dict with:
      cutoff, steps, savings_pct, savings_mb (MiB), worst_drop, mean_drop,
      accuracy_metric, accuracy_drop, accuracy (= -accuracy_drop, higher better).
    """
    if accuracy_metric not in ("worst", "mean"):
        raise ValueError("accuracy_metric must be 'worst' or 'mean'")
    cfg = effective_pool(pool_name)
    members = cfg["members"]
    models = list(cfg["models"])
    baselines = resolve_baselines(cfg)

    load_member_recipe_inputs(members)

    rows, run_base, score_idx = {}, {}, {}
    dec = {}
    for m in members:
        label = m["label"]
        dec[label], rows[label], run_base[label], score_idx[label] = (
            load_model_steps_from_run(m["steps_csv"])
        )

    # Same anchor as build_pool: pipeline-generated pools have no literal/
    # central baselines, so without this every drop is None-arithmetic.
    _anchor_baselines(models, baselines, run_base)

    pool_total_mb = pool_full_size_mib(members)

    def per_model_drops(cutoff: int) -> dict:
        return {
            m: baselines[m] - score_at_cutoff(rows[m], score_idx[m], run_base[m], cutoff)
            for m in models
        }

    max_cutoff = max(
        max((si for (_k, _d, _l, si) in rows[m] if si is not None), default=0)
        for m in models
    )

    table = []
    for c in range(0, max_cutoff + 1):
        mb = freed_distinct_mib(members, {m: c for m in models})
        pct = 100 * mb / pool_total_mb
        drops = per_model_drops(c)
        worst = max(drops.values())            # existing worst_drop formula
        mean = sum(drops.values()) / len(drops)  # new: mean per-model drop
        acc_drop = worst if accuracy_metric == "worst" else mean
        table.append({
            "cutoff": c,
            "steps": c,
            "savings_pct": pct,
            "savings_mb": mb,
            "worst_drop": worst,
            "mean_drop": mean,
            "accuracy_metric": accuracy_metric,
            "accuracy_drop": acc_drop,
            "accuracy": -acc_drop,
            # per-model trajectory data for the companion plot / sweep CSV
            "model_drops": dict(drops),
            "model_scores": {m: baselines[m] - drops[m] for m in models},
        })
    return cfg, models, pool_total_mb, table, dict(baselines)


def mark_pareto_frontier(table: list) -> list:
    """Mark each row's 'frontier' flag via non-domination on (accuracy, savings).

    A point p is dominated if some other point q has accuracy >= p.accuracy AND
    savings_mb >= p.savings_mb, with strict inequality in at least one axis
    (so identical points don't mutually dominate). Returns the frontier subset,
    sorted by memory saved.
    """
    for i, p in enumerate(table):
        dominated = False
        for j, q in enumerate(table):
            if i == j:
                continue
            if (q["accuracy"] >= p["accuracy"] and q["savings_mb"] >= p["savings_mb"]
                    and (q["accuracy"] > p["accuracy"] or q["savings_mb"] > p["savings_mb"])):
                dominated = True
                break
        p["frontier"] = not dominated
    return sorted((p for p in table if p["frontier"]), key=lambda r: r["savings_mb"])


def write_sweep_csv(table: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["cutoff", "steps", "savings_pct", "savings_mb", "worst_drop",
            "mean_drop", "accuracy_metric", "accuracy_drop", "accuracy", "frontier"]
    # per-model trajectory columns (additive; DictReader consumers unaffected)
    labels = sorted((table[0].get("model_scores") or {}).keys()) if table else []
    cols += [f"{m}__score" for m in labels] + [f"{m}__drop" for m in labels]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in table:
            row = [
                r["cutoff"], r["steps"], round(r["savings_pct"], 4),
                round(r["savings_mb"] * MIB_TO_MB, 3), round(r["worst_drop"], 4),
                round(r["mean_drop"], 4), r["accuracy_metric"],
                round(r["accuracy_drop"], 4), round(r["accuracy"], 4),
                int(bool(r.get("frontier", False))),
            ]
            row += [round(r["model_scores"][m], 4) for m in labels]
            row += [round(r["model_drops"][m], 4) for m in labels]
            w.writerow(row)


def plot_pareto(table: list, frontier: list, out_png: Path, pool_name: str,
                accuracy_metric: str, label_cutoffs=None) -> None:
    """Scatter all operating points and overlay the Pareto frontier.

    matplotlib is imported here so the module (and the A/B/C path) keeps working
    without it installed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    xs = [p["savings_mb"] * MIB_TO_MB for p in table]
    ys = [p["accuracy"] for p in table]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xs, ys, s=22, color="0.72", zorder=2, label="all operating points")

    fr = sorted(frontier, key=lambda r: r["savings_mb"])
    ax.plot([p["savings_mb"] * MIB_TO_MB for p in fr], [p["accuracy"] for p in fr],
            "-o", color="C0", zorder=3, label="Pareto frontier")

    # Label a handful of cutoffs along the frontier (endpoints + evenly spaced).
    if label_cutoffs is None:
        if fr:
            n = len(fr)
            idxs = sorted({0, n - 1, n // 4, n // 2, (3 * n) // 4})
            to_label = [fr[i] for i in idxs if 0 <= i < n]
        else:
            to_label = []
    else:
        want = set(label_cutoffs)
        to_label = [p for p in fr if p["cutoff"] in want]
    for p in to_label:
        ax.annotate(f"c={p['cutoff']}",
                    (p["savings_mb"] * MIB_TO_MB, p["accuracy"]),
                    textcoords="offset points", xytext=(6, 5), fontsize=8,
                    color="C0")

    ax.set_xlabel("Memory saved (MB, distinct-tensor)")
    ax.set_ylabel(f"Accuracy  (-{accuracy_metric} per-model drop, %)")
    ax.set_title(f"Memory/accuracy operating points - {pool_name} pool")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def validate_pool(pool_name: str) -> dict:
    """Check the corrected memory model against a verified reference value.

    Computes distinct-tensor freed % at the reference per-model cutoffs and
    compares with the expected percentage (within tolerance). Only pools with
    an entry in VALIDATION_REFERENCE can be validated.
    """
    ref = VALIDATION_REFERENCE.get(pool_name)
    if ref is None:
        raise KeyError(f"no validation reference for pool {pool_name!r}; "
                       f"available: {', '.join(VALIDATION_REFERENCE) or '(none)'}")
    cfg = effective_pool(pool_name)
    members = cfg["members"]
    load_member_recipe_inputs(members)
    freed = freed_distinct_mib(members, dict(ref["cutoffs"]))
    total = pool_full_size_mib(members)
    pct = 100 * freed / total
    return {
        "pool": pool_name, "cutoffs": dict(ref["cutoffs"]),
        "freed_mib": freed, "total_mib": total, "savings_pct": pct,
        "expected_pct": ref["expected_pct"], "tol": ref["tol"],
        "ok": abs(pct - ref["expected_pct"]) <= ref["tol"],
    }


def plot_pareto_models(table: list, baselines: dict, out_png: Path,
                       pool_name: str, tolerance: float = 2.0) -> None:
    """Companion to plot_pareto: one line PER MODEL along the sweep, in the
    style of the plot_steps visualization (per-model score lines with dashed
    baselines), plus a drop panel against the acceptance tolerance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = sorted((table[0].get("model_scores") or {}).keys()) if table else []
    if not labels:
        return
    x = [r["savings_pct"] for r in table]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    cmap = plt.get_cmap("tab10")
    for i, m in enumerate(labels):
        color = cmap(i % 10)
        ax1.plot(x, [r["model_scores"][m] for r in table],
                 color=color, lw=1.6, label=m)
        ax1.axhline(baselines[m], color=color, lw=0.9, ls="--", alpha=0.55)
        ax2.plot(x, [r["model_drops"][m] for r in table], color=color, lw=1.4)
    ax2.plot(x, [r["worst_drop"] for r in table],
             color="black", lw=1.2, ls=":", label="worst drop")
    ax2.axhline(tolerance, color="red", lw=1.0, ls="--", alpha=0.8,
                label=f"tolerance {tolerance:.1f}")
    ax1.set_ylabel("score (eval split)")
    ax1.set_title(f"{pool_name}: per-model accuracy along the sweep "
                  f"(dashed = each model's run baseline)")
    ax1.legend(fontsize=7, loc="center left")
    ax2.set_xlabel("memory savings (%)")
    ax2.set_ylabel("drop vs baseline")
    ax2.legend(fontsize=8, loc="upper left")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def run_sweep(pool_name: str, accuracy_metric: str = "worst",
              plot_out=None, sweep_csv=None):
    """Full sweep -> Pareto -> CSV + PNGs. Returns (table, frontier, paths)."""
    cfg, models, pool_total_mb, table, baselines = sweep_pool(pool_name, accuracy_metric)
    frontier = mark_pareto_frontier(table)

    out_dir = cfg["out_dir"]
    csv_path = Path(sweep_csv) if sweep_csv else out_dir / f"sweep_{pool_name}.csv"
    png_path = Path(plot_out) if plot_out else out_dir / f"pareto_{pool_name}.png"
    models_png = png_path.with_name(png_path.stem + "_models" + png_path.suffix)

    write_sweep_csv(table, csv_path)
    plot_pareto(table, frontier, png_path, pool_name, accuracy_metric)
    plot_pareto_models(table, baselines, models_png, pool_name)
    return table, frontier, {"csv": csv_path, "png": png_path, "models_png": models_png}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # --pool-config may register additional pools, so parse it first and only
    # then validate --pool (no fixed `choices=`, since config can add names).
    ap.add_argument("--pool-config", default=None,
                    help="Optional JSON file mapping pool_name -> pool config "
                         "(same shape as built-in POOLS) to add/override pools.")
    ap.add_argument("--pool", default="deepseek",
                    help="Pool name (built-in: " + ", ".join(POOLS) + "; "
                         "or any defined via --pool-config).")
    ap.add_argument("--save-b", type=float, default=None,
                    help="Override B as a mem-savings target %% (smallest cutoff >= this).")
    ap.add_argument("--save-c", type=float, default=None,
                    help="Override C as a mem-savings target %% (smallest cutoff >= this).")
    # --- full-sweep / Pareto-frontier options (additive) ---
    ap.add_argument("--sweep", action="store_true",
                    help="Also sweep every cutoff, mark the Pareto frontier, and "
                         "write a sweep CSV + scatter/frontier PNG.")
    ap.add_argument("--accuracy-metric", choices=["worst", "mean"], default="worst",
                    help="Accuracy scalar for the sweep: worst per-model drop "
                         "(default) or mean per-model drop.")
    ap.add_argument("--plot-out", default=None,
                    help="PNG path for the Pareto plot (default: "
                         "<out_dir>/pareto_<pool>.png). Implies --sweep.")
    ap.add_argument("--sweep-csv", default=None,
                    help="CSV path for the full swept table (default: "
                         "<out_dir>/sweep_<pool>.csv). Implies --sweep.")
    ap.add_argument("--validate", action="store_true",
                    help="Check the corrected memory model against the pool's "
                         "verified reference value (VALIDATION_REFERENCE) and "
                         "exit (status 1 on mismatch).")
    ap.add_argument("--per-model", action="store_true",
                    help="Also emit per-model-cutoff points (Bpm=<=1%%, Cpm=<=2%%, "
                         "Kpm=knee): each model takes its own deepest cutoff within "
                         "budget, so a fragile member no longer caps a robust one. "
                         "Additive; global B/C are unchanged.")
    ap.add_argument("--paper-mem", type=float, default=None,
                    help="Also emit a paper-closest point P = the global cutoff "
                         "whose savings is nearest this reported memory-reduction "
                         "%% for the figure (omit for sets with no paper number).")
    args = ap.parse_args()

    global PER_MODEL, PAPER_MEM
    PER_MODEL = args.per_model
    PAPER_MEM = args.paper_mem

    load_pool_config(args.pool_config)
    if args.pool not in POOLS:
        ap.error(f"unknown pool {args.pool!r}; known: {', '.join(POOLS)}")

    if args.validate:
        res = validate_pool(args.pool)
        status = "OK" if res["ok"] else "FAIL"
        print(f"[validate {status}] pool={res['pool']} "
              f"savings={res['savings_pct']:.2f}% "
              f"(expected {res['expected_pct']}% +/- {res['tol']}); "
              f"freed={res['freed_mib'] * MIB_TO_MB:.1f} MB of {res['total_mib'] * MIB_TO_MB:.1f} MB "
              f"at cutoffs {res['cutoffs']}")
        raise SystemExit(0 if res["ok"] else 1)

    pts = {p["name"]: p for p in POOLS[args.pool]["points"]}
    if args.save_b is not None and "B" in pts:
        pts["B"].update(kind="mem_ge", target=args.save_b, range=None)
    if args.save_c is not None and "C" in pts:
        pts["C"].update(kind="mem_ge", target=args.save_c, range=None)

    out_dir, points, models, run_base = write_outputs(args.pool)
    print(f"Wrote outputs to {out_dir}")
    for pt in points:
        d = ", ".join(f"{_short(m)} {pt['scores'][m]:.2f} ({pt['drops'][m]:+.2f})"
                      for m in models)
        suffix = "" if pt["name"] == "A" else f" -> {pt['name']}.jsonl"
        cut = pt["cutoff"]
        cut_s = "pm " if isinstance(cut, dict) else f"{cut:>3}"
        print(f"  {pt['name']}: cutoff={cut_s} groups={len(pt['spec']):>3} "
              f"save={pt['savings_mb'] * MIB_TO_MB:>6.0f}MB ({pt['savings_pct']:>4.1f}%) | {d}{suffix}")

    if args.sweep or args.plot_out or args.sweep_csv:
        table, frontier, paths = run_sweep(
            args.pool, accuracy_metric=args.accuracy_metric,
            plot_out=args.plot_out, sweep_csv=args.sweep_csv,
        )
        print(f"Swept {len(table)} cutoffs (accuracy metric = {args.accuracy_metric}); "
              f"{len(frontier)} on the Pareto frontier.")
        print(f"  sweep table -> {paths['csv']}")
        print(f"  pareto plot -> {paths['png']}")


if __name__ == "__main__":
    main()
