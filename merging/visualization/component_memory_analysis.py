import argparse
import os
import glob
from typing import Dict, List, Set, Tuple

import pandas as pd

# ---------- Configuration (copied from plot_steps.py where relevant) ----------

# Memory profiles in MB (based on bfloat16 = 2 bytes)
# Structure: Profile -> Component -> Size(MB)
MEMORY_PROFILES: Dict[str, Dict[str, float]] = {
    "deepseek-7b": {
        # Subcomponents
        "q_proj": 32.0,
        "k_proj": 32.0,
        "v_proj": 32.0,
        "o_proj": 32.0,
        "gate_proj": 86.0,
        "up_proj": 86.0,
        "down_proj": 86.0,
    },
    "llama-8b": {
        # Subcomponents
        "q_proj": 32.0,
        "k_proj": 8.0,
        "v_proj": 8.0,
        "o_proj": 32.0,
        "gate_proj": 112.0,
        "up_proj": 112.0,
        "down_proj": 112.0,
    },
    "llama-8b-lora": {
        # Subcomponents (LoRA adapter sizes)
        # Attn per layer (q/k/v/o_proj) = 6.5 MB total
        # q/o_proj are typically larger than k/v_proj in Llama if rank is consistent,
        # but here we distribute the 6.5 MB proportionally to the base model weights
        # Base: q=32, k=8, v=8, o=32 -> Total=80.
        # q_proj = (32/80) * 6.5 = 2.6
        # k_proj = (8/80) * 6.5 = 0.65
        # v_proj = (8/80) * 6.5 = 0.65
        # o_proj = (32/80) * 6.5 = 2.6
        "q_proj": 2.6,
        "k_proj": 0.65,
        "v_proj": 0.65,
        "o_proj": 2.6,
        # MLP per layer (gate/up/down_proj) = 13.5 MB total
        # Base: gate=112, up=112, down=112 -> Total=336
        # Each is (112/336) * 13.5 = 4.5
        "gate_proj": 4.5,
        "up_proj": 4.5,
        "down_proj": 4.5,
    },
    "qwen-7b": {
        # Subcomponents
        "q_proj": 24.5,
        "k_proj": 3.5,
        "v_proj": 3.5,
        "o_proj": 24.5,
        "gate_proj": 129.5,
        "up_proj": 129.5,
        "down_proj": 129.5,
    },
    "qwen-32b": {
        # Subcomponents
        "q_proj": 80.0,
        "k_proj": 10.0,
        "v_proj": 10.0,
        "o_proj": 80.0,
        "gate_proj": 250.0,
        "up_proj": 250.0,
        "down_proj": 250.0,
    },
}

# Known Qwen 32B model names (lowercased)
QWEN_32B_MODEL_NAMES: Set[str] = {
    "light-if-32b",
    "medgo",
    "t-pro-it-2.0",
}

# Mapping from ops CSV component names to profile component names
COMPONENT_MAPPING: Dict[str, str] = {
    "mlp_down": "down_proj",
    "mlp_up": "up_proj",
    "mlp_gate": "gate_proj",
    "attn_q": "q_proj",
    "attn_k": "k_proj",
    "attn_v": "v_proj",
    "attn_o": "o_proj",
}
FULL_LAYER_COMPONENTS: Tuple[str, ...] = (
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_o",
    "mlp_gate",
    "mlp_up",
    "mlp_down",
)

"""
Per-model step cutoffs (shared with plot_steps.py).

We stop counting accepted layers/components for memory after the given
cutoff for each model.

CUTOFF CONVENTION: the numbers recorded in MODEL_STEP_CUTOFFS and
RUN_SPECIFIC_STEP_CUTOFFS are 1-based CSV *line numbers* in the steps CSV
(line 1 = header, line 2 = the baseline row with step_idx == -1, line 3 =
the first merge step with step_idx == 0). apply_step_cutoff() converts them
to step indices via step_idx = line - 3 by default (cutoff_mode="line");
pass cutoff_mode="step_idx" if a cutoff dict already holds step_idx values.
"""
MODEL_STEP_CUTOFFS: Dict[str, int] = {
    # Qwen 32B family
    "T-pro-it-2.0": 118,
    "MedGo": 122,
    "Light-IF-32B": 122,
    # 5 Llama models
    "fin-llama3.1-8b": 60,                          # Finance
    "Llama-3.1-Hawkish-8B": 60,                    # Finance
    "calme-2.3-legalkit-8b": 52,                    # Legal
    "Llama-SafetyGuard-Content-Binary": 49,         # Safety
    "Llama-3-8B-UltraMedical": 31,                  # Medical
    "Llama-3.1-8B-UltraMedical": 31,                # Medical
    "Llama-3.1-8B-Instruct-multi-truth-judge": 47,  # Truth
    "finance_merged": 60,                          # Finance
    "legal_merged": 52,                    # Legal
    "toxicity_merged": 49,         # Safety
    "medical_merged": 31,                  # Medical
    "truthfulness_merged": 47,  # Truth
    # DeepSeek models
    "deepseek-coder-7b-instruct-v1.5": 16,          # DS-Coder
    "deepseek-math-7b-instruct": 16,                # DS-Math
    # Qwen 7B models
    "Qwen2.5-Math-7B-Instruct": 5,                  # Qwen-Math
    "Qwen2.5-Coder-7B-Instruct": 5,                 # Qwen-Code
}

RUN_SPECIFIC_STEP_CUTOFFS: Dict[str, Dict[str, int]] = {
    "variant2_standardized": {
        "fin-llama3.1-8b": 22,
        "Llama-3.1-Hawkish-8B": 22,
        "calme-2.3-legalkit-8b": 34,
        "Llama-SafetyGuard-Content-Binary": 47,
        "Llama-3-8B-UltraMedical": 50,
        "Llama-3.1-8B-UltraMedical": 50,
        "Llama-3.1-8B-Instruct-multi-truth-judge": 43,
    },
    "variant4c_standardized": {
        "fin-llama3.1-8b": 3,
        "Llama-3.1-Hawkish-8B": 3,
        "calme-2.3-legalkit-8b": 14,
        "Llama-SafetyGuard-Content-Binary": 12,
        "Llama-3-8B-UltraMedical": 14,
        "Llama-3.1-8B-UltraMedical": 14,
        "Llama-3.1-8B-Instruct-multi-truth-judge": 13,
    },
    "7_llamas_standardized": {
        # 1-based CSV line numbers (see cutoff convention above):
        # step_idx cutoffs are line - 3 -> {35, 59, 66, 56, 51}.
        "Llama-3.1-Hawkish-8B": 38,
        "calme-2.3-legalkit-8b": 62,
        "Llama-SafetyGuard-Content-Binary": 69,
        "Llama-3.1-8B-Instruct-multi-truth-judge": 59,
        "Llama-3.1-8B-UltraMedical": 54,
    },
}

# Per-model overrides (mirrors the FP8 heuristic idea, but explicit).
# Use this when model labels don't contain enough hints, or when you want to
# guarantee the intended memory accounting.
MODEL_PROFILE_OVERRIDES: Dict[str, str] = {
    # LoRA-merged-to-full models: still Llama weights, but savings should be
    # computed as if we were deleting LoRA adapter weights component-by-component.
    "finance_merged": "llama-8b",
    "legal_merged": "llama-8b",
    "medical_merged": "llama-8b",
    "toxicity_merged": "llama-8b",
    "truthfulness_merged": "llama-8b",
}

# Optional per-model quantization overrides (multiplier applied on top of profile).
# Keep empty unless you have models whose names don't contain "fp8" but are FP8.
MODEL_QUANTIZATION_FACTOR_OVERRIDES: Dict[str, float] = {}

"""
LoRA memory scaling (per-component).

We want to simulate memory savings as if removing LoRA adapter params, even when
the adapter is merged back into a "full" model. The user-provided LoRA adapter
sizes for Llama 8B are (per-layer, MB):
- Attn q/k/v/o total: 6.5 MB  (base per-layer total: 32+8+8+32 = 80 MB)
- MLP gate/up/down total: 13.5 MB (base per-layer total: 112+112+112 = 336 MB)

So the per-component scaling factors relative to the base bfloat16 profile are:
- Attn projections: 6.5 / 80 = 0.08125
- MLP projections: 13.5 / 336 = 0.04017857142857143
"""

MODEL_LORA_FACTOR_OVERRIDES: Dict[str, bool] = {
    "finance_merged": True,
    "legal_merged": True,
    "medical_merged": True,
    "toxicity_merged": True,
    "truthfulness_merged": True,
}

LORA_COMPONENT_FACTORS: Dict[str, float] = {
    # Attention
    "q_proj": 0.08125,
    "k_proj": 0.08125,
    "v_proj": 0.08125,
    "o_proj": 0.08125,
    # MLP
    "gate_proj": 13.5 / 336.0,
    "up_proj": 13.5 / 336.0,
    "down_proj": 13.5 / 336.0,
}


def _is_lora_accounting_model(model_name: str) -> bool:
    if model_name in MODEL_LORA_FACTOR_OVERRIDES:
        return bool(MODEL_LORA_FACTOR_OVERRIDES[model_name])
    lower = str(model_name).lower()
    return ("lora" in lower) or ("adapter" in lower) or ("merged" in lower)


def get_lora_factor(model_name: str, profile_component_name: str) -> float:
    """
    Return a multiplicative factor to adjust component memory for LoRA accounting.

    This is applied *in addition* to the profile component size and the
    quantization factor.
    """
    if not _is_lora_accounting_model(model_name):
        return 1.0
    return float(LORA_COMPONENT_FACTORS.get(profile_component_name, 1.0))


def get_quantization_factor(model_name: str) -> float:
    """
    Return a multiplicative factor to adjust memory for quantized models.

    MEMORY_PROFILES are in MB for bfloat16 (2 bytes per parameter). For FP8
    quantization (1 byte per parameter), memory is halved, so we use 0.5.

    We detect the FP8/BLOCK models heuristically from the model name.
    """

    if model_name in MODEL_QUANTIZATION_FACTOR_OVERRIDES:
        return float(MODEL_QUANTIZATION_FACTOR_OVERRIDES[model_name])

    lower = str(model_name).lower()
    # Treat any model whose name mentions "fp8" as using FP8 quantization.
    if "fp8" in lower:
        return 0.5
    return 1.0


def get_profile_name(model_name: str) -> str:
    """Map a concrete model name to a memory profile key."""

    if model_name in MODEL_PROFILE_OVERRIDES:
        return MODEL_PROFILE_OVERRIDES[model_name]

    lower = str(model_name).lower()
    # Explicit list of known Qwen 32B models
    if lower in QWEN_32B_MODEL_NAMES:
        return "qwen-32b"
    if "deepseek" in lower:
        return "deepseek-7b"
    if "32b" in lower:
        return "qwen-32b"
    if "qwen" in lower:
        return "qwen-7b"
    # Default to llama-8b for others
    return "llama-8b"


def load_ops_data(data_dir: str, model_name: str, ops_dir: str = "") -> Dict[int, pd.DataFrame]:
    """Load ops_step1/2 CSVs for a given model.

    Looks for files named ops_step{stage}_{model_name}.csv recursively under data_dir
    (or ops_dir if provided).
    Returns a dict: stage -> DataFrame.
    """

    ops_data: Dict[int, pd.DataFrame] = {}
    search_dirs = [data_dir]
    if ops_dir:
        search_dirs.insert(0, ops_dir)
    else:
        # Fallback: check parent directory as in plot_steps.py
        search_dirs.append(os.path.dirname(data_dir))

    for stage in (1, 2):
        files = []
        pattern = f"ops_step{stage}_{model_name}.csv"
        
        for d in search_dirs:
            search_pattern = os.path.join(d, "**", pattern)
            found = glob.glob(search_pattern, recursive=True)
            if found:
                files = found
                break

        if files:
            ops_data[stage] = pd.read_csv(files[0])
        else:
            # It is OK for a stage to be missing; we just skip it.
            print(
                f"  WARNING: No ops_step{stage} CSV found for model {model_name} "
                f"in search paths: {search_dirs}"
            )

    return ops_data


def parse_models_field(models_str: str, default_layer: int) -> List[Tuple[str, int]]:
    """Parse the 'models' column from ops CSVs.

    Each entry is of the form "Model:layer" or just "Model". If layer is
    omitted we fall back to default_layer.
    """

    tokens = [t.strip() for t in str(models_str).split(",") if t.strip()]
    members: List[Tuple[str, int]] = []
    for tok in tokens:
        parts = tok.split(":")
        if len(parts) == 2:
            m_str, layer_str = parts
        else:
            m_str = parts[0]
            layer_str = str(default_layer)
        m_str = m_str.strip()
        try:
            layer_int = int(layer_str)
        except ValueError:
            continue
        members.append((m_str, layer_int))
    return members


def compute_component_memory_for_model(
    model_name: str,
    steps_df: pd.DataFrame,
    ops_data: Dict[int, pd.DataFrame],
    unit: str = "MB",
) -> Dict:
    """For a single model, compute which (layer, logical component) pairs are
    accepted (via step-1 and step-2 ops) and how much parameter memory they
    correspond to.

    Returns a dict with:
      - accepted_pairs_by_stage: {stage -> set((layer, logical_component))}
      - per_step_records: list of per-step dicts (only when new pairs appear)
      - memory_by_stage: {stage -> memory in requested unit}
      - memory_union: total memory across all unique pairs in requested unit
      - unit: "MB" or "GB"
    """

    profile_name = get_profile_name(model_name)
    profile = MEMORY_PROFILES.get(profile_name, MEMORY_PROFILES["llama-8b"])
    quant_factor = get_quantization_factor(model_name)

    # Normalise unit
    unit = unit.upper()
    if unit not in {"MB", "GB"}:
        raise ValueError("unit must be 'MB' or 'GB'")

    accepted_pairs_by_stage: Dict[int, Set[Tuple[int, str]]] = {1: set(), 2: set()}
    seen_pairs: Set[Tuple[int, int, str]] = set()  # (stage, layer, logical_component)
    per_step_records: List[Dict] = []

    if "decision" in steps_df.columns:
        accepted_df = steps_df[steps_df["decision"] == "accepted"].copy()
    else:
        accepted_df = steps_df.copy()

    if "stage" not in accepted_df.columns:
        accepted_df["stage"] = 1

    if "step_idx" not in accepted_df.columns:
        accepted_df = accepted_df.reset_index().rename(columns={"index": "step_idx"})

    accepted_df = accepted_df.sort_values("step_idx")

    cumulative_mem_mb = 0.0

    # Ensure ops layer columns are integer type to match steps_df
    for stage, df_ops in list(ops_data.items()):
        if "layer" in df_ops.columns and not pd.api.types.is_integer_dtype(df_ops["layer"]):
            df_ops = df_ops.copy()
            df_ops["layer"] = df_ops["layer"].astype(int)
            ops_data[stage] = df_ops

    for _, ev in accepted_df.iterrows():
        try:
            stage = int(ev.get("stage", 1))
        except ValueError:
            stage = 1
        try:
            layer = int(ev["layer"])
        except ValueError:
            continue
        comp_type = str(ev["component"])  # "mlp", "attn", or "layer_full"

        ops_df = ops_data.get(stage)
        if ops_df is None:
            continue

        ops_components = ops_df["component"].astype(str)
        if comp_type == "layer_full":
            relevant_ops = ops_df[
                (ops_df["layer"] == layer)
                & (ops_components.isin(FULL_LAYER_COMPONENTS))
            ]
        elif comp_type in {"attn", "mlp"}:
            relevant_ops = ops_df[
                (ops_df["layer"] == layer)
                & (ops_components.str.startswith(comp_type))
            ]
        else:
            relevant_ops = ops_df[
                (ops_df["layer"] == layer)
                & (ops_components == comp_type)
            ]

        step_new_pairs: List[Tuple[int, str, float]] = []  # (layer, logical_component, mem_mb)

        for _, op_row in relevant_ops.iterrows():
            sub_comp = str(op_row["component"])  # e.g. mlp_down, attn_q, ...
            p_comp = COMPONENT_MAPPING.get(sub_comp)
            if not p_comp:
                continue

            members = parse_models_field(op_row.get("models", ""), int(op_row["layer"]))
            belongs = any((m == model_name and l == layer) for (m, l) in members)
            if not belongs:
                continue

            key = (stage, layer, p_comp)
            if key in seen_pairs:
                continue

            seen_pairs.add(key)
            accepted_pairs_by_stage.setdefault(stage, set()).add((layer, p_comp))
            mem_mb = (
                float(profile.get(p_comp, 0.0))
                * quant_factor
                * get_lora_factor(model_name, p_comp)
            )
            cumulative_mem_mb += mem_mb
            step_new_pairs.append((layer, p_comp, mem_mb))

        if step_new_pairs:
            step_idx = int(ev["step_idx"])
            new_mem_mb = sum(m for (_, _, m) in step_new_pairs)
            per_step_records.append(
                {
                    "model": model_name,
                    "step_idx": step_idx,
                    "stage": stage,
                    "generic_component": comp_type,
                    "layer": layer,
                    "num_new_pairs": len(step_new_pairs),
                    "new_pairs": ";".join(f"{ly}:{pc}" for (ly, pc, _) in step_new_pairs),
                    "new_memory_mb": new_mem_mb,
                    "cum_memory_mb": cumulative_mem_mb,
                }
            )

    def sum_pairs_mb(pairs: Set[Tuple[int, str]]) -> float:
        return sum(
            float(profile.get(p_comp, 0.0))
            * quant_factor
            * get_lora_factor(model_name, p_comp)
            for (_, p_comp) in pairs
        )

    mem_stage1_mb = sum_pairs_mb(accepted_pairs_by_stage.get(1, set()))
    mem_stage2_mb = sum_pairs_mb(accepted_pairs_by_stage.get(2, set()))
    union_pairs = accepted_pairs_by_stage.get(1, set()) | accepted_pairs_by_stage.get(2, set())
    mem_union_mb = sum_pairs_mb(union_pairs)

    factor = 1.0 if unit == "MB" else 1.0 / 1024.0

    memory_by_stage = {
        1: mem_stage1_mb * factor,
        2: mem_stage2_mb * factor,
    }
    memory_union = mem_union_mb * factor

    return {
        "accepted_pairs_by_stage": accepted_pairs_by_stage,
        "per_step_records": per_step_records,
        "memory_by_stage": memory_by_stage,
        "memory_union": memory_union,
        "unit": unit,
    }


def apply_step_cutoff(
    df: pd.DataFrame,
    model_name: str,
    cutoffs: Dict[str, int],
    cutoff_mode: str = "line",
) -> pd.DataFrame:
    """
    Apply hard-coded step cutoffs to a *_steps.csv dataframe.

    CUTOFF CONVENTION (cutoff_mode="line", the default): the configured
    cutoffs in MODEL_STEP_CUTOFFS / RUN_SPECIFIC_STEP_CUTOFFS are recorded as
    1-based CSV line numbers of the steps CSV, where line 1 is the header and
    line 2 (the first data row) is the baseline row with step_idx == -1.
    Therefore step_idx = line - 3, and a cutoff at line L keeps rows with
    step_idx <= L - 3 (the baseline row, step_idx == -1, is always kept).

    cutoff_mode="step_idx" is an escape hatch for cutoff dicts that already
    hold step_idx values directly.

    Uses the step_idx column if present; otherwise falls back to index.
    """
    cutoff = cutoffs.get(model_name)
    if cutoff is None:
        return df

    if cutoff_mode == "line":
        cutoff_step_idx = int(cutoff) - 3
    elif cutoff_mode == "step_idx":
        cutoff_step_idx = int(cutoff)
    else:
        raise ValueError(f"Unknown cutoff_mode: {cutoff_mode!r} (use 'line' or 'step_idx')")

    if "step_idx" in df.columns:
        return df[df["step_idx"] <= cutoff_step_idx].copy()
    else:
        return df.iloc[: cutoff_step_idx + 1].copy()


def get_step_cutoffs_for_run(run_name: str) -> Dict[str, int]:
    return RUN_SPECIFIC_STEP_CUTOFFS.get(run_name, MODEL_STEP_CUTOFFS)


def derive_model_name(steps_path: str, df: pd.DataFrame = None) -> str:
    """
    Derive the model name for a steps CSV (shared logic with plot_steps.py).

    Prefer the explicit 'label' column carried by MICR steps CSVs (first
    non-null, non-empty value) — this is the true model label and matches the
    keys used in the cutoff dicts. Fall back to filename conventions, and for
    files literally named steps.csv, to the directory basename (which may be
    a short run alias such as 'hawkish').
    """
    if df is not None and "label" in df.columns and not df.empty:
        labels = df["label"].dropna().astype(str).str.strip()
        labels = labels[(labels != "") & (labels.str.lower() != "nan")]
        if not labels.empty:
            return str(labels.iloc[0])

    base = os.path.basename(steps_path)
    if "micr_results_" in base:
        return base.replace("micr_results_", "").replace(".csv", "")
    if "_steps_" in base:
        return base.split("_steps_")[0]
    if base.endswith("_steps.csv"):
        return base.replace("_steps.csv", "")
    if base == "steps.csv":
        return os.path.basename(os.path.dirname(steps_path))
    return base.replace(".csv", "")


def load_meta_csv(path: str) -> Dict[str, Dict[str, float]]:
    """Optional helper: load per-model metadata with total memory savings and
    final step ids.

    Expected columns: model,total_memory_savings,final_step_idx
    If a column is missing its values are treated as 0.
    """

    meta: Dict[str, Dict[str, float]] = {}
    if not path:
        return meta

    df = pd.read_csv(path)
    for _, row in df.iterrows():
        model = str(row.get("model"))
        if not model:
            continue
        total = float(row.get("total_memory_savings", 0.0))
        step = int(row.get("final_step_idx", 0))
        meta[model] = {
            "total_memory_savings": total,
            "final_step_idx": step,
        }
    return meta


def analyse_runs(
    data_dir: str,
    unit: str = "MB",
    meta_csv: str = "",
    output_summary: str = "",
    output_steps: str = "",
    apply_cutoffs: bool = False,
    ops_dir: str = "",
) -> None:
    """Run analysis over all runs under data_dir (or a single run directory).

    For each model we
      - look at its *_steps.csv decisions
      - consult ops_step1/ops_step2 CSVs
      - collect accepted (layer, component) pairs
      - compute the total parameter memory they correspond to

    Results are printed and optionally written as CSVs.
    """

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"data-dir not found: {data_dir}")

    # Determine run directories (same convention as plot_steps.py)
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if not subdirs:
        run_dirs = [data_dir]
    else:
        run_dirs = [os.path.join(data_dir, d) for d in subdirs]

    meta = load_meta_csv(meta_csv)

    summary_rows: List[Dict] = []
    steps_rows: List[Dict] = []

    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        run_cutoffs = get_step_cutoffs_for_run(run_name)
        # Look for per-model step CSVs in the newer `steps.csv` layout first,
        # then fall back to older naming conventions.
        steps_files = glob.glob(os.path.join(run_dir, "**", "steps.csv"), recursive=True)
        if not steps_files:
            # Look for *results*.csv or *_steps.csv
            steps_files = glob.glob(os.path.join(run_dir, "**", "*_steps.csv"), recursive=True)
        if not steps_files:
            # Fallback: try micr_results_*.csv
            steps_files = glob.glob(os.path.join(run_dir, "**", "micr_results_*.csv"), recursive=True)
        
        if not steps_files:
            print(f"No steps.csv, *_steps.csv, or micr_results_*.csv files found in run directory {run_dir}, skipping.")
            continue

        print(f"Processing run: {run_name}")

        for steps_path in steps_files:
            steps_df = pd.read_csv(steps_path)
            # Prefer the explicit label column (first non-null value), falling
            # back to filename/directory conventions.
            model_name = derive_model_name(steps_path, steps_df)

            # Optionally truncate to the desired cutoff step for this model
            if apply_cutoffs:
                before = len(steps_df)
                steps_df = apply_step_cutoff(steps_df, model_name, run_cutoffs)
                if len(steps_df) < before:
                    print(
                        f"  Applied cutoff for {model_name}: "
                        f"kept {len(steps_df)} of {before} steps"
                    )

            ops_data = load_ops_data(run_dir, model_name, ops_dir=ops_dir)
            if not ops_data:
                print(f"  WARNING: No ops_step CSVs for model {model_name} in run {run_name}, skipping.")
                continue

            result = compute_component_memory_for_model(model_name, steps_df, ops_data, unit=unit)

            accepted_stage1 = result["accepted_pairs_by_stage"].get(1, set())
            accepted_stage2 = result["accepted_pairs_by_stage"].get(2, set())
            union_pairs = accepted_stage1 | accepted_stage2

            def fmt_pairs(pairs: Set[Tuple[int, str]]) -> str:
                if not pairs:
                    return ""
                return ";".join(f"{layer}:{comp}" for (layer, comp) in sorted(pairs))

            mem_info = meta.get(
                model_name,
                {"total_memory_savings": 0.0, "final_step_idx": 0},
            )

            summary_row = {
                "run": run_name,
                "model": model_name,
                "num_pairs_stage1": len(accepted_stage1),
                "num_pairs_stage2": len(accepted_stage2),
                "num_pairs_union": len(union_pairs),
                "memory_stage1_" + unit: result["memory_by_stage"].get(1, 0.0),
                "memory_stage2_" + unit: result["memory_by_stage"].get(2, 0.0),
                "memory_union_" + unit: result["memory_union"],
                "pairs_stage1": fmt_pairs(accepted_stage1),
                "pairs_stage2": fmt_pairs(accepted_stage2),
                "pairs_union": fmt_pairs(union_pairs),
                # User-provided totals (if any), defaults to 0 as requested
                "given_total_memory_savings": mem_info["total_memory_savings"],
                "given_final_step_idx": mem_info["final_step_idx"],
            }
            summary_rows.append(summary_row)

            for rec in result["per_step_records"]:
                rec_with_run = {"run": run_name, **rec}
                steps_rows.append(rec_with_run)

            print(
                f"  Model {model_name}: stage1={summary_row['memory_stage1_' + unit]:.2f} {unit}, "
                f"stage2={summary_row['memory_stage2_' + unit]:.2f} {unit}, "
                f"union={summary_row['memory_union_' + unit]:.2f} {unit}, "
                f"pairs (s1={summary_row['num_pairs_stage1']}, "
                f"s2={summary_row['num_pairs_stage2']}, union={summary_row['num_pairs_union']})"
            )

    if output_summary:
        pd.DataFrame(summary_rows).to_csv(output_summary, index=False)
        print(f"Wrote summary CSV to {output_summary}")

    if output_steps and steps_rows:
        pd.DataFrame(steps_rows).to_csv(output_steps, index=False)
        print(f"Wrote per-step CSV to {output_steps}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse accepted layers/components from step-1 and step-2 ops CSVs "
            "and compute total parameter memory per model."
        )
    )
    parser.add_argument(
        "--data-dir",
        default="csvs_for_merging",
        help=(
            "Directory containing run subdirectories (e.g. ops_step_csvs_qwen32b) "
            "or a single run directory."
        ),
    )
    parser.add_argument(
        "--unit",
        choices=["MB", "GB"],
        default="MB",
        help="Unit to report memory in (MB or GB).",
    )
    parser.add_argument(
        "--meta-csv",
        default="",
        help=(
            "Optional CSV with columns model,total_memory_savings,final_step_idx. "
            "If missing, these are assumed to be 0 per model."
        ),
    )
    parser.add_argument(
        "--output-summary",
        default="",
        help="Optional path to write a model-level summary CSV.",
    )
    parser.add_argument(
        "--output-steps",
        default="",
        help=(
            "Optional path to write a per-step breakdown CSV (only rows where "
            "new (layer,component) pairs are activated)."
        ),
    )
    cutoff_group = parser.add_mutually_exclusive_group()
    cutoff_group.add_argument(
        "--apply-cutoffs",
        dest="apply_cutoffs",
        action="store_true",
        help=(
            "Apply hard-coded per-model step cutoffs so that accepted layers/"
            "components after those steps are ignored for memory."
        ),
    )
    cutoff_group.add_argument(
        "--no-cutoffs",
        dest="apply_cutoffs",
        action="store_false",
        help="Explicitly disable per-model step cutoffs (default behavior).",
    )
    parser.set_defaults(apply_cutoffs=False)

    parser.add_argument(
        "--ops-dir",
        default="",
        help="Optional directory to search for ops CSVs if not in data-dir.",
    )

    args = parser.parse_args()

    analyse_runs(
        data_dir=args.data_dir,
        unit=args.unit,
        meta_csv=args.meta_csv,
        output_summary=args.output_summary,
        output_steps=args.output_steps,
        apply_cutoffs=args.apply_cutoffs,
        ops_dir=args.ops_dir,
    )


if __name__ == "__main__":
    main()
