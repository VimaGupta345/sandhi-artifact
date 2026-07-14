
import pandas as pd
import os
import sys
import glob
import numpy as np
import argparse
import re
from typing import Optional

# matplotlib is imported lazily/optionally so that the data-loading helpers
# (model-name derivation, step cutoffs, baseline lookups) can be used in
# headless or minimal environments without matplotlib installed. plot_run()
# checks for availability before plotting.
try:  # pragma: no cover - environment dependent
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

# Make the merge_tools package importable so the profiler baseline lookup
# (single source of truth) can be reused here.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from merge_tools.micr.baselines import get_baseline as _get_profiler_baseline  # type: ignore
except Exception:  # pragma: no cover - keep plotting usable if the package is unavailable
    def _get_profiler_baseline(
        label: str, task: Optional[str] = None, split: Optional[str] = None
    ) -> Optional[float]:
        return None

# ---------- Configuration ----------

# Default paths - can be overridden by arguments
DEFAULT_DATA_DIR = "csvs_for_merging"
DEFAULT_OUTPUT_DIR = "plots"
DEFAULT_BASELINE_DIR = "merge_tools/gaussian_profiler/output"
DEFAULT_NOISE_PROFILE_DIR = "merge_tools/profiles/llama_latest_noise_profiles"

# Memory profiles in MB (based on bfloat16 = 2 bytes)
# Structure: Profile -> Component -> Size(MB)
MEMORY_PROFILES = {
    "deepseek-7b": {
        # Subcomponents
        "q_proj": 32.0, "k_proj": 32.0, "v_proj": 32.0, "o_proj": 32.0,
        "gate_proj": 86.0, "up_proj": 86.0, "down_proj": 86.0,
    },
    "llama-8b": {
        # Subcomponents
        "q_proj": 32.0, "k_proj": 8.0, "v_proj": 8.0, "o_proj": 32.0,
        "gate_proj": 112.0, "up_proj": 112.0, "down_proj": 112.0,
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
        "q_proj": 24.5, "k_proj": 3.5, "v_proj": 3.5, "o_proj": 24.5,
        "gate_proj": 129.5, "up_proj": 129.5, "down_proj": 129.5,
    },
    "qwen-32b": {
        # Subcomponents
        "q_proj": 80.0, "k_proj": 10.0, "v_proj": 10.0, "o_proj": 80.0,
        "gate_proj": 250.0, "up_proj": 250.0, "down_proj": 250.0,
    }
}

# Known Qwen 32B model names (lowercased)
QWEN_32B_MODEL_NAMES = {
    "light-if-32b",
    "medgo",
    "t-pro-it-2.0",
}

# Mapping from ops CSV component names to profile component names
COMPONENT_MAPPING = {
    "mlp_down": "down_proj",
    "mlp_up": "up_proj",
    "mlp_gate": "gate_proj",
    "attn_q": "q_proj",
    "attn_k": "k_proj",
    "attn_v": "v_proj",
    "attn_o": "o_proj"
}
FULL_LAYER_COMPONENTS = (
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_o",
    "mlp_gate",
    "mlp_up",
    "mlp_down",
)

# Baseline scores are looked up from the gaussian profiler output via
# merge_tools.micr.baselines.get_baseline (single source of truth); the former
# HARDCODED_BASELINE_SCORES dict was removed. Baselines are keyed by evaluation
# split; plots report full-set numbers, so the fallback lookup asks for the
# 'full' split only. The measured steps.csv baseline row still takes precedence
# in get_baseline_score() below.

NOISE_PROFILE_FILES = {
    "fin-llama3.1-8b": "llama_finance_profile.csv",
    "Llama-3.1-Hawkish-8B": "llama_finance_profile.csv",
    "calme-2.3-legalkit-8b": "llama_legal_profile.csv",
    "Llama-3-8B-UltraMedical": "llama_medical_profile.csv",
    "Llama-3.1-8B-UltraMedical": "llama_medical_profile.csv",
    "Llama-SafetyGuard-Content-Binary": "llama_toxicity_profile.csv",
    "Llama-3.1-8B-Instruct-multi-truth-judge": "llama_truthfulness_profile.csv",
}


def _load_micr_baseline_score(steps_df: pd.DataFrame) -> Optional[float]:
    if steps_df is None or steps_df.empty or "score" not in steps_df.columns:
        return None

    baseline_rows = pd.DataFrame()
    if "op" in steps_df.columns:
        baseline_rows = steps_df[steps_df["op"].astype(str).str.lower() == "baseline"].copy()
    if baseline_rows.empty and "component" in steps_df.columns:
        baseline_rows = steps_df[steps_df["component"].astype(str).str.lower() == "baseline"].copy()
    if baseline_rows.empty and "decision" in steps_df.columns:
        baseline_rows = steps_df[steps_df["decision"].astype(str).str.lower() == "baseline"].copy()

    if baseline_rows.empty:
        return None
    try:
        return float(baseline_rows.iloc[0]["score"])
    except Exception:
        return None


def _remove_unscored_rows(steps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only scored merge steps.

    Drops the baseline row and any step whose evaluation produced no score
    (decision == "eval_failed", written with an empty score). Such rows carry no
    accept/reject verdict, so plotting them as deltas against the baseline would
    invent data.
    """
    if steps_df is None or steps_df.empty:
        return steps_df
    df = steps_df.copy()
    mask = pd.Series([False] * len(df), index=df.index)
    if "op" in df.columns:
        mask = mask | (df["op"].astype(str).str.lower() == "baseline")
    if "component" in df.columns:
        mask = mask | (df["component"].astype(str).str.lower() == "baseline")
    if "decision" in df.columns:
        decision = df["decision"].astype(str).str.lower()
        mask = mask | decision.isin(["baseline", "eval_failed"])
    if "score" in df.columns:
        mask = mask | pd.to_numeric(df["score"], errors="coerce").isna()
    return df[~mask].copy()


def _load_gaussian_profiler_baseline_score(baseline_dir: str, model_name: str) -> float | None:
    """
    Read the baseline score from gaussian_profiler output CSV.

    Expected format (first entry):
      layer = -1, perturbation = "baseline", score = <float>
    File name convention:
      gaussian_{model_name}.csv
    """
    if not baseline_dir:
        return None
    path = os.path.join(baseline_dir, f"gaussian_{model_name}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if "score" not in df.columns:
        return None

    # Prefer the explicit baseline row (layer=-1, perturbation=baseline)
    if "perturbation" in df.columns:
        base = df[df["perturbation"].astype(str).str.lower() == "baseline"].copy()
        if "layer" in base.columns:
            try:
                base = base[base["layer"].astype(int) == -1]
            except Exception:
                pass
        if not base.empty:
            try:
                return float(base.iloc[0]["score"])
            except Exception:
                pass

    # Fallback: first row score
    try:
        return float(df.iloc[0]["score"])
    except Exception:
        return None


def _load_noise_profile_baseline_score(noise_profile_dir: str, model_name: str) -> float | None:
    if not noise_profile_dir:
        return None
    filename = NOISE_PROFILE_FILES.get(model_name)
    if not filename:
        return None
    path = os.path.join(noise_profile_dir, filename)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if "score" not in df.columns:
        return None

    baseline_rows = pd.DataFrame()
    if "perturbation" in df.columns:
        baseline_rows = df[df["perturbation"].astype(str).str.lower() == "baseline"].copy()
    if "layer" in baseline_rows.columns:
        try:
            baseline_rows = baseline_rows[baseline_rows["layer"].astype(int) == -1]
        except Exception:
            pass

    if baseline_rows.empty:
        return None
    try:
        return float(baseline_rows.iloc[0]["score"])
    except Exception:
        return None


def get_baseline_score(model_name: str, baseline_dir: str = "", steps_df: Optional[pd.DataFrame] = None) -> float:
    """Baseline score used for delta plots and delta text.

    Precedence: the steps.csv's own baseline row ALWAYS wins -- it is the
    baseline the MICR loop iterated against, measured on the same eval split
    as every plotted step, so deltas are self-consistent. The lookups below
    are last resorts for CSVs without a baseline row; they may come from a
    different split (profiler P / central full), so the fallback is warned
    loudly: deltas against them mix splits and can be off by several points
    on small tasks.
    """
    score = _load_micr_baseline_score(steps_df if steps_df is not None else pd.DataFrame())
    if score is not None:
        return float(score)

    print(f"[plot_steps] WARNING: no baseline row in steps.csv for {model_name}; "
          f"falling back to a profiler/central baseline, which may be measured "
          f"on a DIFFERENT eval split than the plotted steps (deltas suspect)")
    score = _load_noise_profile_baseline_score(DEFAULT_NOISE_PROFILE_DIR, model_name)
    if score is not None:
        return float(score)

    score = _load_noise_profile_baseline_score(baseline_dir, model_name)
    if score is not None:
        return float(score)

    score = _load_gaussian_profiler_baseline_score(baseline_dir, model_name)
    if score is not None:
        return float(score)
    profiler_baseline = _get_profiler_baseline(model_name, split="full")
    return float(profiler_baseline if profiler_baseline is not None else 0.0)

MODEL_LABELS = {
    "Llama-3-8B-UltraMedical": "Medical",
    "Llama-3.1-8B-UltraMedical": "Medical",
    "calme-2.3-legalkit-8b": "Legal",
    "Llama-SafetyGuard-Content-Binary": "Safety",
    "Llama-3.1-8B-Instruct-multi-truth-judge": "Truth",
    "fin-llama3.1-8b": "Finance",
    "Llama-3.1-Hawkish-8B": "Finance",
    # LoRA-merged-to-full models
    "finance_merged": "Finance (LoRA-merged)",
    "legal_merged": "Legal (LoRA-merged)",
    "medical_merged": "Medical (LoRA-merged)",
    "toxicity_merged": "Safety (LoRA-merged)",
    "truthfulness_merged": "Truth (LoRA-merged)",
    "Qwen2.5-Math-7B-Instruct": "Math",
    "Qwen2.5-Coder-7B-Instruct": "Coder",
    "deepseek-math-7b-instruct": "DS-Math",
    "deepseek-coder-7b-instruct-v1.5": "DS-Coder",
    # Qwen 32B family
    "Light-IF-32B": "Light-IF 32B",
    "T-pro-it-2.0": "T-pro-it 2.0",
    "MedGo": "MedGo",
}


def get_display_label(model_name: str) -> str:
    """
    Human-friendly label for legends and delta text.

    Keep internal identifiers (model_name) unchanged for lookups, but remove
    the literal token "BLOCK" from displayed names (e.g. "...-FP8-BLOCK" -> "...-FP8").
    """
    raw = MODEL_LABELS.get(model_name, model_name)
    # Remove a standalone "BLOCK" token with common separators.
    cleaned = re.sub(r"(?i)([-_ ]+)BLOCK\b", "", str(raw))
    cleaned = cleaned.replace("--", "-").replace("__", "_").strip(" -_")
    return cleaned

"""
Per-model step cutoffs.

We stop plotting and stop counting memory savings after the given cutoff
for each model.

CUTOFF CONVENTION: the numbers recorded in MODEL_STEP_CUTOFFS and
RUN_SPECIFIC_STEP_CUTOFFS are 1-based CSV *line numbers* in the steps CSV
(line 1 = header, line 2 = the baseline row with step_idx == -1, line 3 =
the first merge step with step_idx == 0). apply_step_cutoff() converts them
to step indices via step_idx = line - 3 by default (cutoff_mode="line");
pass cutoff_mode="step_idx" if a cutoff dict already holds step_idx values.
"""
MODEL_STEP_CUTOFFS = {
    # Qwen 32B family
    "T-pro-it-2.0": 105,
    "MedGo": 122,
    "Light-IF-32B": 122,
    # 5 Llama models (used across 5-llama, ds+llama, and 9-model runs)
    "fin-llama3.1-8b": 60,                          # Finance
    "Llama-3.1-Hawkish-8B": 60,                     # Finance
    "calme-2.3-legalkit-8b": 52,                    # Legal
    "Llama-SafetyGuard-Content-Binary": 47,         # Safety
    "Llama-3-8B-UltraMedical": 31,                  # Medical
    "Llama-3.1-8B-UltraMedical": 31,                # Medical
    "Llama-3.1-8B-Instruct-multi-truth-judge": 47,  # Truth
    # LoRA-merged-to-full models (optional; only used when --apply-cutoffs is set)
    "finance_merged": 60,
    "legal_merged": 52,
    "medical_merged": 31,
    "toxicity_merged": 49,
    "truthfulness_merged": 47,
    # DeepSeek models
    "deepseek-coder-7b-instruct-v1.5": 16,          # DS-Coder
    "deepseek-math-7b-instruct": 16,                # DS-Math
    # Qwen 7B models (only present in 9-model run)
    "Qwen2.5-Math-7B-Instruct": 5,                  # Qwen-Math
    "Qwen2.5-Coder-7B-Instruct": 5,                 # Qwen-Code

    # Quantized (FP8-BLOCK) run cutoffs (from ops_step_csvs_llama_quant/results/*_steps.csv)
    # Legalkit: keep all entries (last step_idx = 37)
    "calme-2.3-legalkit-8b-FP8-BLOCK": 37,
    # Multi-truth: cutoff at the entry with score 76.66 (step_idx = 43)
    "Llama-3.1-8B-Instruct-multi-truth-judge-FP8-BLOCK": 43,
    # UltraMedical: cutoff at the entry with score 76.84 (step_idx = 31)
    "Llama-3.1-8B-UltraMedical-FP8-BLOCK": 31,
    # Hawkish: keep all entries (last step_idx = 3)
    "Llama-3.1-Hawkish-8B-FP8-BLOCK": 3,
    # Safety: keep all entries (last step_idx = 42)
    "Llama-SafetyGuard-Content-Binary-FP8-BLOCK": 42,
}

RUN_SPECIFIC_STEP_CUTOFFS = {
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
    "6_deepseek_standardized": {
        "deepseek-coder-7b-instruct-v1.5": 14,
        "deepseek-math-7b-instruct": 20,
    },
    "6b_deepseek_standardized": {
        "deepseek-coder-7b-instruct-v1.5": 14,
        "deepseek-math-7b-instruct": 20,
    },
    "6c_deepseek_standardized_layer_level": {
        "deepseek-coder-7b-instruct-v1.5": 1,
        "deepseek-math-7b-instruct": 1,
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
    "7b_llamas_standardized_layer_level": {
        "Llama-3.1-Hawkish-8B": 8,
        "calme-2.3-legalkit-8b": 28,
        "Llama-SafetyGuard-Content-Binary": 26,
        "Llama-3.1-8B-Instruct-multi-truth-judge": 0,
        "Llama-3.1-8B-UltraMedical": 18,
    },
    "8_9model_standardized": {
        "deepseek-coder-7b-instruct-v1.5": 3,
        "deepseek-math-7b-instruct": 11,
        "Llama-3.1-Hawkish-8B": 33,
        "calme-2.3-legalkit-8b": 25,
        "Qwen2.5-Coder-7B-Instruct": 10,
        "Qwen2.5-Math-7B-Instruct": 4,
        "Llama-SafetyGuard-Content-Binary": 68,
        "Llama-3.1-8B-Instruct-multi-truth-judge": 56,
        "Llama-3.1-8B-UltraMedical": 51,
    },
    "8a_dsnoscaler": {
        "deepseek-coder-7b-instruct-v1.5": 29,
        "deepseek-math-7b-instruct": 6,
    },
}


def get_quantization_factor(model_name: str) -> float:
    """
    Return a multiplicative factor to adjust memory for quantized models.

    MEMORY_PROFILES are in MB for bfloat16 (2 bytes per parameter). For FP8
    quantization (1 byte per parameter), memory is halved, so we use 0.5.

    We detect the FP8/BLOCK models heuristically from the model name.
    """

    lower = str(model_name).lower()
    # Treat any model whose name mentions "fp8" as using FP8 quantization.
    if "fp8" in lower:
        return 0.5
    return 1.0


MODEL_LORA_FACTOR_OVERRIDES = {
    "finance_merged": True,
    "legal_merged": True,
    "medical_merged": True,
    "toxicity_merged": True,
    "truthfulness_merged": True,
}

MODEL_PROFILE_OVERRIDES = {
    "finance_merged": "llama-8b",
    "legal_merged": "llama-8b",
    "medical_merged": "llama-8b",
    "toxicity_merged": "llama-8b",
    "truthfulness_merged": "llama-8b",
}

LORA_COMPONENT_FACTORS = {
    # Attention (per-layer total 6.5 MB vs base 80 MB)
    "q_proj": 0.08125,
    "k_proj": 0.08125,
    "v_proj": 0.08125,
    "o_proj": 0.08125,
    # MLP (per-layer total 13.5 MB vs base 336 MB)
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
    if not _is_lora_accounting_model(model_name):
        return 1.0
    return float(LORA_COMPONENT_FACTORS.get(profile_component_name, 1.0))


def get_profile_name(model_name):
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

def load_ops_data(data_dir, model_name, ops_dir=""):
    ops_data = {}
    
    search_dirs = [data_dir]
    if ops_dir:
        search_dirs.insert(0, ops_dir)
    else:
        # Fallback: Check parent directory
        search_dirs.append(os.path.dirname(data_dir))

    for stage in [1, 2]:
        # Try recursive search for ops files
        # Pattern: **/ops_step{stage}_{model_name}.csv
        pattern = f"ops_step{stage}_{model_name}.csv"
        
        files = []
        for d in search_dirs:
            search_pattern = os.path.join(d, "**", pattern)
            found = glob.glob(search_pattern, recursive=True)
            if found:
                files = found
                break
        
        if files:
            # Pick the first one found
            # print(f"  Found ops file for {model_name} stage {stage}: {files[0]}")
            ops_data[stage] = pd.read_csv(files[0])
        else:
            print(f"  WARNING: No ops file found for {model_name} stage {stage} in search paths: {search_dirs}")
            
    return ops_data

def get_model_family(model_name):
    """Extracts a simple family identifier from the model name."""
    lower = str(model_name).lower()
    if lower in QWEN_32B_MODEL_NAMES:
        return "Qwen"
    if "deepseek" in lower:
        return "Deepseek"
    if "qwen" in lower:
        return "Qwen"
    if "llama" in lower or "calme" in lower or "fin-" in lower: # Heuristic for Llama derivatives
        return "Llama"
    return "Llama" # Default


def apply_step_cutoff(df, model_name, cutoffs, cutoff_mode="line"):
    """
    Apply a hard-coded step cutoff to a model's *_steps.csv dataframe.

    CUTOFF CONVENTION (cutoff_mode="line", the default): the configured
    cutoffs in MODEL_STEP_CUTOFFS / RUN_SPECIFIC_STEP_CUTOFFS are recorded as
    1-based CSV line numbers of the steps CSV, where line 1 is the header and
    line 2 (the first data row) is the baseline row with step_idx == -1.
    Therefore step_idx = line - 3, and a cutoff at line L keeps rows with
    step_idx <= L - 3 (the baseline row, step_idx == -1, is always kept).

    cutoff_mode="step_idx" is an escape hatch for cutoff dicts that already
    hold step_idx values directly.

    We use the step_idx column if present; otherwise we fall back to the
    dataframe index (assumed to match the plotted step indices).
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
        filtered = df[df["step_idx"] <= cutoff_step_idx].copy()
    else:
        # Fallback: assume 0-based contiguous index matching step indices
        filtered = df.iloc[: cutoff_step_idx + 1].copy()

    return filtered


def get_step_cutoffs_for_run(run_name: str):
    return RUN_SPECIFIC_STEP_CUTOFFS.get(run_name, MODEL_STEP_CUTOFFS)


def derive_model_name(steps_path: str, df: Optional[pd.DataFrame] = None) -> str:
    """
    Derive the model name for a steps CSV.

    Prefer the explicit 'label' column carried by MICR steps CSVs (first
    non-null, non-empty value) — this is the true model label and matches the
    keys used in MODEL_LABELS and the cutoff dicts. Fall back to filename
    conventions, and for files literally named steps.csv, to the directory
    basename (which may be a short run alias such as 'hawkish').
    """
    if df is not None and "label" in df.columns and not df.empty:
        labels = df["label"].dropna().astype(str).str.strip()
        labels = labels[(labels != "") & (labels.str.lower() != "nan")]
        if not labels.empty:
            return str(labels.iloc[0])

    basename = os.path.basename(steps_path)
    if "micr_results_" in basename:
        return basename.replace("micr_results_", "").replace(".csv", "")
    if "_steps_" in basename:
        return basename.split("_steps_")[0]
    if "_steps.csv" in basename:
        return basename.replace("_steps.csv", "")
    if basename == "steps.csv":
        return os.path.basename(os.path.dirname(steps_path))
    return basename.replace(".csv", "")

DEBUG_MEMORY_LOG = True


def plot_run(
    run_name,
    model_dfs,
    ops_dfs,
    output_dir,
    family_only=False,
    gb_mode=False,
    show_family_avg=False,
    family_avg_only=False,
    baseline_dir="",
):
    if not model_dfs:
        print("No model dataframes found to plot.")
        return

    if plt is None:
        raise ImportError(
            "matplotlib is required for plot_run() but is not installed."
        )

    print(f"Plotting run: {run_name} with {len(model_dfs)} models")
    
    fig, ax1 = plt.subplots(figsize=(20, 12))
    
    max_steps = 0
    # --- Plot Accuracy on Primary Y-Axis (ax1) ---
    if family_avg_only:
        # Aggregate accuracy per model family: average score at each local step index.
        fam_step_scores = {}  # family -> {local_idx -> [scores]}
        fam_baselines = {}    # family -> list of per-model baselines

        for model, df in model_dfs.items():
            fam = get_model_family(model)
            baseline = get_baseline_score(model, baseline_dir=baseline_dir, steps_df=df)
            fam_baselines.setdefault(fam, []).append(baseline)

            df_local = _remove_unscored_rows(df).reset_index(drop=True)
            for idx, row in df_local.iterrows():
                fam_step_scores.setdefault(fam, {}).setdefault(idx, []).append(row["score"])

        fam_names = sorted(fam_step_scores.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(fam_names)))
        fam_color_map = {fam: color for fam, color in zip(fam_names, colors)}

        for fam in fam_names:
            step_to_scores = fam_step_scores[fam]
            indices = sorted(step_to_scores.keys())
            if not indices:
                continue

            # Average baseline across models in this family
            baselines = fam_baselines.get(fam, [0.0])
            baseline_fam = float(sum(baselines)) / max(len(baselines), 1)

            scores = [
                float(sum(step_to_scores[i])) / len(step_to_scores[i])
                for i in indices
            ]

            # Build delta plot starting from baseline
            df_plot = pd.concat(
                [
                    pd.DataFrame(
                        [{"score": baseline_fam, "stage": 1, "decision": "start"}]
                    ),
                    pd.DataFrame(
                        {
                            "score": scores,
                            "stage": 1,
                            "decision": ["avg"] * len(scores),
                        }
                    ),
                ],
                ignore_index=True,
            )
            df_plot["delta"] = df_plot["score"] - baseline_fam

            color = fam_color_map[fam]
            max_steps = max(max_steps, len(scores))

            for i in range(1, len(df_plot)):
                linestyle = "-"
                # Plot horizontal segment from previous step
                ax1.plot(
                    [i - 2, i - 1],
                    [df_plot.iloc[i - 1]["delta"], df_plot.iloc[i - 1]["delta"]],
                    color=color,
                    linestyle=linestyle,
                    linewidth=2,
                )
                # Plot vertical segment for score change
                ax1.plot(
                    [i - 1, i - 1],
                    [df_plot.iloc[i - 1]["delta"], df_plot.iloc[i]["delta"]],
                    color=color,
                    linestyle=linestyle,
                    linewidth=2,
                )

            # Legend entry: one line per family
            ax1.plot([], [], label=f"{fam} avg", color=color, linewidth=2)

            # Scatter markers for each averaged step
            for i, delta_val in enumerate(df_plot["delta"][1:]):  # skip baseline
                ax1.scatter(i, delta_val, marker="+", color=color, s=80, zorder=5)

            # Baseline at y=0 for delta plot
            ax1.axhline(0, linestyle=":", alpha=0.6, color=color)
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_dfs)))
        model_color_map = {model: color for model, color in zip(model_dfs.keys(), colors)}

        for model, df in model_dfs.items():
            label = get_display_label(model)
            baseline = get_baseline_score(model, baseline_dir=baseline_dir, steps_df=df)
            df_no_baseline = _remove_unscored_rows(df).reset_index(drop=True)
            color = model_color_map[model]
            max_steps = max(max_steps, len(df_no_baseline))

            # Create a starting point for the plot at the baseline
            # Delta mode: Start at 0 (score - baseline = 0)
            df_plot = pd.concat(
                [
                    pd.DataFrame(
                        [{"score": baseline, "stage": 1, "decision": "start"}]
                    ),
                    df_no_baseline,
                ],
                ignore_index=True,
            )

            # Calculate delta for all points
            df_plot["delta"] = df_plot["score"] - baseline

            # Plot segments based on stage
            for i in range(1, len(df_plot)):
                linestyle = "--" if df_plot.iloc[i].get("stage", 1) == 2 else "-"
                # Plot horizontal segment from previous step
                ax1.plot(
                    [i - 2, i - 1],
                    [df_plot.iloc[i - 1]["delta"], df_plot.iloc[i - 1]["delta"]],
                    color=color,
                    linestyle=linestyle,
                    linewidth=2,
                )
                # Plot vertical segment for score change
                ax1.plot(
                    [i - 1, i - 1],
                    [df_plot.iloc[i - 1]["delta"], df_plot.iloc[i]["delta"]],
                    color=color,
                    linestyle=linestyle,
                    linewidth=2,
                )

            # Create a single line for the legend
            ax1.plot([], [], label=label, color=color, linewidth=2)

            for i, row in df_no_baseline.iterrows():
                marker = "x" if row["decision"] == "rejected" else "+"
                delta_val = row["score"] - baseline
                ax1.scatter(i, delta_val, marker=marker, color=color, s=80, zorder=5)

            # Baseline at y=0 for delta plot
            ax1.axhline(0, linestyle=":", alpha=0.6, color=color)

    ax1.set_xlabel("Steps", fontsize=14)
    ax1.set_ylabel("Accuracy Delta (%)", fontsize=14)
    ax1.set_title(f"Accuracy Delta & Memory Savings - {run_name}", fontsize=18)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.set_xlim(left=0, right=max_steps)
    ax1.set_ylim(bottom=-10, top=10)

    # --- Plot Memory Savings on Secondary Y-Axis (ax2) ---
    ax2 = ax1.twinx()

        # Build a single events table: all models, all stages
    all_events_list = []
    for model, df in model_dfs.items():
        temp = _remove_unscored_rows(df).copy()
        temp["model_name"] = model
        if "step_idx" not in temp.columns:
            temp = temp.sort_index().reset_index(drop=True)
            temp["step_idx"] = np.arange(len(temp))
        all_events_list.append(temp)

    all_events = pd.concat(all_events_list, ignore_index=True)

    # --- Precompute candidate groups from ops_dfs ---
    # Each candidate is defined by (component, full models list with their local layers)
    valid_models = set(model_dfs.keys())
    candidates = {}      # cand_id -> {'p_comp': str, 'members': [(model, layer)], 'size': float}
    candidate_active = {}  # cand_id -> set of active member_keys "model:layer"

    for anchor_model, stage_dict in ops_dfs.items():
        for stage, ops_df in stage_dict.items():
            if ops_df is None:
                continue
            for _, op_row in ops_df.iterrows():
                sub_comp = op_row["component"]
                p_comp = COMPONENT_MAPPING.get(sub_comp)
                if not p_comp:
                    continue

                models_str = op_row.get("models", "")
                raw_tokens = [t.strip() for t in str(models_str).split(",") if t.strip()]
                members = []
                for tok in raw_tokens:
                    parts = tok.split(":")
                    if len(parts) == 2:
                        m_str, layer_str = parts
                    else:
                        m_str = parts[0]
                        layer_str = op_row["layer"]
                    m_str = m_str.strip()
                    if m_str not in valid_models:
                        continue
                    try:
                        layer_int = int(layer_str)
                    except ValueError:
                        continue
                    members.append((m_str, layer_int))

                # Only groups of 2+ models can give savings
                if len(members) < 2:
                    continue

                member_keys = tuple(sorted(f"{m}:{layer}" for (m, layer) in members))
                cand_id = (p_comp, member_keys)

                if cand_id not in candidates:
                    # Component size: take the max size across members (safe upper bound),
                    # adjusted for quantization (e.g., FP8 halves memory).
                    size = 0.0
                    for m_name, _ in members:
                        p_name = get_profile_name(m_name)
                        profile = MEMORY_PROFILES.get(p_name, MEMORY_PROFILES["llama-8b"])
                        base_size = profile.get(p_comp, 0.0)
                        size = max(
                            size,
                            base_size
                            * get_quantization_factor(m_name)
                            * get_lora_factor(m_name, p_comp),
                        )
                    candidates[cand_id] = {"p_comp": p_comp, "members": members, "size": size}
                    candidate_active[cand_id] = set()
    # State: which (layer, sub_component) are merged per model
    model_merge_status = {m: {} for m in model_dfs.keys()}
       # --- Candidate-based K tracking and incremental savings ---
    memory_savings_history = []
    cumulative_saved = 0.0

    # Process steps in order of step_idx; one x-axis tick = one iteration through all CSVs
    for step_idx, step_events in all_events.sort_values("step_idx").groupby("step_idx", sort=True):
        # Snapshot previous active counts for each candidate
        prev_counts = {cid: len(active) for cid, active in candidate_active.items()}

        if DEBUG_MEMORY_LOG:
            print(f"\n=== Global Step {step_idx} ===")

        # 1) Apply all accepted merges at this step across all models
        for _, ev in step_events.iterrows():
            if ev.get("decision") != "accepted":
                continue

            model_name  = ev["model_name"]
            stage       = ev.get("stage", 1)
            layer_event = ev["layer"]
            comp_type   = ev["component"]  # 'mlp', 'attn', or 'layer_full'

            if DEBUG_MEMORY_LOG:
                print(f"  [ACCEPT] model={model_name}, stage={stage}, layer={layer_event}, comp={comp_type}")

            ops_df = ops_dfs.get(model_name, {}).get(stage)
            if ops_df is None:
                if DEBUG_MEMORY_LOG:
                    print("    -> no ops_df for this stage")
                continue

            # Find all subcomponents of this generic component at this event's layer
            if comp_type == "layer_full":
                relevant_ops = ops_df[
                    (ops_df["layer"] == layer_event)
                    & (ops_df["component"].astype(str).isin(FULL_LAYER_COMPONENTS))
                ]
            elif comp_type in {"attn", "mlp"}:
                relevant_ops = ops_df[
                    (ops_df["layer"] == layer_event)
                    & (ops_df["component"].astype(str).str.startswith(comp_type))
                ]
            else:
                relevant_ops = ops_df[
                    (ops_df["layer"] == layer_event)
                    & (ops_df["component"].astype(str) == str(comp_type))
                ]

            if relevant_ops.empty and DEBUG_MEMORY_LOG:
                print("    -> no matching ops rows")

            for _, op_row in relevant_ops.iterrows():
                sub_comp = op_row["component"]
                p_comp   = COMPONENT_MAPPING.get(sub_comp)
                if not p_comp:
                    continue

                models_str = op_row.get("models", "")
                raw_tokens = [t.strip() for t in str(models_str).split(",") if t.strip()]
                members = []
                for tok in raw_tokens:
                    parts = tok.split(":")
                    if len(parts) == 2:
                        m_str, layer_str = parts
                    else:
                        m_str = parts[0]
                        layer_str = op_row["layer"]
                    m_str = m_str.strip()
                    if m_str not in valid_models:
                        continue
                    try:
                        layer_int = int(layer_str)
                    except ValueError:
                        continue
                    members.append((m_str, layer_int))

                if len(members) < 2:
                    continue

                member_keys = tuple(sorted(f"{m}:{layer}" for (m, layer) in members))
                cand_id     = (p_comp, member_keys)

                # Ensure candidate exists (in case not seen during pre-scan)
                if cand_id not in candidates:
                    size = 0.0
                    for m_name, _ in members:
                        p_name = get_profile_name(m_name)
                        profile = MEMORY_PROFILES.get(p_name, MEMORY_PROFILES["llama-8b"])
                        base_size = profile.get(p_comp, 0.0)
                        size = max(
                            size,
                            base_size
                            * get_quantization_factor(m_name)
                            * get_lora_factor(m_name, p_comp),
                        )
                    candidates[cand_id] = {"p_comp": p_comp, "members": members, "size": size}
                    candidate_active[cand_id] = set()

                # Mark only THIS model's participation in this candidate as active
                for (m_name, m_layer) in members:
                    if m_name == model_name:
                        mk = f"{m_name}:{m_layer}"
                        candidate_active[cand_id].add(mk)
                        if DEBUG_MEMORY_LOG:
                            print(f"    -> candidate {cand_id}: activate member {mk}")

        # 2) Compute incremental savings at this step
        step_saved = 0.0
        for cand_id, active_set in candidate_active.items():
            size = candidates[cand_id]["size"]
            if size <= 0:
                continue

            k_prev = prev_counts.get(cand_id, 0)
            k_curr = len(active_set)
            if k_curr <= k_prev:
                continue  # no new participants

            # First time: if k_prev == 0, initial savings = (k_curr - 1)*size
            # Later: only new participants save, each saves size
            if k_prev == 0:
                k_new = max(0, k_curr - 1)
            else:
                k_new = k_curr - k_prev

            if k_new <= 0:
                continue

            saved = k_new * size
            step_saved += saved

            if DEBUG_MEMORY_LOG:
                print(
                    f"  Candidate {cand_id}: k_prev={k_prev}, k_curr={k_curr}, "
                    f"k_new={k_new}, size={size:.2f} MB -> step_saved={saved:.2f} MB"
                )

        cumulative_saved += step_saved
        memory_savings_history.append(cumulative_saved)

        if DEBUG_MEMORY_LOG:
            print(
                f"  Total cumulative memory saved after step {step_idx}: "
                f"{cumulative_saved:.2f} MB"
            )

    # Convert to MB/GB and plot
    mb_history = np.array(memory_savings_history)
    if gb_mode:
        mb_history = mb_history / 1024.0
        unit_label = "GB"
    else:
        unit_label = "MB"

    # Compute max memory savings (in current unit) for logging and annotation
    if len(mb_history) > 0:
        max_mem_saved = float(np.max(mb_history))
    else:
        max_mem_saved = 0.0

    if max_mem_saved > 0:
        print(f"Max memory savings: {max_mem_saved:.2f} {unit_label}")
    else:
        print("WARNING: No memory savings calculated (Max is 0).")

    mem_x_axis = np.arange(len(mb_history))
    ax2.plot(mem_x_axis, mb_history, color="black", linewidth=2.5,
             label=f"Memory Saved ({unit_label})", drawstyle="steps-post")
    ax2.set_ylabel(f"Cumulative Memory Saved ({unit_label})", fontsize=14)
    ax2.set_ylim(bottom=0)

    # --- Final Touches ---
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', ncol=2, fontsize='medium')
    
    # percent_accepted = (accepted_count / total_decisions) * 100 if total_decisions > 0 else 0
    # stats_text = f"Accepted Merges: {percent_accepted:.1f}%"

    delta_texts = []
    family_final_scores = {}  # family -> list of final scores

    if family_avg_only:
        # For text summary, show only per-family averages.
        fam_scores = {}      # family -> list of final scores
        fam_baselines = {}   # family -> list of baselines

        for model in sorted(model_dfs.keys()):
            df = model_dfs[model]
            baseline = get_baseline_score(model, baseline_dir=baseline_dir, steps_df=df)
            accepted_df = _remove_unscored_rows(df)
            accepted_df = accepted_df[accepted_df["decision"] == "accepted"]
            final_score = (
                accepted_df["score"].iloc[-1] if not accepted_df.empty else baseline
            )

            fam = get_model_family(model)
            fam_scores.setdefault(fam, []).append(final_score)
            fam_baselines.setdefault(fam, []).append(baseline)

        for fam in sorted(fam_scores.keys()):
            scores = fam_scores[fam]
            baselines = fam_baselines.get(fam, [0.0])
            avg_score = sum(scores) / len(scores)
            avg_baseline = sum(baselines) / max(len(baselines), 1)
            delta = avg_score - avg_baseline
            delta_texts.append(f"{fam} avg Δ: {delta:+.2f}")

            if show_family_avg:
                family_final_scores.setdefault(fam, []).append(avg_score)
    else:
        for model in sorted(model_dfs.keys()):
            df = model_dfs[model]
            baseline = get_baseline_score(model, baseline_dir=baseline_dir, steps_df=df)
            accepted_df = _remove_unscored_rows(df)
            accepted_df = accepted_df[accepted_df["decision"] == "accepted"]
            final_score = (
                accepted_df["score"].iloc[-1] if not accepted_df.empty else baseline
            )
            delta = final_score - baseline
            label = get_display_label(model)
            delta_texts.append(f"{label} Δ: {delta:+.2f}")

            # Track final scores per family for optional averaging
            if show_family_avg:
                fam = get_model_family(model)
                family_final_scores.setdefault(fam, []).append(final_score)
    
    deltas_text = " | ".join(delta_texts)

    # Text annotation: max memory savings on the figure
    mem_stats_text = f"Max memory saved: {max_mem_saved:.2f} {unit_label}"
    fig.text(0.5, 0.045, mem_stats_text, ha='center', fontsize=12)

    # Optional: per-family average final accuracy
    if show_family_avg and family_final_scores:
        fam_texts = []
        for fam in sorted(family_final_scores.keys()):
            scores = family_final_scores[fam]
            avg_score = sum(scores) / len(scores)
            fam_texts.append(f"{fam} avg: {avg_score:.2f}")
        fam_stats_text = " | ".join(fam_texts)
        fig.text(0.5, 0.025, fam_stats_text, ha='center', fontsize=12)

    # fig.text(0.5, 0.04, stats_text, ha='center', fontsize=14, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    fig.text(0.5, 0.005, deltas_text, ha='center', fontsize=12)
    
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    
    suffix = "_family_only" if family_only else ""
    output_path = os.path.join(output_dir, f"plot_combined_unsync_{run_name}{suffix}.png")
    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Plot merging steps accuracy and memory.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory containing CSV files")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save plots")
    parser.add_argument(
        "--baseline-dir",
        default=DEFAULT_BASELINE_DIR,
        help=(
            "Directory containing gaussian_profiler baseline CSVs "
            "(files named gaussian_{model}.csv). Baselines are resolved in this order: "
            "MICR results CSV baseline row, current noise profiles, gaussian baseline CSVs, "
            "then hardcoded fallback."
        ),
    )
    parser.add_argument("--family-only", action="store_true", help="Filter merged models to only include same family")
    parser.add_argument(
        "--max-stage",
        type=int,
        default=2,
        help="Maximum stage to include from *_steps.csv (e.g., 1 to only use stage 1).",
    )
    parser.add_argument(
        "--show-family-avg",
        action="store_true",
        help="Show average final accuracy per model family in the figure text.",
    )
    parser.add_argument(
        "--family-avg-only",
        action="store_true",
        help="Plot only per-family average accuracy curves (no per-model curves).",
    )
    parser.add_argument(
        "--allow-families",
        type=str,
        default="",
        help=(
            "Comma-separated list of model families to include "
            "(e.g., 'Qwen,Deepseek,Llama'). If empty, all families are used."
        ),
    )
    parser.add_argument("--gb", action="store_true", help="Plot memory savings in GB instead of MB")
    cutoff_group = parser.add_mutually_exclusive_group()
    cutoff_group.add_argument(
        "--apply-cutoffs",
        dest="apply_cutoffs",
        action="store_true",
        help=(
            "Apply hard-coded per-model step cutoffs so that plotting and memory "
            "calculations stop at those steps."
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

    # Parse allowed families filter (if any)
    if args.allow_families:
        allowed_families = {fam.strip() for fam in args.allow_families.split(",") if fam.strip()}
        print(f"Restricting to model families: {sorted(allowed_families)}")
    else:
        allowed_families = None

    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.data_dir):
        print(f"Directory not found: {args.data_dir}")
        return

    # If the data directory directly contains per-model result folders with
    # `steps.csv`, treat it as a single combined run. Otherwise, fall back to
    # the older convention where subdirectories represent separate runs.
    direct_model_steps = glob.glob(os.path.join(args.data_dir, "*", "steps.csv"))

    # Check for subdirectories (different runs) vs flat directory
    subdirs = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    
    # If direct model folders exist, or there are no subdirs, assume data-dir is
    # already the run directory.
    if direct_model_steps or not subdirs:
        run_dirs = [args.data_dir]
    else:
        run_dirs = [os.path.join(args.data_dir, d) for d in subdirs]
        
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        run_cutoffs = get_step_cutoffs_for_run(run_name)
        # Skip if run_name is just the data_dir name and we want to process subdirs... 
        # But here we just process what we found.
        
        # Check for step CSVs recursively, preferring the newer per-model steps.csv layout.
        csv_files = glob.glob(os.path.join(run_dir, "**", "steps.csv"), recursive=True)
        if not csv_files:
            csv_files = glob.glob(os.path.join(run_dir, "**", "*_steps.csv"), recursive=True)
        if not csv_files:
            csv_files = glob.glob(os.path.join(run_dir, "**", "*_steps_*.csv"), recursive=True)
        if not csv_files:
            # Fallback for MICR results
            csv_files = glob.glob(os.path.join(run_dir, "**", "micr_results_*.csv"), recursive=True)
        
        if not csv_files:
            # If we are iterating subdirs, this is fine. If flat, we might print warning.
            if len(run_dirs) == 1:
                print(f"No step CSVs found in {run_dir}")
            continue
            
        print(f"Processing run: {run_name}")
        
        model_dfs = {}
        ops_dfs = {}
        
        for f in csv_files:
            df = pd.read_csv(f)

            # Prefer the explicit label from the MICR results CSV (first
            # non-null value), falling back to filename/directory conventions.
            model_name = derive_model_name(f, df)

            # Optionally filter by allowed model families
            family = get_model_family(model_name)
            if allowed_families is not None and family not in allowed_families:
                continue

            # Optionally restrict to a maximum stage (e.g., stage 1 only)
            if "stage" in df.columns and args.max_stage is not None:
                df_before_stage = len(df)
                df = df[df["stage"] <= args.max_stage].copy()
                if df_before_stage != len(df):
                    print(
                        f"  Applied stage filter for {model_name}: "
                        f"kept {len(df)} of {df_before_stage} rows (stage <= {args.max_stage})"
                    )

            # Optionally apply per-model step cutoffs
            if args.apply_cutoffs:
                df_before = len(df)
                df = apply_step_cutoff(df, model_name, run_cutoffs)
                if len(df) < df_before:
                    print(
                        f"  Applied cutoff for {model_name}: "
                        f"kept {len(df)} of {df_before} steps"
                    )

            model_dfs[model_name] = df
            ops_dfs[model_name] = load_ops_data(run_dir, model_name, ops_dir=args.ops_dir)

        if not model_dfs:
            print(f"  No models matched allowed_families in {run_dir}, skipping run.")
            continue

        plot_run(
            run_name,
            model_dfs,
            ops_dfs,
            args.output_dir,
            args.family_only,
            args.gb,
            args.show_family_avg,
             args.family_avg_only,
            args.baseline_dir,
        )

if __name__ == "__main__":
    main()
