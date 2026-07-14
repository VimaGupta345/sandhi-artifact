import math
from typing import Dict, List, Tuple, Optional, Set

from pathlib import Path

import numpy as np
import pandas as pd
import torch
try:
    from matplotlib import pyplot as plt  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    plt = None  # type: ignore

try:
    import seaborn as sns  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    sns = None  # type: ignore
from scipy.cluster.hierarchy import linkage, fcluster
from transformers import AutoModelForCausalLM


# For nicer plots when using this module directly
if sns is not None:  # pragma: no cover
    sns.set(style="whitegrid")


# -----------------------------------------------------------------------------
# Low-level model / stats helpers
# -----------------------------------------------------------------------------


def _get_layer_module(model: torch.nn.Module, layer_idx: int):
    """Return the transformer block at a given index.

    Supports LLaMA-style (``model.model.layers``) and GPT-style (``transformer.h``).
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    raise RuntimeError(
        "Unsupported model structure: expected `model.model.layers` or `transformer.h`."
    )


def _get_num_layers(model: torch.nn.Module) -> int:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    raise RuntimeError(
        "Unsupported model structure: expected `model.model.layers` or `transformer.h`."
    )


def _load_model(path: str, device_hint: str = "cpu") -> torch.nn.Module:
    """Load a causal LM on CPU or CUDA (if available and requested)."""
    device = "cuda" if torch.cuda.is_available() and device_hint == "cuda" else "cpu"
    print(f"Loading model from {path} on device={device} (hint={device_hint})")
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    if device == "cuda":
        model.to(device)
    model.eval()
    return model


def _iter_attn_components(layer_module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Return a mapping of attention subcomponent name -> weight tensor.

    Names follow: ``attn_q``, ``attn_k``, ``attn_v``, ``attn_o``.
    Missing components are skipped.
    """
    out: Dict[str, torch.Tensor] = {}
    attn_module = getattr(layer_module, "self_attn", None)
    if attn_module is None:
        return out
    mapping = {
        "attn_q": "q_proj",
        "attn_k": "k_proj",
        "attn_v": "v_proj",
        "attn_o": "o_proj",
    }
    for public_name, attr_name in mapping.items():
        mod = getattr(attn_module, attr_name, None)
        if hasattr(mod, "weight"):
            out[public_name] = mod.weight
    return out


def _iter_mlp_components(layer_module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Return a mapping of MLP subcomponent name -> weight tensor.

    Names follow: ``mlp_gate``, ``mlp_up``, ``mlp_down``.
    Missing components are skipped.
    """
    out: Dict[str, torch.Tensor] = {}
    mlp_module = getattr(layer_module, "mlp", None)
    if mlp_module is None:
        return out
    mapping = {
        "mlp_gate": "gate_proj",
        "mlp_up": "up_proj",
        "mlp_down": "down_proj",
    }
    for public_name, attr_name in mapping.items():
        mod = getattr(mlp_module, attr_name, None)
        if hasattr(mod, "weight"):
            out[public_name] = mod.weight
    return out


def _pop_stats_from_tensor(weight: torch.Tensor) -> Dict[str, float]:
    """Compute population mean/std of all entries in a tensor (no grad)."""
    with torch.no_grad():
        t = weight.detach().double().view(-1)
        if t.numel() == 0:
            return {"mean": float("nan"), "std": float("nan")}
        sum_vals = t.sum().item()
        sum_sq_vals = (t * t).sum().item()
        n = t.numel()
        mean_val = sum_vals / n
        var_val = max((sum_sq_vals / n) - (mean_val * mean_val), 0.0)
        std_val = math.sqrt(var_val)
    return {"mean": float(mean_val), "std": float(std_val)}


def _agg_group_stats(weights: List[torch.Tensor]) -> Optional[Dict[str, float]]:
    """Aggregate multiple tensors into a single population mean/std.

    Mirrors the logic used in ``mean_std_profiler``.
    """
    if not weights:
        return None
    with torch.no_grad():
        total_elems = 0
        sum_vals = 0.0
        sum_sq_vals = 0.0
        for w in weights:
            t = w.detach().double().view(-1)
            total_elems += t.numel()
            sum_vals += t.sum().item()
            sum_sq_vals += (t * t).sum().item()
        if total_elems == 0:
            return None
        mean_val = sum_vals / total_elems
        var_val = max((sum_sq_vals / total_elems) - (mean_val * mean_val), 0.0)
        std_val = math.sqrt(var_val)
    return {"mean": float(mean_val), "std": float(std_val)}


# -----------------------------------------------------------------------------
# Stats computation over models
# -----------------------------------------------------------------------------


def compute_model_stats(
    model_path: str,
    model_label: Optional[str] = None,
    device: str = "cpu",
    start_layer: Optional[int] = None,
    end_layer: Optional[int] = None,
) -> pd.DataFrame:
    """Compute per-layer mean/std for groups and subcomponents for a single model.

    Returns a tidy DataFrame with columns:
      - model
      - layer
      - kind          ("group" or "subcomponent")
      - component     (e.g., "attn", "mlp", "attn_q", "mlp_up", ...)
      - mean
      - std
      - shape         (tensor shape as string, e.g., "(4096, 4096)")
    """
    model_label = model_label or Path(model_path).name
    model = _load_model(model_path, device_hint=device)
    total_layers = _get_num_layers(model)

    s = 0 if start_layer is None else max(0, start_layer)
    e = (total_layers - 1) if end_layer is None else min(end_layer, total_layers - 1)
    print(f"Computing stats for {model_label}: layers {s}..{e} (total={total_layers})")

    rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for layer_idx in range(s, e + 1):
            layer_module = _get_layer_module(model, layer_idx)

            # Subcomponents
            attn_sub = _iter_attn_components(layer_module)
            mlp_sub = _iter_mlp_components(layer_module)

            # Group-level (attn / mlp)
            group_map = {
                "attn": list(attn_sub.values()),
                "mlp": list(mlp_sub.values()),
            }
            for group_name, tensors in group_map.items():
                g_stats = _agg_group_stats(tensors)
                if g_stats is None:
                    continue
                group_shape = str(tensors[0].shape) if tensors else "()"
                rows.append(
                    {
                        "model": model_label,
                        "layer": layer_idx,
                        "kind": "group",
                        "component": group_name,
                        "mean": g_stats["mean"],
                        "std": g_stats["std"],
                        "shape": group_shape,
                    }
                )

            # Individual subcomponents
            for comp_name, tensor in {**attn_sub, **mlp_sub}.items():
                s_stats = _pop_stats_from_tensor(tensor)
                rows.append(
                    {
                        "model": model_label,
                        "layer": layer_idx,
                        "kind": "subcomponent",
                        "component": comp_name,
                        "mean": s_stats["mean"],
                        "std": s_stats["std"],
                        "shape": str(tensor.shape),
                    }
                )

    # Cleanup to free memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    return df


def compute_stats_for_models(
    model_paths: List[str],
    model_labels: Optional[List[str]] = None,
    device: str = "cpu",
    start_layer: Optional[int] = None,
    end_layer: Optional[int] = None,
) -> pd.DataFrame:
    """Compute stats for a list of models and concatenate into one DataFrame."""
    if model_labels is None:
        model_labels = [Path(p).name for p in model_paths]
    if len(model_labels) != len(model_paths):
        raise ValueError("model_labels (if provided) must match length of model_paths")

    all_dfs = []
    for path, label in zip(model_paths, model_labels):
        df = compute_model_stats(
            model_path=path,
            model_label=label,
            device=device,
            start_layer=start_layer,
            end_layer=end_layer,
        )
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame(
            columns=["model", "layer", "kind", "component", "mean", "std", "shape"]
        )

    full_df = pd.concat(all_dfs, ignore_index=True)
    return full_df


# -----------------------------------------------------------------------------
# Augmenting stats with external profiles
# -----------------------------------------------------------------------------

# Component ordering within each layer
ATTN_SUBCOMPS = ["attn", "attn_q", "attn_k", "attn_v", "attn_o"]
MLP_SUBCOMPS = ["mlp", "mlp_gate", "mlp_up", "mlp_down"]
COMPONENT_ORDER = ATTN_SUBCOMPS + MLP_SUBCOMPS
COMP_ORDER_MAP = {comp: i for i, comp in enumerate(COMPONENT_ORDER)}


def _augment_one_model(
    stats_sub: pd.DataFrame,
    config: dict,
    baseline_drop_threshold: float = 2.0,
) -> pd.DataFrame:
    """Augment a single model's slice of stats_df with profile-based columns.

    Adds per-layer values for the corresponding group (attn/mlp):
    - cosine: from magnitude / stdmean profile
    - avgable: True if ``avg`` score is within ``baseline_drop_threshold`` of baseline
    - replaceable: True if ``replace`` score is within ``baseline_drop_threshold`` of baseline

    Values are duplicated across subcomponents within each family (attn vs mlp).
    """
    df_model = stats_sub.copy()

    # ---- stdmean_profile / magnitude_profile: cosine per group (attn / mlp) ----
    std_path = Path(config["std_path"])
    if not std_path.exists():
        print(f"[augment_profiles] Missing stdmean/magnitude file: {std_path}")
    else:
        std_df = pd.read_csv(std_path)
        if "group" not in std_df.columns:
            print(f"[augment_profiles] 'group' column missing in {std_path}")
        else:
            attn_cos = std_df[std_df["group"] == "attn"][
                ["layer", "cosine"]
            ].rename(columns={"cosine": "cosine_attn"})
            mlp_cos = std_df[std_df["group"] == "mlp"]["layer"].to_frame()
            mlp_cos["cosine_mlp"] = std_df[std_df["group"] == "mlp"]["cosine"].values

            df_model = df_model.merge(attn_cos, how="left", on="layer")
            df_model = df_model.merge(mlp_cos, how="left", on="layer")

    # ---- noise_profile: avgable / replaceable per group (attn / mlp) ----
    noise_path = Path(config["noise_path"])
    if not noise_path.exists():
        print(f"[augment_profiles] Missing noise file: {noise_path}")
        return df_model

    noise_df = pd.read_csv(noise_path)

    baseline_rows = noise_df[noise_df["perturbation"] == "baseline"]
    if baseline_rows.empty:
        print(f"[augment_profiles] No baseline row in {noise_path}")
        return df_model
    baseline = float(baseline_rows["score"].iloc[0])
    threshold_score = baseline - float(baseline_drop_threshold)

    def _build_flags(sub_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        sub = sub_df.copy()
        avg = sub[sub["perturbation"] == "avg"]["layer"].to_frame()
        avg[f"{prefix}_score_avg"] = sub[sub["perturbation"] == "avg"]["score"].values

        rep = sub[sub["perturbation"] == "replace"]["layer"].to_frame()
        rep[f"{prefix}_score_replace"] = sub[sub["perturbation"] == "replace"]["score"].values

        tmp = avg.merge(rep, on="layer", how="outer")
        tmp[f"{prefix}_avgable"] = tmp[f"{prefix}_score_avg"] >= threshold_score
        tmp[f"{prefix}_replaceable"] = tmp[f"{prefix}_score_replace"] >= threshold_score
        return tmp[
            [
                "layer",
                f"{prefix}_score_avg",
                f"{prefix}_score_replace",
                f"{prefix}_avgable",
                f"{prefix}_replaceable",
            ]
        ]

    noise_attn = noise_df[noise_df["variant"] == "attn"]
    noise_mlp = noise_df[noise_df["variant"] == "mlp"]

    flags_attn = (
        _build_flags(noise_attn, "attn")
        if not noise_attn.empty
        else pd.DataFrame(columns=["layer", "attn_avgable", "attn_replaceable"])
    )
    flags_mlp = (
        _build_flags(noise_mlp, "mlp")
        if not noise_mlp.empty
        else pd.DataFrame(columns=["layer", "mlp_avgable", "mlp_replaceable"])
    )

    df_model = df_model.merge(flags_attn, how="left", on="layer")
    df_model = df_model.merge(flags_mlp, how="left", on="layer")

    # ---- collapse to generic columns: cosine, avgable, replaceable, score_avg, score_replace ----
    df_model["cosine"] = pd.NA
    df_model["avgable"] = pd.NA
    df_model["replaceable"] = pd.NA
    df_model["score_avg"] = pd.NA
    df_model["score_replace"] = pd.NA

    mask_attn = df_model["component"].isin(ATTN_SUBCOMPS)
    mask_mlp = df_model["component"].isin(MLP_SUBCOMPS)

    # Cosine
    if "cosine_attn" in df_model.columns:
        df_model.loc[mask_attn, "cosine"] = df_model.loc[mask_attn, "cosine_attn"]
    if "cosine_mlp" in df_model.columns:
        df_model.loc[mask_mlp, "cosine"] = df_model.loc[mask_mlp, "cosine_mlp"]

    # Booleans
    df_model.loc[mask_attn, "avgable"] = df_model.loc[mask_attn, "attn_avgable"].astype(
        "boolean"
    )
    df_model.loc[mask_mlp, "avgable"] = df_model.loc[mask_mlp, "mlp_avgable"].astype(
        "boolean"
    )

    df_model.loc[mask_attn, "replaceable"] = df_model.loc[mask_attn, "attn_replaceable"].astype(
        "boolean"
    )
    df_model.loc[mask_mlp, "replaceable"] = df_model.loc[mask_mlp, "mlp_replaceable"].astype(
        "boolean"
    )

    # Raw scores
    df_model.loc[mask_attn, "score_avg"] = df_model.loc[mask_attn, "attn_score_avg"]
    df_model.loc[mask_mlp, "score_avg"] = df_model.loc[mask_mlp, "mlp_score_avg"]

    df_model.loc[mask_attn, "score_replace"] = df_model.loc[mask_attn, "attn_score_replace"]
    df_model.loc[mask_mlp, "score_replace"] = df_model.loc[mask_mlp, "mlp_score_replace"]

    df_model = df_model.drop(
        columns=[
            "cosine_attn",
            "cosine_mlp",
            "attn_score_avg",
            "attn_score_replace",
            "mlp_score_avg",
            "mlp_score_replace",
            "attn_avgable",
            "attn_replaceable",
            "mlp_avgable",
            "mlp_replaceable",
        ],
        errors="ignore",
    )

    return df_model


def augment_stats_with_profiles(
    stats_df: pd.DataFrame,
    model_configs: Dict[str, dict],
    baseline_drop_threshold: float = 2.0,
) -> pd.DataFrame:
    """Append cosine/avgable/replaceable columns using external profiles.

    ``model_configs`` should map model label -> dict with keys ``task``,
    ``std_path`` and ``noise_path``.
    """
    parts: List[pd.DataFrame] = []
    for model_label, config in model_configs.items():
        sub = stats_df[stats_df["model"] == model_label]
        if sub.empty:
            continue
        augmented = _augment_one_model(
            sub,
            config,
            baseline_drop_threshold=baseline_drop_threshold,
        )
        parts.append(augmented)

    rest = stats_df[~stats_df["model"].isin(model_configs.keys())]

    if parts:
        return pd.concat([rest] + parts, ignore_index=True)
    else:
        return stats_df.copy()


def get_model_family(name: str) -> str:
    lower = name.lower()
    if "qwen" in lower:
        return "Qwen"
    if "deepseek" in lower:
        return "Deepseek"
    return "Llama"


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------


def plot_mean_std_components(
    stats_df: pd.DataFrame,
    title_prefix: str = "",
) -> None:
    """Create 9 scatter plots (3x3) for mean vs std, one per component."""
    if plt is None or sns is None:
        raise RuntimeError(
            "Plotting requires `matplotlib` and `seaborn`. Install them to use plot_* helpers."
        )
    df = stats_df.copy()
    df = df[df["component"].isin(COMPONENT_ORDER)]
    if df.empty:
        print("No matching components to plot.")
        return

    df["component"] = pd.Categorical(
        df["component"], categories=COMPONENT_ORDER, ordered=True
    )

    fig, axes = plt.subplots(3, 3, figsize=(16, 14), sharex=False, sharey=False)
    axes = axes.flatten()

    for idx, comp in enumerate(COMPONENT_ORDER):
        ax = axes[idx]
        sub = df[df["component"] == comp]
        if sub.empty:
            ax.set_title(f"{comp} (no data)")
            ax.set_xlabel("mean")
            ax.set_ylabel("std")
            continue
        sns.scatterplot(
            data=sub,
            x="mean",
            y="std",
            hue="model",
            ax=ax,
            alpha=0.8,
            s=40,
        )
        ax.set_title(comp)
        ax.set_xlabel("mean")
        ax.set_ylabel("std")
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Model", loc="upper right")

    if title_prefix:
        fig.suptitle(f"{title_prefix} - Mean vs Std by Component", fontsize=16)
    else:
        fig.suptitle("Mean vs Std by Component", fontsize=16)

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.show()


def plot_mean_std_components_with_layer_ids(
    stats_df: pd.DataFrame,
    title_prefix: str = "",
) -> None:
    """Create 9 scatter plots (3x3) for mean vs std, labeling points by layer id."""
    if plt is None or sns is None:
        raise RuntimeError(
            "Plotting requires `matplotlib` and `seaborn`. Install them to use plot_* helpers."
        )
    df = stats_df.copy()
    df = df[df["component"].isin(COMPONENT_ORDER)]
    if df.empty:
        print("No matching components to plot.")
        return

    df["component"] = pd.Categorical(
        df["component"], categories=COMPONENT_ORDER, ordered=True
    )

    model_order = sorted(df["model"].unique())
    palette = sns.color_palette(n_colors=len(model_order))
    model_to_color = {m: c for m, c in zip(model_order, palette)}

    fig, axes = plt.subplots(3, 3, figsize=(16, 14), sharex=False, sharey=False)
    axes = axes.flatten()

    for idx, comp in enumerate(COMPONENT_ORDER):
        ax = axes[idx]
        sub = df[df["component"] == comp]
        if sub.empty:
            ax.set_title(f"{comp} (no data)")
            ax.set_xlabel("mean")
            ax.set_ylabel("std")
            continue

        ax.scatter(sub["mean"], sub["std"], alpha=0.0)

        for model in model_order:
            model_sub = sub[sub["model"] == model]
            if model_sub.empty:
                continue
            color = model_to_color[model]
            for _, row in model_sub.iterrows():
                ax.text(
                    row["mean"],
                    row["std"],
                    str(int(row["layer"])),
                    color=color,
                    fontsize=7,
                    ha="center",
                    va="center",
                )

        if idx == 0:
            for model in model_order:
                ax.scatter([], [], color=model_to_color[model], label=model)

        ax.set_title(comp)
        ax.set_xlabel("mean")
        ax.set_ylabel("std")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Model", loc="upper right")

    if title_prefix:
        fig.suptitle(f"{title_prefix} - Mean vs Std by Component", fontsize=16)
    else:
        fig.suptitle("Mean vs Std by Component", fontsize=16)

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.show()


# -----------------------------------------------------------------------------
# Candidate utilities
# -----------------------------------------------------------------------------


def _get_component_group(comp: str, component_to_group: Dict[str, str]) -> str:
    """Get the group (e.g. ``mlp`` or ``attn``) for a component name."""
    return component_to_group.get(comp, comp)


def _count_models_in_candidate(cand: dict) -> int:
    """Count number of unique models in a candidate description."""
    return len(cand.get("models", []))


def build_ops_df_from_groups(groups: List[dict]) -> pd.DataFrame:
    """Build a global ops DataFrame from a list of candidate groups.

    Each group dict must contain::

        'rows': DataFrame with columns ['model', 'layer', 'component', 'avgable', 'replaceable']
    """
    ops_rows: List[dict] = []

    for cand in groups:
        rows = cand["rows"].copy()

        if "avgable" not in rows.columns:
            rows["avgable"] = True
        if "replaceable" not in rows.columns:
            rows["replaceable"] = False

        rows_sorted = rows.sort_values("model")

        models_str = ",".join(
            f"{m}:{int(l)}"
            for m, l in rows_sorted[["model", "layer"]].itertuples(index=False, name=None)
        )

        for row in rows_sorted.itertuples(index=False):
            m = row.model
            layer = int(row.layer)
            comp = row.component

            is_avg = getattr(row, "avgable", True)
            is_rep = getattr(row, "replaceable", False)

            if is_avg:
                op_type = "merge"
            elif is_rep:
                op_type = "replace"
            else:
                op_type = "merge"

            source_str = f"{m}:{layer}"
            target_list = [x for x in models_str.split(",") if x.strip() and x.strip() != source_str]
            targets_str = ",".join(target_list)

            ops_rows.append(
                {
                    "model": m,
                    "op": op_type,
                    "component": comp,
                    "layer": layer,
                    "models": models_str,
                    "source": source_str,
                    "targets": targets_str,
                    "avgable": is_avg,
                    "replaceable": is_rep,
                }
            )

    if not ops_rows:
        return pd.DataFrame(
            columns=[
                "model",
                "op",
                "component",
                "layer",
                "models",
                "source",
                "targets",
                "avgable",
                "replaceable",
            ]
        )

    ops_df = pd.DataFrame(ops_rows)
    ops_df = ops_df.sort_values(["model", "layer", "component"]).reset_index(drop=True)
    return ops_df


def write_per_model_csvs(ops_df: pd.DataFrame, stage_name: str) -> None:
    """Write per-model CSVs for a given stage (step1 / step2 / step3)."""
    if ops_df.empty:
        print(f"No ops to write for {stage_name}")
        return

    if "pair_order" not in ops_df.columns:
        raise ValueError(f"ops_df for {stage_name} must contain 'pair_order'.")

    if "group" in ops_df.columns:
        ops_df["group"] = pd.Categorical(
            ops_df["group"],
            categories=["mlp", "attn", "other"],
            ordered=True,
        )

    for model_id, sub in ops_df.groupby("model"):
        df_m = sub.copy()

        df_m["comp_order"] = (
            df_m["component"].map(COMP_ORDER_MAP).fillna(len(COMP_ORDER_MAP)).astype(int)
        )

        df_m = df_m.sort_values(["pair_order", "comp_order", "component"])

        cols_to_drop = [
            "model",
            "group",
            "pair_order",
            "comp_order",
            "avgable",
            "replaceable",
        ]
        df_m_out = df_m.drop(
            columns=[c for c in cols_to_drop if c in df_m.columns], errors="ignore"
        ).reset_index(drop=True)

        final_cols = ["op", "component", "layer", "models", "source", "targets"]
        df_m_out = df_m_out[[c for c in final_cols if c in df_m_out.columns]]

        safe_model = str(model_id).replace("/", "_")
        fname = f"./ops_step_csvs/ops_{stage_name}_{safe_model}.csv"

        df_m_out.to_csv(fname, index=False)
        print(f"Wrote {len(df_m_out)} rows to {fname}")


def shift_model_std_to_target(
    df: pd.DataFrame,
    src_model: str,
    tgt_model: str,
    group_name: str,
    components: List[str],
) -> pd.DataFrame:
    """Shift std values of ``src_model`` components to match ``tgt_model`` mean std.

    Operates in-place on a copy of ``df`` and returns it.
    """
    src_mask = df["model"] == src_model
    tgt_mask = df["model"] == tgt_model

    if not src_mask.any() or not tgt_mask.any():
        print(f"Skipping {group_name}: missing data for {src_model} or {tgt_model}.")
        return df

    src_mean = df[src_mask & df["component"].isin(components)]["std"].mean()
    tgt_mean = df[tgt_mask & df["component"].isin(components)]["std"].mean()

    if pd.isna(src_mean) or pd.isna(tgt_mean):
        print(f"Skipping {group_name}: insufficient std values for {src_model} or {tgt_model}.")
        return df

    delta = src_mean - tgt_mean
    if abs(delta) < 1e-12:
        print(f"{group_name}: already aligned (delta≈0).")
        return df

    print(
        f"Shifting {src_model} {group_name} std by {-delta:+.6f} to match {tgt_model}."
    )
    mask = src_mask & df["component"].isin(components)
    df.loc[mask, "std"] = df.loc[mask, "std"] - delta
    return df


# -----------------------------------------------------------------------------
# Clustering helpers
# -----------------------------------------------------------------------------


def cluster_stats_df_by_component_target_nmodels(
    stats_df: pd.DataFrame,
    components=None,  # kept for backward compatibility but unused
    linkage_method: str = "ward",
) -> pd.DataFrame:
    """Cluster rows per (component, shape) combination using (mean, std).

    The number of clusters ``k`` is chosen so that each cluster has ~ ``n_models``
    entries on average, where ``n_models`` is the number of distinct models for
    that (component, shape) combination.
    """
    df = stats_df.copy()
    df["cluster_id"] = np.nan
    df["cluster_size"] = np.nan

    if "shape" not in df.columns:
        print("Warning: 'shape' column missing. Clustering by component only.")
        has_shape = False
    else:
        has_shape = True

    if has_shape:
        component_shape_groups = df.groupby(["component", "shape"], group_keys=False)
    else:
        component_shape_groups = df.groupby("component", group_keys=False)

    cluster_counter = 1
    for comp_shape_key, df_group in component_shape_groups:
        if has_shape:
            comp_name, shape_val = comp_shape_key
        else:
            comp_name = comp_shape_key
            shape_val = None

        n_rows = len(df_group)
        if n_rows == 0:
            continue

        n_models = df_group["model"].nunique()

        if n_rows <= n_models or n_models == 0:
            labels = np.arange(cluster_counter, cluster_counter + n_rows)
            sizes = np.ones(n_rows, dtype=int)
            df.loc[df_group.index, "cluster_id"] = labels
            df.loc[df_group.index, "cluster_size"] = sizes
            cluster_counter += n_rows
            continue

        X = df_group[["mean", "std"]].to_numpy(dtype=float)
        mu = X.mean(axis=0, keepdims=True)
        sigma = X.std(axis=0, keepdims=True) + 1e-12
        X_norm = (X - mu) / sigma

        k = max(1, int(round(n_rows / n_models)))
        Z = linkage(X_norm, method=linkage_method, metric="euclidean")
        labels_raw = fcluster(Z, t=k, criterion="maxclust")

        labels = labels_raw + cluster_counter - 1

        _, counts = np.unique(labels_raw, return_counts=True)
        size_map = {cid: c for cid, c in zip(sorted(np.unique(labels_raw)), counts)}
        sizes = np.array([size_map[cid] for cid in labels_raw])

        df.loc[df_group.index, "cluster_id"] = labels
        df.loc[df_group.index, "cluster_size"] = sizes

        cluster_counter += int(labels.max()) - cluster_counter + 1

    return df


def second_stage_hierarchical_clustering(
    second_stage_candidates_df: pd.DataFrame,
    components=None,
    linkage_method: str = "ward",
):
    """Stage-2 clustering on remaining candidates.

    For each component (and shape, if present), cluster remaining rows in
    (mean, std) space. For each cluster, keep at most one row per model
    (closest to centroid). Clusters with >= 2 distinct models become
    stage-2 merge candidates.
    """
    df = second_stage_candidates_df.copy()
    df["cluster2_id"] = np.nan

    if components is None:
        components = sorted(df["component"].unique())

    stage2_merge_candidates: List[dict] = []

    has_shape = "shape" in df.columns

    for comp_name in components:
        df_c = df[df["component"] == comp_name]
        n_rows = len(df_c)
        if n_rows < 2:
            continue

        n_models = df_c["model"].nunique()
        if n_models < 2:
            continue

        if has_shape:
            shape_groups = df_c.groupby("shape", group_keys=False)
        else:
            shape_groups = [(None, df_c)]

        for shape_val, df_c_shape in shape_groups:
            if df_c_shape.empty or len(df_c_shape) < 2:
                continue

            n_rows_shape = len(df_c_shape)
            n_models_shape = df_c_shape["model"].nunique()
            if n_models_shape < 2:
                continue

            X = df_c_shape[["mean", "std"]].to_numpy(dtype=float)
            mu = X.mean(axis=0, keepdims=True)
            sigma = X.std(axis=0, keepdims=True) + 1e-12
            X_norm = (X - mu) / sigma

            k = max(1, int(round(n_rows_shape / n_models_shape)))
            Z = linkage(X_norm, method=linkage_method, metric="euclidean")
            labels = fcluster(Z, t=k, criterion="maxclust")

            df_c_local = df_c_shape.copy()
            df_c_local["cluster2_id"] = labels

            df.loc[df_c_local.index, "cluster2_id"] = labels

            for cl_id, group in df_c_local.groupby("cluster2_id"):
                cl_id = int(cl_id)

                mean_center = group["mean"].mean()
                std_center = group["std"].mean()

                chosen_indices: List[int] = []
                chosen_rows: List[pd.Series] = []

                for model_id, g_m in group.groupby("model"):
                    d2 = (g_m["mean"] - mean_center) ** 2 + (g_m["std"] - std_center) ** 2
                    idx_best = d2.idxmin()
                    chosen_indices.append(idx_best)
                    chosen_rows.append(df.loc[idx_best])

                if len(chosen_rows) < 2:
                    continue

                rows_df = pd.DataFrame(chosen_rows)
                stage2_merge_candidates.append(
                    {
                        "component": comp_name,
                        "cluster2_id": cl_id,
                        "models": list(rows_df["model"].unique()),
                        "rows": rows_df,
                        "indices": chosen_indices,
                    }
                )

    return df, stage2_merge_candidates


# -----------------------------------------------------------------------------
# Reordering helpers
# -----------------------------------------------------------------------------


def reorder_ops_log_for_schedule(
    ops_log: pd.DataFrame,
    *,
    num_layers: int = 32,
    component_to_group: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Reorder ops_log using a two-phase schedule; replaces still come last.

    Phase 1 (mlp):
      - Schedule mlp-group merges across layers, grouping all mlp components of a
        layer together.
      - Prefer more total parameters, mid layers, and avoid consecutive layers
        for the same group if possible.
    Phase 2 (attn):
      - Schedule attn-group merges (attn_q/k/v/o collapse to 'attn') across layers,
        grouping all attn components of a layer together.
      - Same scoring: prefer more params, mid layers, and avoid consecutive layers.
    Finally, append all replace operations in their original order.
    """
    if ops_log is None or ops_log.empty:
        return ops_log

    df = ops_log.copy()

    merges = df[df["op"] == "merge"].copy()
    replaces = df[df["op"] == "replace"].copy()

    if merges.empty:
        return pd.concat([df[df["op"] != "replace"], replaces], ignore_index=True)

    def _count_models(models_str: str) -> int:
        if not isinstance(models_str, str) or not models_str.strip():
            return 0
        return sum(1 for tok in models_str.split(",") if tok.strip())

    merges = merges.reset_index().rename(columns={"index": "orig_idx"})
    merges["n_models"] = merges["models"].apply(_count_models)

    def _comp_group(comp: str) -> str:
        comp = str(comp)
        mapping = component_to_group or {}
        return mapping.get(comp, comp)

    merges["group"] = merges["component"].apply(_comp_group)
    merges["layer"] = merges["layer"].astype(int)

    attn_mask = merges["group"] == "attn"
    mlp_mask = merges["group"] == "mlp"

    attn_param_by_layer: Dict[int, float] = {}
    if attn_mask.any():
        _attn_sum = merges.loc[attn_mask].groupby("layer")["n_models"].sum()
        attn_param_by_layer = {int(k): float(v) for k, v in _attn_sum.items()}

    mlp_param_by_layer: Dict[int, float] = {}
    if mlp_mask.any():
        _mlp_sum = merges.loc[mlp_mask].groupby("layer")["n_models"].sum()
        mlp_param_by_layer = {int(k): (8.0 / 3.0) * float(v) for k, v in _mlp_sum.items()}

    mid_layer = (num_layers - 1) / 2.0

    def _reorder_one_group(
        group_name: str,
        param_count_by_layer: Dict[int, float],
    ) -> Tuple[pd.DataFrame, Set[int]]:
        sub = merges[merges["group"] == group_name].copy()
        if sub.empty:
            return sub, set()

        layer_to_rows: Dict[int, pd.DataFrame] = {}
        for layer, block in sub.sort_values("orig_idx").groupby("layer"):
            layer_to_rows[int(layer)] = block

        remaining_layers: Set[int] = set(layer_to_rows.keys())
        used_layers: Set[int] = set()
        selected_layers: List[int] = []

        def _is_adjacent_to_used(layer: int) -> bool:
            return any(abs(layer - ul) == 1 for ul in used_layers)

        def _score(layer: int) -> Tuple[int, float, float]:
            base_p = param_count_by_layer.get(
                layer, float(layer_to_rows[layer]["n_models"].sum())
            )
            dist_mid = abs(layer - mid_layer)
            consecutive_penalty = 1 if _is_adjacent_to_used(layer) else 0
            return (-consecutive_penalty, base_p, -dist_mid)

        while remaining_layers:
            candidates = list(remaining_layers)
            best_layer = max(candidates, key=_score)
            selected_layers.append(best_layer)
            remaining_layers.remove(best_layer)
            used_layers.add(best_layer)

        ordered = pd.concat(
            [layer_to_rows[L] for L in selected_layers], ignore_index=True
        )
        return ordered, used_layers

    phase1, _ = _reorder_one_group("mlp", mlp_param_by_layer)
    phase2, _ = _reorder_one_group("attn", attn_param_by_layer)

    cols_to_drop = [c for c in ["orig_idx", "n_models"] if c in phase1.columns]
    if not phase1.empty:
        phase1 = phase1.drop(columns=cols_to_drop)
    if not phase2.empty:
        phase2 = phase2.drop(columns=cols_to_drop)

    reordered = pd.concat(
        [
            phase1,
            phase2,
            replaces,
        ],
        ignore_index=True,
    )

    if "group" in reordered.columns:
        reordered["group"] = reordered["group"].fillna("other")
        cols = [c for c in reordered.columns if c != "group"] + ["group"]
        reordered = reordered[cols]

    return reordered
