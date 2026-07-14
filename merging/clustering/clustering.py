"""
Standalone runner for the clustering pipeline in `clustering/clustering.ipynb`.

Goal:
  - Replicate the notebook's behavior (operations + CSV generation) in a script.

Outputs (by default):
  - Per-model CSVs under `clustering_algorithm/ops_step_csvs/`:
      ops_step1_<model>.csv
      ops_step2_<model>.csv

Typical usage:
  python clustering/clustering.py --select-labels Light-IF-32B T-pro-it-2.0 MedGo
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


# Ensure we can import `clustering_helpers.py` regardless of cwd.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from clustering_helpers import (  # noqa: E402
    augment_stats_with_profiles,
    build_ops_df_from_groups,
    cluster_stats_df_by_component_target_nmodels,
    compute_stats_for_models,
    reorder_ops_log_for_schedule,
    second_stage_hierarchical_clustering,
    shift_model_std_to_target,
)


def _repo_root() -> Path:
    # `.../merge_tools/clustering/clustering.py` -> repo root is parent of `clustering/`
    return Path(__file__).resolve().parents[1]


def build_default_registry(root: Path) -> Tuple[List[str], List[str], Dict[str, dict]]:
    """Mirror the notebook's default model registry + profile paths."""
    llama_mag_dir = root / "clustering" / "magnitude_profile" / "llama_latest"
    llama_noise_dir = root / "clustering" / "noise_profile" / "llama_latest"

    mag_qwen_dir = root / "clustering" / "magnitude_profile" / "qwen"
    noise_qwen_dir = root / "clustering" / "noise_profile" / "qwen"

    mag_deepseek_dir = root / "clustering" / "magnitude_profile" / "deepseek"
    noise_deepseek_dir = root / "clustering" / "noise_profile" / "deepseek"

    mag_qwen32_dir = root / "clustering" / "magnitude_profile" / "qwen-32b"
    noise_qwen32_dir = root / "clustering" / "noise_profile" / "qwen-32b"

    model_configs: Dict[str, dict] = {
        # Llama 8B models
        "Llama-3.1-8B-UltraMedical": {
            "task": "medical",
            "std_path": llama_mag_dir / "medical_magnitude_profile.csv",
            "noise_path": llama_noise_dir / "llama_medical_profile.csv",
        },
        "Llama-3.1-Hawkish-8B": {
            "task": "finance",
            "std_path": llama_mag_dir / "finance_magnitude_profile.csv",
            "noise_path": llama_noise_dir / "llama_finance_profile.csv",
        },
        "calme-2.3-legalkit-8b": {
            "task": "legal",
            "std_path": llama_mag_dir / "legal_magnitude_profile.csv",
            "noise_path": llama_noise_dir / "llama_legal_profile.csv",
        },
        "Llama-SafetyGuard-Content-Binary": {
            "task": "toxicity",
            "std_path": llama_mag_dir / "toxicity_magnitude_profile.csv",
            "noise_path": llama_noise_dir / "llama_toxicity_profile.csv",
        },
        "Llama-3.1-8B-Instruct-multi-truth-judge": {
            "task": "truthfulness",
            "std_path": llama_mag_dir / "truthfulness_magnitude_profile.csv",
            "noise_path": llama_noise_dir / "llama_truthfulness_profile.csv",
        },
        # Qwen 7B models
        "Qwen2.5-Math-7B-Instruct": {
            "task": "math",
            "std_path": mag_qwen_dir / "qwen25_7b_math_stats.csv",
            "noise_path": noise_qwen_dir / "qwen_math_profile.csv",
        },
        "Qwen2.5-Coder-7B-Instruct": {
            "task": "coder",
            "std_path": mag_qwen_dir / "qwen25_7b_coder_stats.csv",
            "noise_path": noise_qwen_dir / "qwen_coder_profile.csv",
        },
        # DeepSeek 7B models
        "deepseek-math-7b-instruct": {
            "task": "math",
            "std_path": mag_deepseek_dir / "dsmath_magnitude_profile.csv",
            "noise_path": noise_deepseek_dir / "dsmath_profile.csv",
        },
        "deepseek-coder-7b-instruct-v1.5": {
            "task": "coder",
            "std_path": mag_deepseek_dir / "dscoder_magnitude_profile.csv",
            "noise_path": noise_deepseek_dir / "dscoder_profile.csv",
        },
        # Qwen 32B models
        "Light-IF-32B": {
            "task": "tinymmlu",
            "std_path": mag_qwen32_dir / "Light-IF-32B_stats.csv",
            "noise_path": noise_qwen32_dir / "light-if-32b_tinymmlu_noise_profile.csv",
        },
        "T-pro-it-2.0": {
            "task": "mmlu_ru",
            "std_path": mag_qwen32_dir / "T-pro-it-2.0_stats.csv",
            "noise_path": noise_qwen32_dir / "T-pro-it-2.0_mmlu_ru_noise_profile.csv",
        },
        "MedGo": {
            "task": "medqa",
            "std_path": mag_qwen32_dir / "MedGo_stats.csv",
            "noise_path": noise_qwen32_dir / "medgo_medqa_noise_profile.csv",
        },
    }

    all_model_labels: List[str] = [
        "Qwen2.5-Math-7B-Instruct",
        "Qwen2.5-Coder-7B-Instruct",
        "deepseek-math-7b-instruct",
        "deepseek-coder-7b-instruct-v1.5",
        "Llama-3.1-8B-UltraMedical",
        "Llama-3.1-Hawkish-8B",
        "calme-2.3-legalkit-8b",
        "Llama-SafetyGuard-Content-Binary",
        "Llama-3.1-8B-Instruct-multi-truth-judge",
        "Light-IF-32B",
        "T-pro-it-2.0",
        "MedGo",
    ]

    # Resolve model weights to ONE local cache root, matching run_figures.load_registry:
    # $SANDHI_MODELS_DIR if set, else $HF_HOME/models (repo-local default). No hardcoded
    # /scratch path -- clustering runs as a run_figures subprocess and inherits HF_HOME,
    # so these match the profiler's model paths in both fully-local and reuse runs.
    import os as _os
    _cache_root = _os.environ.get("SANDHI_MODELS_DIR") or _os.path.join(
        _os.environ.get("HF_HOME") or str(root / "hf_cache"), "models")
    all_model_paths: List[str] = [_os.path.join(_cache_root, lab) for lab in all_model_labels]

    if len(all_model_paths) != len(all_model_labels):
        raise RuntimeError("Internal error: ALL_MODEL_PATHS and ALL_MODEL_LABELS length mismatch.")

    return all_model_paths, all_model_labels, model_configs


def select_registry(
    *,
    all_model_paths: List[str],
    all_model_labels: List[str],
    all_model_configs: Dict[str, dict],
    selected_labels: Optional[List[str]],
) -> Tuple[List[str], List[str], Dict[str, dict]]:
    if not selected_labels:
        return list(all_model_paths), list(all_model_labels), dict(all_model_configs)

    selected_set: Set[str] = set(selected_labels)
    missing = sorted(list(selected_set - set(all_model_labels)))
    if missing:
        print("Warning: requested labels not found and will be ignored:", missing)

    model_paths: List[str] = []
    model_labels: List[str] = []
    for p, lab in zip(all_model_paths, all_model_labels):
        if lab in selected_set:
            model_paths.append(p)
            model_labels.append(lab)

    model_configs = {k: v for k, v in all_model_configs.items() if k in selected_set}
    return model_paths, model_labels, model_configs


def zscore_within_component_shape(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Match the notebook's per-(component,shape) z-scoring for mean/std."""
    df = stats_df.copy()
    cols_to_normalize = [c for c in ["mean", "std"] if c in df.columns]
    if not cols_to_normalize:
        return df

    group_cols = ["component"]
    if "shape" in df.columns:
        group_cols.append("shape")

    def zscore(x: pd.Series) -> pd.Series:
        if len(x) <= 1 or float(x.std()) == 0.0:
            return x - x.mean()
        return (x - x.mean()) / x.std()

    for col in cols_to_normalize:
        df[col] = df.groupby(group_cols)[col].transform(zscore)

    return df


def build_stage1_candidates(clustered_df: pd.DataFrame) -> List[dict]:
    """Build initial merge candidate groups (stage-1) as in the notebook."""
    merge_candidates_initial: List[dict] = []

    if "shape" in clustered_df.columns:
        groupby_keys = ["component", "cluster_id", "shape"]
    else:
        groupby_keys = ["component", "cluster_id"]
        print("Warning: 'shape' column not found. Grouping by component and cluster_id only.")

    for group_key, comp_cluster_group in clustered_df.groupby(groupby_keys):
        if isinstance(group_key, tuple):
            if len(group_key) == 3:
                comp_name, cid, _shape_val = group_key
            else:
                comp_name, cid = group_key
        else:
            comp_name, cid = group_key

        if pd.isna(cid):
            continue
        cid_int = int(cid)

        groups: List[dict] = []
        used_indices: Set[int] = set()

        # 1) same-layer groups first
        for layer_idx, layer_group in comp_cluster_group.groupby("layer"):
            if layer_group["model"].nunique() < 2:
                continue

            idxs = list(layer_group.index)
            models = set(layer_group["model"])

            mean_center = layer_group["mean"].mean()
            std_center = layer_group["std"].mean()

            groups.append(
                {
                    "component": comp_name,
                    "cluster_id": cid_int,
                    "anchor_layer": int(layer_idx),
                    "indices": idxs[:],
                    "models": set(models),
                    "centroid_mean": float(mean_center),
                    "centroid_std": float(std_center),
                }
            )
            used_indices.update(idxs)

        # 2) fill with leftovers (different layers), only if we have base groups
        if groups:
            leftover = comp_cluster_group[~comp_cluster_group.index.isin(used_indices)]
            for idx, row in leftover.iterrows():
                model_id = row["model"]
                mean_val = row["mean"]
                std_val = row["std"]

                best_group = None
                best_d2 = None

                for g in groups:
                    if model_id in g["models"]:
                        continue
                    d2 = (mean_val - g["centroid_mean"]) ** 2 + (std_val - g["centroid_std"]) ** 2
                    if best_d2 is None or d2 < best_d2:
                        best_d2 = d2
                        best_group = g

                if best_group is not None:
                    best_group["indices"].append(idx)
                    best_group["models"].add(model_id)

                    # update centroid
                    n_old = len(best_group["indices"]) - 1
                    best_group["centroid_mean"] = (best_group["centroid_mean"] * n_old + mean_val) / (
                        n_old + 1
                    )
                    best_group["centroid_std"] = (best_group["centroid_std"] * n_old + std_val) / (
                        n_old + 1
                    )
                    used_indices.add(idx)

        # 3) register groups
        for g in groups:
            rows = clustered_df.loc[g["indices"]].copy()
            merge_candidates_initial.append(
                {
                    "component": g["component"],
                    "cluster_id": g["cluster_id"],
                    "anchor_layer": g["anchor_layer"],
                    "models": sorted(list(g["models"])),
                    "rows": rows,
                    "indices": g["indices"],
                }
            )

    return merge_candidates_initial


def split_and_order_candidates(
    merge_candidates_initial: List[dict],
) -> Tuple[List[dict], List[dict]]:
    """Split candidates into step1 (aligned layers) and step2 (unaligned layers)."""
    step1_candidates: List[dict] = []
    step2_candidates: List[dict] = []

    for cand in merge_candidates_initial:
        rows = cand["rows"]
        anchor_layer = cand["anchor_layer"]

        unique_layers = rows["layer"].unique()
        is_aligned = len(unique_layers) == 1

        if is_aligned:
            step1_candidates.append(cand)
        else:
            aligned_rows = rows[rows["layer"] == anchor_layer].copy()
            aligned_indices = aligned_rows.index.tolist()

            if len(aligned_rows) >= 2 and aligned_rows["model"].nunique() >= 2:
                step1_candidates.append(
                    {
                        "component": cand["component"],
                        "cluster_id": cand["cluster_id"],
                        "anchor_layer": anchor_layer,
                        "models": sorted(list(aligned_rows["model"].unique())),
                        "rows": aligned_rows,
                        "indices": aligned_indices,
                    }
                )

            # full candidate to step2
            step2_candidates.append(cand)

    # Reorder step1 candidates (mlp first, then attn) – matches notebook helper.
    component_to_group = {
        "mlp_gate": "mlp",
        "mlp_up": "mlp",
        "mlp_down": "mlp",
        "attn_q": "attn",
        "attn_k": "attn",
        "attn_v": "attn",
        "attn_o": "attn",
    }

    def _get_component_group(comp: str) -> str:
        return component_to_group.get(comp, comp)

    def _count_models_in_candidate(cand: dict) -> int:
        return len(cand.get("models", []))

    def reorder_step1_candidates(candidates: List[dict], num_layers: Optional[int] = None) -> List[dict]:
        if not candidates:
            return candidates

        if num_layers is None:
            max_layer = max(int(c["anchor_layer"]) for c in candidates)
            num_layers = max_layer + 1

        enriched: List[dict] = []
        for cand in candidates:
            comp = cand["component"]
            layer = int(cand["anchor_layer"])
            n_models = _count_models_in_candidate(cand)
            group = _get_component_group(comp)
            enriched.append({**cand, "group": group, "n_models": n_models, "layer": layer})

        mlp_param_by_layer: Dict[int, float] = {}
        attn_param_by_layer: Dict[int, float] = {}
        for cand in enriched:
            layer = int(cand["layer"])
            group = cand["group"]
            n_models = float(cand["n_models"])
            if group == "mlp":
                mlp_param_by_layer[layer] = mlp_param_by_layer.get(layer, 0.0) + (8.0 / 3.0) * n_models
            elif group == "attn":
                attn_param_by_layer[layer] = attn_param_by_layer.get(layer, 0.0) + n_models

        mid_layer = (num_layers - 1) / 2.0

        def _reorder_one_group(group_name: str, param_count_by_layer: Dict[int, float]) -> List[dict]:
            sub = [c for c in enriched if c["group"] == group_name]
            if not sub:
                return []

            layer_to_cands: Dict[int, List[dict]] = {}
            for cand in sub:
                layer_to_cands.setdefault(int(cand["layer"]), []).append(cand)

            remaining_layers: Set[int] = set(layer_to_cands.keys())
            used_layers: Set[int] = set()
            selected_layers: List[int] = []

            def _is_adjacent_to_used(layer: int) -> bool:
                return any(abs(layer - ul) == 1 for ul in used_layers)

            def _score(layer: int) -> Tuple[float, float, float]:
                import os as _os
                _mid = _os.environ.get("ORD_MIDDLE", "1") == "1"
                _spc = _os.environ.get("ORD_SPACING", "1") == "1"
                base_p = float(
                    param_count_by_layer.get(layer, sum(float(c["n_models"]) for c in layer_to_cands[layer]))
                )
                dist_mid = abs(layer - mid_layer)
                consecutive_penalty = 1.0 if _is_adjacent_to_used(layer) else 0.0
                return (base_p, (-dist_mid if _mid else 0.0), (-consecutive_penalty if _spc else 0.0))

            while remaining_layers:
                best_layer = max(list(remaining_layers), key=_score)
                selected_layers.append(best_layer)
                remaining_layers.remove(best_layer)
                used_layers.add(best_layer)

            ordered: List[dict] = []
            for layer in selected_layers:
                ordered.extend(layer_to_cands[layer])
            return ordered

        import os as _os
        if _os.environ.get("ORD_MLP_FIRST", "1") == "1":
            phase1 = _reorder_one_group("mlp", mlp_param_by_layer)
            phase2 = _reorder_one_group("attn", attn_param_by_layer)
            reordered = phase1 + phase2
        else:
            # MLP-first OFF: order mlp+attn together by a combined per-layer score
            _comb = {}
            for _l in set(list(mlp_param_by_layer) + list(attn_param_by_layer)):
                _comb[_l] = mlp_param_by_layer.get(_l, 0.0) + attn_param_by_layer.get(_l, 0.0)
            _l2c: Dict[int, List[dict]] = {}
            for _c in enriched:
                _l2c.setdefault(int(_c["layer"]), []).append(_c)
            _rem = set(_l2c); _used: Set[int] = set(); _sel = []
            _mid2 = _os.environ.get("ORD_MIDDLE", "1") == "1"; _spc2 = _os.environ.get("ORD_SPACING", "1") == "1"
            def _adj2(l): return any(abs(l - u) == 1 for u in _used)
            def _sc2(l): return (_comb.get(l, 0.0), (-abs(l - mid_layer) if _mid2 else 0.0), (-(1.0 if _adj2(l) else 0.0) if _spc2 else 0.0))
            while _rem:
                _b = max(_rem, key=_sc2); _sel.append(_b); _rem.discard(_b); _used.add(_b)
            reordered = [c for _l in _sel for c in _l2c[_l]]

        # strip metadata
        for cand in reordered:
            cand.pop("group", None)
            cand.pop("n_models", None)
            cand.pop("layer", None)
        return reordered

    step1_candidates = reorder_step1_candidates(step1_candidates)

    # Order step2_candidates: those also in step1 first, in step1 order.
    step1_key_to_index: Dict[Tuple[str, int], int] = {}
    for idx, cand in enumerate(step1_candidates):
        step1_key_to_index[(cand["component"], int(cand["cluster_id"]))] = idx

    step2_in_step1: List[Tuple[int, dict]] = []
    step2_only: List[dict] = []
    for cand in step2_candidates:
        key = (cand["component"], int(cand["cluster_id"]))
        if key in step1_key_to_index:
            step2_in_step1.append((step1_key_to_index[key], cand))
        else:
            step2_only.append(cand)

    step2_in_step1.sort(key=lambda x: x[0])
    step2_candidates = [cand for _, cand in step2_in_step1] + step2_only

    return step1_candidates, step2_candidates


def build_second_stage_candidates_df(clustered_df: pd.DataFrame, step1: List[dict], step2: List[dict]) -> pd.DataFrame:
    all_step1_indices: Set[int] = set()
    for cand in step1:
        all_step1_indices.update(cand["indices"])

    all_step2_indices: Set[int] = set()
    for cand in step2:
        all_step2_indices.update(cand["indices"])

    step2_rows_df = clustered_df.loc[sorted(all_step2_indices)].copy() if all_step2_indices else pd.DataFrame()

    leftover_indices = set(clustered_df.index) - set(all_step1_indices) - set(all_step2_indices)
    leftover_df = clustered_df.loc[sorted(leftover_indices)].copy() if leftover_indices else pd.DataFrame()

    if not step2_rows_df.empty and not leftover_df.empty:
        return pd.concat([step2_rows_df, leftover_df], ignore_index=False)
    if not step2_rows_df.empty:
        return step2_rows_df
    if not leftover_df.empty:
        return leftover_df
    return pd.DataFrame(columns=clustered_df.columns)


def schedule_pair_order(ops_df_stage1: pd.DataFrame, ops_df_stage2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Attach a global `pair_order` column to both stages (matches notebook)."""
    pair_key = ["component", "layer"]

    if not ops_df_stage1.empty:
        s1_pairs = ops_df_stage1[pair_key].drop_duplicates().reset_index(drop=True)
        s1_pairs["pair_order"] = range(len(s1_pairs))
    else:
        s1_pairs = pd.DataFrame(columns=pair_key + ["pair_order"])

    if not ops_df_stage2.empty:
        s2_pairs = ops_df_stage2[pair_key].drop_duplicates()
        s2_pairs = s2_pairs.merge(s1_pairs[pair_key + ["pair_order"]], on=pair_key, how="left")

        new_pairs = s2_pairs[s2_pairs["pair_order"].isna()][pair_key].drop_duplicates().reset_index(drop=True)
        if not new_pairs.empty:
            new_pairs["pair_order"] = range(len(s1_pairs), len(s1_pairs) + len(new_pairs))
            pair_order_df = pd.concat([s1_pairs, new_pairs], ignore_index=True)
        else:
            pair_order_df = s1_pairs
    else:
        pair_order_df = s1_pairs

    if not ops_df_stage1.empty:
        ops_df_stage1 = ops_df_stage1.merge(pair_order_df, on=pair_key, how="left")
    if not ops_df_stage2.empty:
        ops_df_stage2 = ops_df_stage2.merge(pair_order_df, on=pair_key, how="left")

    return ops_df_stage1, ops_df_stage2


def write_per_model_csvs_to_dir(ops_df: pd.DataFrame, stage_name: str, out_dir: Path) -> None:
    """Equivalent to `write_per_model_csvs`, but with explicit output directory."""
    if ops_df.empty:
        print(f"No ops to write for {stage_name}")
        return

    if "pair_order" not in ops_df.columns:
        raise ValueError(f"ops_df for {stage_name} must contain 'pair_order'.")

    out_dir.mkdir(parents=True, exist_ok=True)

    comp_order = [
        "attn",
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_o",
        "mlp",
        "mlp_gate",
        "mlp_up",
        "mlp_down",
    ]
    comp_order_map = {comp: i for i, comp in enumerate(comp_order)}

    if "group" in ops_df.columns:
        ops_df = ops_df.copy()
        ops_df["group"] = pd.Categorical(ops_df["group"], categories=["mlp", "attn", "other"], ordered=True)

    for model_id, sub in ops_df.groupby("model"):
        df_m = sub.copy()
        df_m["comp_order"] = df_m["component"].map(comp_order_map).fillna(len(comp_order_map)).astype(int)
        df_m = df_m.sort_values(["pair_order", "comp_order", "component"])

        cols_to_drop = ["model", "group", "pair_order", "comp_order", "avgable", "replaceable"]
        df_m_out = df_m.drop(columns=[c for c in cols_to_drop if c in df_m.columns], errors="ignore").reset_index(
            drop=True
        )

        final_cols = ["op", "component", "layer", "models", "source", "targets"]
        df_m_out = df_m_out[[c for c in final_cols if c in df_m_out.columns]]

        safe_model = str(model_id).replace("/", "_")
        fname = out_dir / f"ops_{stage_name}_{safe_model}.csv"
        df_m_out.to_csv(fname, index=False)
        print(f"Wrote {len(df_m_out)} rows to {fname}")


def main() -> None:
    root = _repo_root()

    p = argparse.ArgumentParser(description="Run clustering pipeline and write ops_step CSVs.")
    p.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device hint for model loading (uses cuda only if available).",
    )
    p.add_argument("--start-layer", type=int, default=None)
    p.add_argument("--end-layer", type=int, default=None)
    p.add_argument(
        "--select-labels",
        nargs="*",
        default=None,
        help="Subset of model labels to run (space-separated). If omitted, uses all.",
    )
    p.add_argument(
        "--stats-csv",
        type=str,
        default=None,
        help="Optional: path to a precomputed stats_df CSV to skip model loading.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(root / "clustering" / "ops_step_csvs"),
        help="Directory where per-model ops CSVs will be written.",
    )
    p.add_argument(
        "--no-augment-profiles",
        action="store_true",
        help="Skip augmenting stats with cosine/noise profiles (avgable filtering will be disabled).",
    )
    p.add_argument(
        "--no-std-shift",
        action="store_true",
        help="Disable the notebook's std-shifting alignment across model families.",
    )
    p.add_argument(
        "--skip-stage2",
        action="store_true",
        help="Skip stage-2 clustering (writes step1 CSVs only).",
    )
    p.add_argument(
        "--baseline-drop-threshold",
        type=float,
        default=2.0,
        help="Mark a row avgable/replaceable if its profile score is at least baseline minus this value.",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)

    all_paths, all_labels, all_configs = build_default_registry(root)
    model_paths, model_labels, model_configs = select_registry(
        all_model_paths=all_paths,
        all_model_labels=all_labels,
        all_model_configs=all_configs,
        selected_labels=args.select_labels,
    )

    # Re-run wiring: MICR_NOISE_PROFILE_DIR points at a directory of fresh
    # gaussian profiler outputs (gaussian_<label>.csv). Any label with a file
    # there has its noise_path overridden; others keep the registry default.
    # The gaussian CSV schema is a subset-compatible superset of the legacy
    # noise-profile schema (perturbation/variant/layer/score all present).
    _noise_dir = os.environ.get("MICR_NOISE_PROFILE_DIR")
    if _noise_dir:
        for _lbl, _cfg in model_configs.items():
            _cand = Path(_noise_dir) / f"gaussian_{_lbl}.csv"
            if _cand.exists():
                print(f"[noise-override] {_lbl}: {_cand}")
                _cfg["noise_path"] = _cand
    print("Active models:", model_labels)

    # --- compute / load stats ---
    if args.stats_csv:
        stats_df = pd.read_csv(args.stats_csv)
        print(f"Loaded stats_df from {args.stats_csv} with {len(stats_df)} rows.")
    else:
        stats_df = compute_stats_for_models(
            model_paths=model_paths,
            model_labels=model_labels,
            device=args.device,
            start_layer=args.start_layer,
            end_layer=args.end_layer,
        )
        print("Computed stats_df rows:", len(stats_df))

    # --- augment + filter ---
    if not args.no_augment_profiles:
        stats_df = augment_stats_with_profiles(
            stats_df,
            model_configs,
            baseline_drop_threshold=args.baseline_drop_threshold,
        )

        if not args.no_std_shift:
            # Cross-base std alignment. shift_model_std_to_target no-ops when the
            # named target is absent from the pool, so each line only fires for
            # the pool it applies to. DeepSeek-Coder-v1.5 and DeepSeek-Math have
            # DIFFERENT bases, so a same-pool run (deepseek2) must shift one onto
            # the other (math -> coder), mirroring the Qwen pair below. The old
            # deepseek -> Llama-Hawkish shift only made sense in the retired
            # JOINT llama+deepseek clustering (we now cluster each family
            # separately and compose at analysis), and it no-ops in a
            # deepseek-only pool -> 0 candidates. Kept as an extra line so a
            # legacy joint pool still aligns.
            stats_df = shift_model_std_to_target(stats_df, "deepseek-math-7b-instruct", "deepseek-coder-7b-instruct-v1.5", "attn", ["attn", "attn_q", "attn_k", "attn_v", "attn_o"])
            stats_df = shift_model_std_to_target(stats_df, "deepseek-math-7b-instruct", "deepseek-coder-7b-instruct-v1.5", "mlp", ["mlp", "mlp_gate", "mlp_up", "mlp_down"])
            for m_src in ["deepseek-coder-7b-instruct-v1.5", "deepseek-math-7b-instruct"]:
                stats_df = shift_model_std_to_target(stats_df, m_src, "Llama-3.1-Hawkish-8B", "attn", ["attn", "attn_q", "attn_k", "attn_v", "attn_o"])
                stats_df = shift_model_std_to_target(stats_df, m_src, "Llama-3.1-Hawkish-8B", "mlp", ["mlp", "mlp_gate", "mlp_up", "mlp_down"])

            # Shift Qwen Math to Qwen Coder (as in notebook)
            stats_df = shift_model_std_to_target(
                stats_df,
                "Qwen2.5-Math-7B-Instruct",
                "Qwen2.5-Coder-7B-Instruct",
                "attn",
                ["attn", "attn_q", "attn_k", "attn_v", "attn_o"],
            )
            stats_df = shift_model_std_to_target(
                stats_df,
                "Qwen2.5-Math-7B-Instruct",
                "Qwen2.5-Coder-7B-Instruct",
                "mlp",
                ["mlp", "mlp_gate", "mlp_up", "mlp_down"],
            )

        # Keep rows where avgable is True (matches notebook default)
        if "avgable" in stats_df.columns:
            stats_df = stats_df[stats_df["avgable"] == True]  # noqa: E712

    keep_components = ["mlp_gate", "mlp_up", "mlp_down", "attn_q", "attn_k", "attn_v", "attn_o"]
    stats_df = stats_df[stats_df["component"].isin(keep_components)].reset_index(drop=True)

    stats_df = zscore_within_component_shape(stats_df)

    if "shape" in stats_df.columns:
        print("Size after component filtering:", len(stats_df))
        print("Unique (component, shape) groups:", stats_df.groupby(["component", "shape"]).ngroups)

    # --- stage 1 clustering + grouping ---
    clustered_df = cluster_stats_df_by_component_target_nmodels(stats_df)
    print("Clustered rows:", len(clustered_df))

    merge_candidates_initial = build_stage1_candidates(clustered_df)
    print("Number of merge_candidates_initial groups:", len(merge_candidates_initial))

    step1_candidates, step2_candidates = split_and_order_candidates(merge_candidates_initial)
    print("Step 1 candidates (aligned layers only):", len(step1_candidates))
    print("Step 2 candidates (unaligned layers):", len(step2_candidates))

    # The notebook overwrites merge_candidates_initial with step1 (aligned-only).
    merge_candidates_initial = step1_candidates

    # Build second-stage candidate pool (step2 candidates + leftovers)
    second_stage_candidates_df = build_second_stage_candidates_df(clustered_df, step1_candidates, step2_candidates)
    print("Rows in second_stage_candidates_df:", len(second_stage_candidates_df))

    stage2_merge_candidates: List[dict] = []
    if not args.skip_stage2 and not second_stage_candidates_df.empty:
        all_components = keep_components[:]  # notebook's ALL_COMPONENTS at this point
        _, stage2_merge_candidates = second_stage_hierarchical_clustering(
            second_stage_candidates_df,
            components=all_components,
            linkage_method="ward",
        )
        print("Total stage-2 merge candidate groups:", len(stage2_merge_candidates))

    # --- build ops dfs ---
    ops_df_stage1 = build_ops_df_from_groups(merge_candidates_initial)
    if "source" in ops_df_stage1.columns or "targets" in ops_df_stage1.columns:
        ops_df_stage1 = ops_df_stage1.drop(columns=["source", "targets"], errors="ignore")

    if stage2_merge_candidates:
        ops_df_stage2 = build_ops_df_from_groups(stage2_merge_candidates)
        ops_df_stage2 = ops_df_stage2.drop(columns=["source", "targets"], errors="ignore")
    else:
        ops_df_stage2 = pd.DataFrame(columns=["model", "op", "component", "layer", "models"])

    # schedule stage 1 (two-phase schedule)
    attn_subcomps = ["attn", "attn_q", "attn_k", "attn_v", "attn_o"]
    mlp_subcomps = ["mlp", "mlp_gate", "mlp_up", "mlp_down"]
    component_to_group = {c: "attn" for c in attn_subcomps}
    component_to_group.update({c: "mlp" for c in mlp_subcomps})

    ops_df_stage1 = reorder_ops_log_for_schedule(ops_df_stage1, component_to_group=component_to_group)

    # schedule stage 2 while preserving comp-layer order seen in stage 1
    if not ops_df_stage2.empty:
        if not ops_df_stage1.empty:
            s1_comp_layer_order = ops_df_stage1[["component", "layer"]].drop_duplicates()
            s1_comp_layer_order["s1_order"] = range(len(s1_comp_layer_order))

            ops_df_stage2_merged = pd.merge(
                ops_df_stage2, s1_comp_layer_order, on=["component", "layer"], how="left"
            )
            s2_seen_in_s1 = (
                ops_df_stage2_merged[ops_df_stage2_merged["s1_order"].notna()]
                .sort_values("s1_order")
                .drop(columns="s1_order")
            )
            s2_new = ops_df_stage2_merged[ops_df_stage2_merged["s1_order"].isna()].drop(columns="s1_order")

            s2_new_scheduled = reorder_ops_log_for_schedule(s2_new)
            ops_df_stage2 = pd.concat([s2_seen_in_s1, s2_new_scheduled], ignore_index=True)
        else:
            ops_df_stage2 = reorder_ops_log_for_schedule(ops_df_stage2)

    print("Stage 1 ops rows:", len(ops_df_stage1))
    print("Stage 2 ops rows:", len(ops_df_stage2))

    # attach pair_order and write per-model csvs
    ops_df_stage1, ops_df_stage2 = schedule_pair_order(ops_df_stage1, ops_df_stage2)
    write_per_model_csvs_to_dir(ops_df_stage1, "step1", out_dir)
    write_per_model_csvs_to_dir(ops_df_stage2, "step2", out_dir)


if __name__ == "__main__":
    main()

