from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Set, Tuple

import pandas as pd

import clustering_helpers as ch
from clustering import (
    build_default_registry,
    build_second_stage_candidates_df,
    build_stage1_candidates,
    schedule_pair_order,
    select_registry,
    split_and_order_candidates,
    write_per_model_csvs_to_dir,
    zscore_within_component_shape,
)


LLAMA_LABELS = [
    "Llama-3.1-8B-UltraMedical",
    "Llama-3.1-Hawkish-8B",
    "calme-2.3-legalkit-8b",
    "Llama-SafetyGuard-Content-Binary",
    "Llama-3.1-8B-Instruct-multi-truth-judge",
]

DEEPSEEK_LABELS = [
    "deepseek-math-7b-instruct",
    "deepseek-coder-7b-instruct-v1.5",
]

QWEN7B_LABELS = [
    "Qwen2.5-Math-7B-Instruct",
    "Qwen2.5-Coder-7B-Instruct",
]

STANDARDIZED_CASE_SPECS: Dict[str, List[str]] = {
    "6_deepseek_standardized": DEEPSEEK_LABELS,
    "6d_deepseek_noprofiling": DEEPSEEK_LABELS,
    "6pa_deepseek_noprofiling": DEEPSEEK_LABELS,
    "6pb_deepseek_noprofiling": DEEPSEEK_LABELS,
    "6pc_deepseek_noprofiling": DEEPSEEK_LABELS,
    "6profiler": DEEPSEEK_LABELS,
    "7_llamas_standardized": LLAMA_LABELS,
    "9model_standardized": [*DEEPSEEK_LABELS, *LLAMA_LABELS, *QWEN7B_LABELS],
}

LAYER_LEVEL_STANDARDIZED_CASE_SPECS: Dict[str, List[str]] = {
    "6c_deepseek_standardized_layer_level": DEEPSEEK_LABELS,
    "6c_deepseek_standardized_layer_level_thigh": DEEPSEEK_LABELS,
    "7b_llamas_standardized_layer_level": LLAMA_LABELS,
}

KEEP_COMPONENTS = [
    "mlp_gate",
    "mlp_up",
    "mlp_down",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_o",
]

ATTN_SUBCOMPS = ["attn_q", "attn_k", "attn_v", "attn_o"]
MLP_SUBCOMPS = ["mlp_gate", "mlp_up", "mlp_down"]
MID_LAYER = 15.5


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _candidates_root(root: Path) -> Path:
    return root / "clustering" / "candidates"


def _artifacts_root(root: Path) -> Path:
    return _candidates_root(root) / "_artifacts"


def _get_llama_registry(root: Path) -> Tuple[List[str], List[str], Dict[str, dict]]:
    all_paths, all_labels, all_configs = build_default_registry(root)
    return select_registry(
        all_model_paths=all_paths,
        all_model_labels=all_labels,
        all_model_configs=all_configs,
        selected_labels=LLAMA_LABELS,
    )


def _get_registry_for_labels(root: Path, labels: Sequence[str]) -> Tuple[List[str], List[str], Dict[str, dict]]:
    all_paths, all_labels, all_configs = build_default_registry(root)
    return select_registry(
        all_model_paths=all_paths,
        all_model_labels=all_labels,
        all_model_configs=all_configs,
        selected_labels=list(labels),
    )


def _write_summary(path: Path, summary: Dict[str, object]) -> None:
    lines = [f"{k}: {v}" for k, v in summary.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_prepared_stats(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / name, index=False)


def _standardize_by_model(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def _zscore(series: pd.Series) -> pd.Series:
        if len(series) <= 1 or float(series.std()) == 0.0:
            return series - series.mean()
        return (series - series.mean()) / series.std()

    for col in ["mean", "std"]:
        out[col] = out.groupby("model")[col].transform(_zscore)
    return out


def _normalize_by_model(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def _minmax(series: pd.Series) -> pd.Series:
        lo = float(series.min())
        hi = float(series.max())
        if math.isclose(lo, hi):
            return pd.Series([0.0] * len(series), index=series.index)
        return (series - lo) / (hi - lo)

    for col in ["mean", "std"]:
        out[col] = out.groupby("model")[col].transform(_minmax)
    return out


def _prepare_subcomponent_stats(
    raw_stats_df: pd.DataFrame,
    model_configs: Dict[str, dict],
    transform: Callable[[pd.DataFrame], pd.DataFrame] | None,
    *,
    apply_avgable_filter: bool = True,
    baseline_drop_threshold: float = 2.0,
) -> pd.DataFrame:
    df = ch.augment_stats_with_profiles(
        raw_stats_df,
        model_configs,
        baseline_drop_threshold=baseline_drop_threshold,
    )

    if apply_avgable_filter and "avgable" in df.columns:
        df = df[df["avgable"] == True]  # noqa: E712

    df = df[df["component"].isin(KEEP_COMPONENTS)].reset_index(drop=True)

    if transform is None:
        df = zscore_within_component_shape(df)
    else:
        df = transform(df)

    return df


def _shape_numel(shape_val: object) -> int:
    nums = [int(x) for x in re.findall(r"\d+", str(shape_val))]
    if not nums:
        return 0
    total = 1
    for n in nums:
        total *= n
    return total


def _build_layer_level_stats(
    raw_stats_df: pd.DataFrame,
    model_configs: Dict[str, dict],
    *,
    baseline_drop_threshold: float = 2.0,
) -> pd.DataFrame:
    df = ch.augment_stats_with_profiles(
        raw_stats_df,
        model_configs,
        baseline_drop_threshold=baseline_drop_threshold,
    )

    group_rows = df[(df["kind"] == "group") & (df["component"].isin(["attn", "mlp"]))].copy()
    if group_rows.empty:
        return pd.DataFrame(columns=["model", "layer", "component", "mean", "std", "shape", "avgable", "replaceable"])

    attn_flags = group_rows[group_rows["component"] == "attn"][
        ["model", "layer", "avgable", "replaceable"]
    ].rename(columns={"avgable": "attn_avgable", "replaceable": "attn_replaceable"})
    mlp_flags = group_rows[group_rows["component"] == "mlp"][
        ["model", "layer", "avgable", "replaceable"]
    ].rename(columns={"avgable": "mlp_avgable", "replaceable": "mlp_replaceable"})

    layer_flags = attn_flags.merge(mlp_flags, on=["model", "layer"], how="inner")
    layer_flags = layer_flags[
        (layer_flags["attn_avgable"] == True) & (layer_flags["mlp_avgable"] == True)  # noqa: E712
    ].copy()
    if layer_flags.empty:
        return pd.DataFrame(columns=["model", "layer", "component", "mean", "std", "shape", "avgable", "replaceable"])

    sub_rows = df[(df["kind"] == "subcomponent") & (df["component"].isin(KEEP_COMPONENTS))].copy()
    sub_rows = sub_rows.merge(layer_flags[["model", "layer"]], on=["model", "layer"], how="inner")
    if sub_rows.empty:
        return pd.DataFrame(columns=["model", "layer", "component", "mean", "std", "shape", "avgable", "replaceable"])

    layer_rows: List[Dict[str, object]] = []
    for (model, layer), group in sub_rows.groupby(["model", "layer"]):
        numels = group["shape"].map(_shape_numel)
        total_n = int(numels.sum())
        if total_n <= 0:
            continue

        mean_val = float((group["mean"] * numels).sum() / total_n)
        second_moment = float((((group["std"] ** 2) + (group["mean"] ** 2)) * numels).sum() / total_n)
        std_val = math.sqrt(max(second_moment - (mean_val * mean_val), 0.0))

        flags_row = layer_flags[(layer_flags["model"] == model) & (layer_flags["layer"] == layer)].iloc[0]
        layer_rows.append(
            {
                "model": model,
                "layer": int(layer),
                "component": "layer_full",
                "mean": mean_val,
                "std": std_val,
                "shape": "layer_full",
                "avgable": True,
                "replaceable": bool(flags_row["attn_replaceable"]) and bool(flags_row["mlp_replaceable"]),
            }
        )

    return pd.DataFrame(layer_rows)


def _split_and_order_layer_candidates(
    merge_candidates_initial: List[dict],
) -> Tuple[List[dict], List[dict]]:
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
            if len(aligned_rows) >= 2 and aligned_rows["model"].nunique() >= 2:
                step1_candidates.append(
                    {
                        "component": cand["component"],
                        "cluster_id": cand["cluster_id"],
                        "anchor_layer": anchor_layer,
                        "models": sorted(list(aligned_rows["model"].unique())),
                        "rows": aligned_rows,
                        "indices": aligned_rows.index.tolist(),
                    }
                )
            step2_candidates.append(cand)

    def _step1_sort_key(cand: dict) -> Tuple[float, int]:
        layer = int(cand["anchor_layer"])
        return (abs(layer - MID_LAYER), layer)

    step1_candidates = sorted(step1_candidates, key=_step1_sort_key)

    step1_key_to_index = {
        (cand["component"], int(cand["cluster_id"])): idx for idx, cand in enumerate(step1_candidates)
    }
    step2_in_step1: List[Tuple[int, dict]] = []
    step2_only: List[dict] = []
    for cand in step2_candidates:
        key = (cand["component"], int(cand["cluster_id"]))
        if key in step1_key_to_index:
            step2_in_step1.append((step1_key_to_index[key], cand))
        else:
            step2_only.append(cand)

    step2_in_step1.sort(key=lambda x: x[0])
    step2_only = sorted(step2_only, key=lambda x: (abs(int(x["anchor_layer"]) - MID_LAYER), int(x["anchor_layer"])))
    return step1_candidates, [cand for _, cand in step2_in_step1] + step2_only


def _expand_layer_candidates(layer_candidates: List[dict]) -> List[dict]:
    expanded: List[dict] = []
    for bundle_id, cand in enumerate(layer_candidates):
        base_rows = cand["rows"][["model", "layer", "avgable", "replaceable"]].drop_duplicates().copy()
        for comp in KEEP_COMPONENTS:
            rows = base_rows.copy()
            rows["component"] = comp
            expanded.append(
                {
                    "component": comp,
                    "cluster_id": cand.get("cluster_id", cand.get("cluster2_id", -1)),
                    "anchor_layer": cand.get("anchor_layer", int(base_rows["layer"].iloc[0])),
                    "models": list(cand["models"]),
                    "rows": rows[["model", "layer", "component", "avgable", "replaceable"]].copy(),
                    "indices": list(cand["indices"]),
                    "bundle_id": bundle_id,
                }
            )
    return expanded


def _build_ops_df_from_groups_with_pair_order(
    groups: List[dict],
    *,
    pair_order_offset: int = 0,
) -> Tuple[pd.DataFrame, int]:
    ops_rows: List[dict] = []

    for cand_idx, cand in enumerate(groups):
        rows = cand["rows"].copy()
        if "avgable" not in rows.columns:
            rows["avgable"] = True
        if "replaceable" not in rows.columns:
            rows["replaceable"] = False

        rows_sorted = rows.sort_values("model")
        models_str = ",".join(
            f"{m}:{int(l)}" for m, l in rows_sorted[["model", "layer"]].itertuples(index=False, name=None)
        )
        pair_order = pair_order_offset + int(cand.get("bundle_id", cand_idx))

        for row in rows_sorted.itertuples(index=False):
            source_str = f"{row.model}:{int(row.layer)}"
            target_list = [x for x in models_str.split(",") if x.strip() and x.strip() != source_str]
            ops_rows.append(
                {
                    "model": row.model,
                    "op": "merge" if getattr(row, "avgable", True) or not getattr(row, "replaceable", False) else "replace",
                    "component": row.component,
                    "layer": int(row.layer),
                    "models": models_str,
                    "source": source_str,
                    "targets": ",".join(target_list),
                    "avgable": getattr(row, "avgable", True),
                    "replaceable": getattr(row, "replaceable", False),
                    "pair_order": pair_order,
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
                "pair_order",
            ]
        ), 0

    ops_df = pd.DataFrame(ops_rows)
    bundle_count = len({int(c.get("bundle_id", idx)) for idx, c in enumerate(groups)})
    return ops_df, bundle_count


def _run_ops_pipeline(
    *,
    stats_df: pd.DataFrame,
    out_dir: Path,
    components_for_stage2: Sequence[str],
    candidate_builder: Callable[[pd.DataFrame], Tuple[List[dict], List[dict]]] = None,
    expand_candidates: Callable[[List[dict]], List[dict]] | None = None,
    reorder_for_schedule: bool = True,
) -> Dict[str, object]:
    clustered_df = ch.cluster_stats_df_by_component_target_nmodels(stats_df)
    merge_candidates_initial = build_stage1_candidates(clustered_df)

    if candidate_builder is None:
        step1_candidates, step2_candidates = split_and_order_candidates(merge_candidates_initial)
    else:
        step1_candidates, step2_candidates = candidate_builder(merge_candidates_initial)

    second_stage_candidates_df = build_second_stage_candidates_df(clustered_df, step1_candidates, step2_candidates)

    stage2_merge_candidates: List[dict] = []
    if not second_stage_candidates_df.empty:
        _, stage2_merge_candidates = ch.second_stage_hierarchical_clustering(
            second_stage_candidates_df,
            components=list(components_for_stage2),
            linkage_method="ward",
        )

    if expand_candidates is not None:
        step1_candidates = expand_candidates(step1_candidates)
        stage2_merge_candidates = expand_candidates(stage2_merge_candidates)

    if reorder_for_schedule:
        ops_df_stage1 = ch.build_ops_df_from_groups(step1_candidates)
        if "source" in ops_df_stage1.columns or "targets" in ops_df_stage1.columns:
            ops_df_stage1 = ops_df_stage1.drop(columns=["source", "targets"], errors="ignore")

        if stage2_merge_candidates:
            ops_df_stage2 = ch.build_ops_df_from_groups(stage2_merge_candidates)
            ops_df_stage2 = ops_df_stage2.drop(columns=["source", "targets"], errors="ignore")
        else:
            ops_df_stage2 = pd.DataFrame(columns=["model", "op", "component", "layer", "models"])
    else:
        ops_df_stage1, step1_bundle_count = _build_ops_df_from_groups_with_pair_order(step1_candidates)
        if "source" in ops_df_stage1.columns or "targets" in ops_df_stage1.columns:
            ops_df_stage1 = ops_df_stage1.drop(columns=["source", "targets"], errors="ignore")

        if stage2_merge_candidates:
            ops_df_stage2, _ = _build_ops_df_from_groups_with_pair_order(
                stage2_merge_candidates,
                pair_order_offset=step1_bundle_count,
            )
            ops_df_stage2 = ops_df_stage2.drop(columns=["source", "targets"], errors="ignore")
        else:
            ops_df_stage2 = pd.DataFrame(columns=["model", "op", "component", "layer", "models", "pair_order"])
    
    if reorder_for_schedule:
        component_to_group = {c: "attn" for c in ["attn", *ATTN_SUBCOMPS]}
        component_to_group.update({c: "mlp" for c in ["mlp", *MLP_SUBCOMPS]})

        ops_df_stage1 = ch.reorder_ops_log_for_schedule(ops_df_stage1, component_to_group=component_to_group)
        if not ops_df_stage2.empty:
            if not ops_df_stage1.empty:
                s1_comp_layer_order = ops_df_stage1[["component", "layer"]].drop_duplicates().copy()
                s1_comp_layer_order["s1_order"] = range(len(s1_comp_layer_order))

                ops_df_stage2_merged = ops_df_stage2.merge(
                    s1_comp_layer_order, on=["component", "layer"], how="left"
                )
                s2_seen_in_s1 = (
                    ops_df_stage2_merged[ops_df_stage2_merged["s1_order"].notna()]
                    .sort_values("s1_order")
                    .drop(columns="s1_order")
                )
                s2_new = ops_df_stage2_merged[ops_df_stage2_merged["s1_order"].isna()].drop(columns="s1_order")
                s2_new_scheduled = ch.reorder_ops_log_for_schedule(s2_new, component_to_group=component_to_group)
                ops_df_stage2 = pd.concat([s2_seen_in_s1, s2_new_scheduled], ignore_index=True)
            else:
                ops_df_stage2 = ch.reorder_ops_log_for_schedule(ops_df_stage2, component_to_group=component_to_group)

        ops_df_stage1, ops_df_stage2 = schedule_pair_order(ops_df_stage1, ops_df_stage2)
    else:
        ops_df_stage1 = ops_df_stage1.sort_values(["pair_order", "model"]).reset_index(drop=True)
        if not ops_df_stage2.empty:
            ops_df_stage2 = ops_df_stage2.sort_values(["pair_order", "model"]).reset_index(drop=True)

    write_per_model_csvs_to_dir(ops_df_stage1, "step1", out_dir)
    write_per_model_csvs_to_dir(ops_df_stage2, "step2", out_dir)

    summary = {
        "prepared_rows": len(stats_df),
        "clustered_rows": len(clustered_df),
        "initial_candidates": len(merge_candidates_initial),
        "step1_candidates": len(step1_candidates),
        "step2_input_candidates": len(step2_candidates),
        "stage2_candidates": len(stage2_merge_candidates),
        "stage1_ops_rows": len(ops_df_stage1),
        "stage2_ops_rows": len(ops_df_stage2),
    }
    _write_summary(out_dir / "summary.txt", summary)
    return summary


def compute_or_load_raw_stats(
    *,
    model_paths: List[str],
    model_labels: List[str],
    device: str,
    stats_csv: Path,
) -> pd.DataFrame:
    stats_csv.parent.mkdir(parents=True, exist_ok=True)
    if stats_csv.exists():
        print(f"Loading cached raw stats from {stats_csv}")
        return pd.read_csv(stats_csv)

    raw_stats_df = ch.compute_stats_for_models(
        model_paths=model_paths,
        model_labels=model_labels,
        device=device,
    )
    raw_stats_df.to_csv(stats_csv, index=False)
    print(f"Wrote raw stats cache to {stats_csv}")
    return raw_stats_df


def _run_standardized_case(
    *,
    root: Path,
    case_name: str,
    labels: Sequence[str],
    device: str,
    output_root: Path,
    artifacts_root: Path,
    baseline_drop_threshold: float,
) -> Dict[str, object]:
    model_paths, model_labels, model_configs = _get_registry_for_labels(root, labels)
    stats_csv = artifacts_root / f"{case_name}_raw_stats.csv"
    raw_stats_df = compute_or_load_raw_stats(
        model_paths=model_paths,
        model_labels=model_labels,
        device=device,
        stats_csv=stats_csv,
    )

    out_dir = output_root / case_name
    prepared_df = _prepare_subcomponent_stats(
        raw_stats_df,
        model_configs,
        transform=_standardize_by_model,
        apply_avgable_filter=True,
        baseline_drop_threshold=baseline_drop_threshold,
    )
    _save_prepared_stats(prepared_df, out_dir, "prepared_stats.csv")
    summary = _run_ops_pipeline(
        stats_df=prepared_df,
        out_dir=out_dir,
        components_for_stage2=KEEP_COMPONENTS,
    )
    _write_summary(
        out_dir / "run_config.txt",
        {
            "case": case_name,
            "models": ",".join(model_labels),
            "transform": "standardize_by_model",
            "baseline_drop_threshold": baseline_drop_threshold,
        },
    )
    return summary


def _run_no_avgable_filter_standardized_case(
    *,
    root: Path,
    case_name: str,
    labels: Sequence[str],
    device: str,
    output_root: Path,
    artifacts_root: Path,
    baseline_drop_threshold: float,
) -> Dict[str, object]:
    model_paths, model_labels, model_configs = _get_registry_for_labels(root, labels)
    stats_csv = artifacts_root / f"{case_name}_raw_stats.csv"
    raw_stats_df = compute_or_load_raw_stats(
        model_paths=model_paths,
        model_labels=model_labels,
        device=device,
        stats_csv=stats_csv,
    )

    out_dir = output_root / case_name
    prepared_df = _prepare_subcomponent_stats(
        raw_stats_df,
        model_configs,
        transform=_standardize_by_model,
        apply_avgable_filter=False,
        baseline_drop_threshold=baseline_drop_threshold,
    )
    _save_prepared_stats(prepared_df, out_dir, "prepared_stats.csv")
    summary = _run_ops_pipeline(
        stats_df=prepared_df,
        out_dir=out_dir,
        components_for_stage2=KEEP_COMPONENTS,
    )
    _write_summary(
        out_dir / "run_config.txt",
        {
            "case": case_name,
            "models": ",".join(model_labels),
            "transform": "standardize_by_model_no_avgable_filter",
            "baseline_drop_threshold": baseline_drop_threshold,
        },
    )
    return summary


def _split_layers_into_partitions(layers: Sequence[int], num_partitions: int = 3) -> List[List[int]]:
    ordered = sorted(int(layer) for layer in layers)
    if not ordered or num_partitions <= 0:
        return []

    n = len(ordered)
    base = n // num_partitions
    remainder = n % num_partitions

    partitions: List[List[int]] = []
    start = 0
    for idx in range(num_partitions):
        size = base + (1 if idx < remainder else 0)
        end = start + size
        partitions.append(ordered[start:end])
        start = end
    return partitions


def _select_top_profiler_fraction(
    df: pd.DataFrame,
    *,
    score_column: str = "score_avg",
    fraction: float = 1.0 / 3.0,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if score_column not in df.columns:
        raise ValueError(f"Required profiler column '{score_column}' not found.")
    if fraction <= 0:
        raise ValueError("fraction must be > 0")

    selected_groups: List[pd.DataFrame] = []
    for (_, _), group in df.groupby(["model", "component"], sort=False):
        ordered = group.sort_values([score_column, "layer"], ascending=[False, True]).copy()
        keep_n = max(1, math.ceil(len(ordered) * fraction))
        selected_groups.append(ordered.head(keep_n))
    if not selected_groups:
        return df.iloc[0:0].copy()
    return pd.concat(selected_groups, ignore_index=True)


def _write_partitioned_ops_dir(
    *,
    src_dir: Path,
    dst_dir: Path,
    allowed_layers: Set[int],
) -> Dict[str, int]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    written_counts: Dict[str, int] = {}

    for csv_path in src_dir.glob("ops_step*.csv"):
        df = pd.read_csv(csv_path)
        if "layer" in df.columns:
            filtered = df[df["layer"].astype(int).isin(sorted(allowed_layers))].copy()
        else:
            filtered = df.copy()
        filtered.to_csv(dst_dir / csv_path.name, index=False)
        written_counts[csv_path.name] = len(filtered)

    return written_counts


def _run_partitioned_no_avgable_filter_standardized_case(
    *,
    root: Path,
    case_name: str,
    labels: Sequence[str],
    device: str,
    output_root: Path,
    artifacts_root: Path,
    baseline_drop_threshold: float,
    partition_index: int,
) -> Dict[str, object]:
    model_paths, model_labels, model_configs = _get_registry_for_labels(root, labels)
    stats_csv = artifacts_root / "6d_deepseek_noprofiling_raw_stats.csv"
    raw_stats_df = compute_or_load_raw_stats(
        model_paths=model_paths,
        model_labels=model_labels,
        device=device,
        stats_csv=stats_csv,
    )

    prepared_full_df = _prepare_subcomponent_stats(
        raw_stats_df,
        model_configs,
        transform=_standardize_by_model,
        apply_avgable_filter=False,
        baseline_drop_threshold=baseline_drop_threshold,
    )
    layer_partitions = _split_layers_into_partitions(sorted(prepared_full_df["layer"].astype(int).unique()), 3)
    if partition_index < 0 or partition_index >= len(layer_partitions):
        raise ValueError(f"Invalid partition_index={partition_index} for case {case_name}")

    allowed_layers = set(layer_partitions[partition_index])
    source_dir = artifacts_root / "6d_deepseek_noprofiling_full_ops"
    _run_ops_pipeline(
        stats_df=prepared_full_df,
        out_dir=source_dir,
        components_for_stage2=KEEP_COMPONENTS,
    )

    out_dir = output_root / case_name
    filtered_prepared_df = prepared_full_df[prepared_full_df["layer"].astype(int).isin(sorted(allowed_layers))].copy()
    _save_prepared_stats(filtered_prepared_df, out_dir, "prepared_stats.csv")
    written_counts = _write_partitioned_ops_dir(src_dir=source_dir, dst_dir=out_dir, allowed_layers=allowed_layers)
    summary = {
        "prepared_rows": len(filtered_prepared_df),
        "layer_partition_index": partition_index,
        "layer_partition_layers": ",".join(str(layer) for layer in sorted(allowed_layers)),
        "stage1_ops_rows": sum(count for name, count in written_counts.items() if "ops_step1_" in name),
        "stage2_ops_rows": sum(count for name, count in written_counts.items() if "ops_step2_" in name),
    }
    _write_summary(out_dir / "summary.txt", summary)
    _write_summary(
        out_dir / "run_config.txt",
        {
            "case": case_name,
            "models": ",".join(model_labels),
            "transform": "standardize_by_model_no_avgable_filter_partitioned_from_full_ops",
            "baseline_drop_threshold": baseline_drop_threshold,
            "layer_partition_index": partition_index,
            "layer_partition_layers": ",".join(str(layer) for layer in sorted(allowed_layers)),
        },
    )
    return summary


def _run_top_profiler_standardized_case(
    *,
    root: Path,
    case_name: str,
    labels: Sequence[str],
    device: str,
    output_root: Path,
    artifacts_root: Path,
    baseline_drop_threshold: float,
    profiler_fraction: float = 1.0 / 3.0,
) -> Dict[str, object]:
    model_paths, model_labels, model_configs = _get_registry_for_labels(root, labels)
    stats_csv = artifacts_root / "6d_deepseek_noprofiling_raw_stats.csv"
    raw_stats_df = compute_or_load_raw_stats(
        model_paths=model_paths,
        model_labels=model_labels,
        device=device,
        stats_csv=stats_csv,
    )

    out_dir = output_root / case_name
    prepared_full_df = _prepare_subcomponent_stats(
        raw_stats_df,
        model_configs,
        transform=_standardize_by_model,
        apply_avgable_filter=False,
        baseline_drop_threshold=baseline_drop_threshold,
    )
    prepared_df = _select_top_profiler_fraction(
        prepared_full_df,
        score_column="score_avg",
        fraction=profiler_fraction,
    )
    _save_prepared_stats(prepared_df, out_dir, "prepared_stats.csv")
    summary = _run_ops_pipeline(
        stats_df=prepared_df,
        out_dir=out_dir,
        components_for_stage2=KEEP_COMPONENTS,
    )
    _write_summary(
        out_dir / "run_config.txt",
        {
            "case": case_name,
            "models": ",".join(model_labels),
            "transform": "standardize_by_model_top_profiler_fraction",
            "baseline_drop_threshold": baseline_drop_threshold,
            "profiler_score_column": "score_avg",
            "profiler_fraction": profiler_fraction,
        },
    )
    return summary


def _run_layer_level_standardized_case(
    *,
    root: Path,
    case_name: str,
    labels: Sequence[str],
    device: str,
    output_root: Path,
    artifacts_root: Path,
    baseline_drop_threshold: float,
) -> Dict[str, object]:
    model_paths, model_labels, model_configs = _get_registry_for_labels(root, labels)
    stats_csv = artifacts_root / f"{case_name}_raw_stats.csv"
    raw_stats_df = compute_or_load_raw_stats(
        model_paths=model_paths,
        model_labels=model_labels,
        device=device,
        stats_csv=stats_csv,
    )

    out_dir = output_root / case_name
    prepared_df = _standardize_by_model(
        _build_layer_level_stats(
            raw_stats_df,
            model_configs,
            baseline_drop_threshold=baseline_drop_threshold,
        )
    )
    _save_prepared_stats(prepared_df, out_dir, "prepared_stats.csv")
    summary = _run_ops_pipeline(
        stats_df=prepared_df,
        out_dir=out_dir,
        components_for_stage2=["layer_full"],
        candidate_builder=_split_and_order_layer_candidates,
        expand_candidates=_expand_layer_candidates,
        reorder_for_schedule=False,
    )
    _write_summary(
        out_dir / "run_config.txt",
        {
            "case": case_name,
            "models": ",".join(model_labels),
            "transform": "layer_level_joint_standardized",
            "baseline_drop_threshold": baseline_drop_threshold,
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run four llama clustering candidate approaches.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--stats-csv", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=sorted(
            [*STANDARDIZED_CASE_SPECS.keys(), *LAYER_LEVEL_STANDARDIZED_CASE_SPECS.keys()]
        ),
        default=None,
        help="Run only the requested standardized candidate cases.",
    )
    parser.add_argument(
        "--baseline-drop-threshold",
        type=float,
        default=2.0,
        help="Mark a row avgable if its noise-profile score is at least baseline minus this value.",
    )
    args = parser.parse_args()

    root = _repo_root()
    output_root = Path(args.output_root) if args.output_root else _candidates_root(root)
    artifacts_root = _artifacts_root(root)
    output_root.mkdir(parents=True, exist_ok=True)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    if args.cases:
        for case_name in args.cases:
            if case_name == "6d_deepseek_noprofiling":
                _run_no_avgable_filter_standardized_case(
                    root=root,
                    case_name=case_name,
                    labels=STANDARDIZED_CASE_SPECS[case_name],
                    device=args.device,
                    output_root=output_root,
                    artifacts_root=artifacts_root,
                    baseline_drop_threshold=args.baseline_drop_threshold,
                )
            elif case_name in {
                "6pa_deepseek_noprofiling",
                "6pb_deepseek_noprofiling",
                "6pc_deepseek_noprofiling",
            }:
                partition_index = {
                    "6pa_deepseek_noprofiling": 0,
                    "6pb_deepseek_noprofiling": 1,
                    "6pc_deepseek_noprofiling": 2,
                }[case_name]
                _run_partitioned_no_avgable_filter_standardized_case(
                    root=root,
                    case_name=case_name,
                    labels=STANDARDIZED_CASE_SPECS[case_name],
                    device=args.device,
                    output_root=output_root,
                    artifacts_root=artifacts_root,
                    baseline_drop_threshold=args.baseline_drop_threshold,
                    partition_index=partition_index,
                )
            elif case_name == "6profiler":
                _run_top_profiler_standardized_case(
                    root=root,
                    case_name=case_name,
                    labels=STANDARDIZED_CASE_SPECS[case_name],
                    device=args.device,
                    output_root=output_root,
                    artifacts_root=artifacts_root,
                    baseline_drop_threshold=args.baseline_drop_threshold,
                    profiler_fraction=1.0 / 3.0,
                )
            elif case_name in STANDARDIZED_CASE_SPECS:
                _run_standardized_case(
                    root=root,
                    case_name=case_name,
                    labels=STANDARDIZED_CASE_SPECS[case_name],
                    device=args.device,
                    output_root=output_root,
                    artifacts_root=artifacts_root,
                    baseline_drop_threshold=args.baseline_drop_threshold,
                )
            else:
                _run_layer_level_standardized_case(
                    root=root,
                    case_name=case_name,
                    labels=LAYER_LEVEL_STANDARDIZED_CASE_SPECS[case_name],
                    device=args.device,
                    output_root=output_root,
                    artifacts_root=artifacts_root,
                    baseline_drop_threshold=args.baseline_drop_threshold,
                )
        return

    model_paths, model_labels, model_configs = _get_llama_registry(root)
    stats_csv = Path(args.stats_csv) if args.stats_csv else artifacts_root / "llama_raw_stats.csv"
    raw_stats_df = compute_or_load_raw_stats(
        model_paths=model_paths,
        model_labels=model_labels,
        device=args.device,
        stats_csv=stats_csv,
    )

    approach2_dir = output_root / "2_standardized"
    prepared_2 = _prepare_subcomponent_stats(
        raw_stats_df,
        model_configs,
        transform=_standardize_by_model,
        apply_avgable_filter=True,
        baseline_drop_threshold=args.baseline_drop_threshold,
    )
    _save_prepared_stats(prepared_2, approach2_dir, "prepared_stats.csv")
    _run_ops_pipeline(
        stats_df=prepared_2,
        out_dir=approach2_dir,
        components_for_stage2=KEEP_COMPONENTS,
    )

    approach3_dir = output_root / "3_standardized_normalized"
    prepared_3 = _prepare_subcomponent_stats(
        raw_stats_df,
        model_configs,
        transform=lambda df: _normalize_by_model(_standardize_by_model(df)),
        apply_avgable_filter=True,
        baseline_drop_threshold=args.baseline_drop_threshold,
    )
    _save_prepared_stats(prepared_3, approach3_dir, "prepared_stats.csv")
    _run_ops_pipeline(
        stats_df=prepared_3,
        out_dir=approach3_dir,
        components_for_stage2=KEEP_COMPONENTS,
    )

    approach5_dir = output_root / "5_standardized_ignore_noise"
    prepared_5 = _prepare_subcomponent_stats(
        raw_stats_df,
        model_configs,
        transform=_standardize_by_model,
        apply_avgable_filter=False,
        baseline_drop_threshold=args.baseline_drop_threshold,
    )
    _save_prepared_stats(prepared_5, approach5_dir, "prepared_stats.csv")
    _run_ops_pipeline(
        stats_df=prepared_5,
        out_dir=approach5_dir,
        components_for_stage2=KEEP_COMPONENTS,
    )

    approach4_dir = output_root / "4_layer_level_joint"
    prepared_4 = _build_layer_level_stats(
        raw_stats_df,
        model_configs,
        baseline_drop_threshold=args.baseline_drop_threshold,
    )
    _save_prepared_stats(prepared_4, approach4_dir, "prepared_stats.csv")
    _run_ops_pipeline(
        stats_df=prepared_4,
        out_dir=approach4_dir,
        components_for_stage2=["layer_full"],
        candidate_builder=_split_and_order_layer_candidates,
        expand_candidates=_expand_layer_candidates,
        reorder_for_schedule=False,
    )

    approach4c_dir = output_root / "4c_layer_level_joint_standardized"
    prepared_4c = _standardize_by_model(
        _build_layer_level_stats(
            raw_stats_df,
            model_configs,
            baseline_drop_threshold=args.baseline_drop_threshold,
        )
    )
    _save_prepared_stats(prepared_4c, approach4c_dir, "prepared_stats.csv")
    _run_ops_pipeline(
        stats_df=prepared_4c,
        out_dir=approach4c_dir,
        components_for_stage2=["layer_full"],
        candidate_builder=_split_and_order_layer_candidates,
        expand_candidates=_expand_layer_candidates,
        reorder_for_schedule=False,
    )

    approach4b_dir = output_root / "4b_layer_level_joint_standardized_normalized"
    prepared_4b = _normalize_by_model(
        _standardize_by_model(
            _build_layer_level_stats(
                raw_stats_df,
                model_configs,
                baseline_drop_threshold=args.baseline_drop_threshold,
            )
        )
    )
    _save_prepared_stats(prepared_4b, approach4b_dir, "prepared_stats.csv")
    _run_ops_pipeline(
        stats_df=prepared_4b,
        out_dir=approach4b_dir,
        components_for_stage2=["layer_full"],
        candidate_builder=_split_and_order_layer_candidates,
        expand_candidates=_expand_layer_candidates,
        reorder_for_schedule=False,
    )


if __name__ == "__main__":
    main()
