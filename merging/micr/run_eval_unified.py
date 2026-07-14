#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified runner for the micr evaluation pipelines.

This script dispatches to one of the existing implementations:
- normal weight-merge:     `micr/run_eval.py`
- large-model (32B) merge: `micr/run_eval_32b.py`  (thin shim over run_eval.py)
- LoRA adapter merge:      `micr/run_eval_lora.py`

The goal is a single CLI with a `--mode` switch.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional


def _ensure_project_root_on_path() -> None:
    # Register the in-repo merge_tools package keyed to this file's location
    # (dirname-independent; sys.modules wins over any same-named sibling on path).
    import types as _types
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if "merge_tools" not in sys.modules:
        pkg = _types.ModuleType("merge_tools")
        pkg.__path__ = [root]
        sys.modules["merge_tools"] = pkg


def _import_sibling(py_filename: str, module_name: str):
    """
    Import a sibling python file (in the same directory as this script)
    as a module without requiring `micr/` to be a package.
    """
    here = Path(__file__).resolve().parent
    mod_path = here / py_filename
    if not mod_path.exists():
        raise FileNotFoundError(f"Expected sibling file not found: {mod_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(mod_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for: {mod_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_mode(mode: str) -> str:
    m = str(mode or "").strip().lower()
    if m in {"normal", "base", "default", "weights"}:
        return "normal"
    if m in {"32b", "large", "big", "streaming", "large_model"}:
        return "32b"
    if m in {"lora", "adapter", "adapters"}:
        return "lora"
    if m in {"quantized", "fp8", "compressed"}:
        return "quantized"
    raise ValueError(
        f"Unknown mode '{mode}'. Use one of: normal | 32b (large) | lora | quantized"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified runner for micr merge/eval pipelines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        required=True,
        help="Which pipeline to run: normal | 32b (large) | lora | quantized.",
    )

    csv_group = parser.add_mutually_exclusive_group(required=True)
    csv_group.add_argument(
        "--ops_csv",
        help="Path to operations CSV (legacy single file mode).",
        default=None,
    )
    csv_group.add_argument(
        "--ops_step_csvs_dir",
        help="Directory containing per-model CSV files (ops_step1_*.csv, ops_step2_*.csv).",
        default=None,
    )

    parser.add_argument("--target_label", required=True, help="Target model label to modify/evaluate.")
    parser.add_argument("--domain", required=True, help="Domain or registry task key for evaluation.")
    parser.add_argument("--gpu_ids", default="0", help="CUDA_VISIBLE_DEVICES value for evaluation.")
    parser.add_argument("--no_eval", action="store_true", help="Disable evaluation (accept all steps).")

    # Common-ish knobs (some are ignored depending on mode).
    parser.add_argument("--label_map_json", default=None, help="JSON mapping label -> model path (required for normal/32b).")
    parser.add_argument("--working_root", default=None, help="Root directory for working copies (mode-dependent default).")
    parser.add_argument("--results_csv", default=None, help="Per-step results CSV (mode-dependent default).")
    parser.add_argument("--output_dir", default=None, help="Evaluation outputs directory.")
    parser.add_argument("--timeout_minutes", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--drop_tolerance", type=float, default=None, help="Allowed absolute drop (percentage points) before rejecting a step.")

    # Weight-merge scheduling / filtering.
    parser.add_argument("--sorted_ops_out", default=None, help="Where to export sorted ops CSV (kept for compatibility).")
    parser.add_argument(
        "--sort_mode",
        choices=["normal", "separate", "together"],
        default=None,
        help="Ordering policy for ops in normal/32b pipelines.",
    )
    parser.add_argument(
        "--ignore-other-families",
        dest="ignore_other_families",
        action="store_true",
        help="Ignore donors from other model families (normal/32b only).",
    )

    # Only implemented in normal `run_eval.py`
    parser.add_argument("--initial_baseline", type=float, default=None, help="Force an initial baseline score (normal only).")
    parser.add_argument("--force-calc-baseline", action="store_true", help="Force baseline eval even if hardcoded (normal only).")
    parser.add_argument("--enable-scaling", action="store_true", help="Enable std scaling (normal only; implementation-dependent).")
    parser.add_argument("--merge_device", choices=["cuda", "cpu", "auto"], default="auto",
                        help="Device for the merge arithmetic (auto = cpu unless scaling).")

    return parser


def main() -> None:
    _ensure_project_root_on_path()

    parser = _build_parser()
    args = parser.parse_args()

    mode = _normalize_mode(args.mode)
    eval_enabled = not bool(args.no_eval)

    # Mode-dependent defaults (match the underlying scripts as closely as possible).
    working_root: str
    results_csv: str
    output_dir: str = args.output_dir or "./evaluation_results"
    timeout_minutes: int = int(args.timeout_minutes) if args.timeout_minutes is not None else 15
    temperature: float = float(args.temperature) if args.temperature is not None else 0.0
    drop_tolerance: float = float(args.drop_tolerance) if args.drop_tolerance is not None else 2.0
    sorted_ops_out: str = args.sorted_ops_out or "./sorted_ops.csv"
    sort_mode: str = args.sort_mode or "normal"

    if mode in {"normal", "32b", "quantized"}:
        working_root = args.working_root or os.environ.get("MICR_WORKING_ROOT", "/tmp/micr_merged_models")
        results_csv = args.results_csv or "./ops_step_csvs/target_steps.csv"
        batch_size = int(args.batch_size) if args.batch_size is not None else 64

        if not args.label_map_json:
            parser.error(f"--label_map_json is required for --mode {mode}")

        if mode == "normal":
            impl = _import_sibling("run_eval.py", "micr_run_eval")
            impl.run_single_target_pipeline(
                ops_csv=args.ops_csv,
                ops_step_csvs_dir=args.ops_step_csvs_dir,
                sorted_ops_out=sorted_ops_out,
                label_map_json=args.label_map_json,
                target_label=args.target_label,
                target_domain_or_task=args.domain,
                working_root=working_root,
                results_csv=results_csv,
                output_dir=output_dir,
                gpu_ids=args.gpu_ids,
                timeout_minutes=timeout_minutes,
                batch_size=batch_size,
                temperature=temperature,
                drop_tolerance=drop_tolerance,
                eval_enabled=eval_enabled,
                sort_mode=sort_mode,
                ignore_other_families=bool(args.ignore_other_families),
                initial_baseline=args.initial_baseline,
                force_calc_baseline=bool(args.force_calc_baseline),
                enable_scaling=bool(args.enable_scaling),
                merge_device=args.merge_device,
            )
        elif mode == "quantized":
            impl = _import_sibling("run_eval_quantized.py", "micr_run_eval_quantized")
            impl.run_single_target_pipeline(
                ops_csv=args.ops_csv,
                ops_step_csvs_dir=args.ops_step_csvs_dir,
                sorted_ops_out=sorted_ops_out,
                label_map_json=args.label_map_json,
                target_label=args.target_label,
                target_domain_or_task=args.domain,
                working_root=working_root,
                results_csv=results_csv,
                output_dir=output_dir,
                gpu_ids=args.gpu_ids,
                timeout_minutes=timeout_minutes,
                batch_size=batch_size,
                temperature=temperature,
                drop_tolerance=drop_tolerance,
                eval_enabled=eval_enabled,
                sort_mode=sort_mode,
                ignore_other_families=bool(args.ignore_other_families),
                initial_baseline=args.initial_baseline,
                force_calc_baseline=bool(args.force_calc_baseline),
                enable_scaling=bool(args.enable_scaling),
                merge_device=args.merge_device,
            )
        else:
            impl = _import_sibling("run_eval_32b.py", "micr_run_eval_32b")
            impl.run_single_target_pipeline(
                ops_csv=args.ops_csv,
                ops_step_csvs_dir=args.ops_step_csvs_dir,
                sorted_ops_out=sorted_ops_out,
                label_map_json=args.label_map_json,
                target_label=args.target_label,
                target_domain_or_task=args.domain,
                working_root=working_root,
                results_csv=results_csv,
                output_dir=output_dir,
                gpu_ids=args.gpu_ids,
                timeout_minutes=timeout_minutes,
                batch_size=batch_size,
                temperature=temperature,
                drop_tolerance=drop_tolerance,
                eval_enabled=eval_enabled,
                sort_mode=sort_mode,
                ignore_other_families=bool(args.ignore_other_families),
            )
        return

    if mode == "lora":
        # LoRA defaults match `run_eval_lora.py` CLI defaults.
        working_root = args.working_root or "./lora_work"
        results_csv = args.results_csv or "lora_results.csv"
        batch_size = int(args.batch_size) if args.batch_size is not None else 32

        impl = _import_sibling("run_eval_lora.py", "micr_run_eval_lora")
        impl.run_lora_pipeline(
            ops_csv=args.ops_csv,
            ops_step_csvs_dir=args.ops_step_csvs_dir,
            target_label=args.target_label,
            domain=args.domain,
            gpu_ids=args.gpu_ids,
            eval_enabled=eval_enabled,
            working_root=working_root,
            results_csv=results_csv,
            timeout_minutes=timeout_minutes,
            batch_size=batch_size,
            temperature=temperature,
            output_dir=output_dir,
            drop_tolerance=drop_tolerance,
        )
        return

    raise RuntimeError(f"Unhandled mode: {mode}")


if __name__ == "__main__":
    main()

