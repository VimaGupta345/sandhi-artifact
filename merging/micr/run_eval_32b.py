#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thin delegating shim over ``micr/run_eval.py`` -- the former large-model (32B)
merge-and-evaluate runner.

``run_eval.py`` is now the single implementation of the weight-merge pipeline.
This module keeps the historical ``run_eval_32b`` surface alive:

* **CLI**: the argparse surface below is exactly what this script always
  exposed (same flags, same defaults: ``--batch_size`` is an int defaulting to
  64, ``--eval_split`` defaults to ``full``). It forwards into
  ``run_eval.run_single_target_pipeline``.
* **Module API**: every module-level name that other code and the test suites
  import from here is re-exported (``evaluate_model_for_task``,
  ``evaluate_with_retry``, ``evaluate_baseline_blocking``,
  ``create_unified_eval_context``, ``TASK_REGISTRY``,
  ``DOMAIN_TO_REGISTRY_TASK``, ``run_single_target_pipeline``, the component
  maps, ...), so existing imports keep working unchanged.

What changed for 32B models
---------------------------
32B models simply run through ``run_eval`` now:

* The streaming donor-average path is gone. ``run_eval`` loads contributors via
  ``top_k_experiment._load_model`` (``low_cpu_mem_usage`` mmap), which keeps
  large models tractable, and the arithmetic is the same N-way fp32 average
  (provably identical math for the unscaled merge).
* The unconditional per-component std rescale (``get_model_component_stds`` +
  ``target_std``) was retired with it: merges are plain elementwise averages,
  so ``--merge_device auto`` now resolves to **cpu** (bitwise-identical to
  cuda and faster; see ``micr/merge_device.py``). Row-wise affine moment
  matching remains available through ``run_eval``'s ``--enable-scaling``.
* The ``generation_config`` save fix-up was ported into
  ``run_eval._save_model_to_temp_dir_for_eval``.
* vLLM tensor-parallel sizing needs no port: ``micr/eval_harness.py`` already
  derives it as ``len(gpu_ids.split(','))``; the old
  ``eval_settings["tensor_parallel_size"]`` key was never read.
* Evaluation blocks stay per contiguous (stage, layer, attn/mlp) group: the
  delegation pins ``bundle_eval_mode="group"``, matching the old 32B walker.
  ``run_eval`` additionally gains delta-shard candidate saves and per-step
  crash-safe accept swaps, which the old 32B runner lacked.
"""

import argparse
import builtins
import os
import sys
import types
from typing import Dict, List, Optional

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import types as _types
if "merge_tools" not in sys.modules:
    _mt_pkg = _types.ModuleType("merge_tools")
    _mt_pkg.__path__ = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
    sys.modules["merge_tools"] = _mt_pkg

from merge_tools.micr import run_eval as _run_eval  # type: ignore

# Names re-exported for compatibility with the old module surface. The shared
# helpers live in run_eval.py; nothing here redefines behavior.
from merge_tools.micr.run_eval import (  # type: ignore  # noqa: F401
    ATTN_COMPONENT_ORDER,
    ATTN_COMPONENT_TO_ATTR,
    COMPONENT_TO_GROUP,
    MLP_COMPONENT_ORDER,
    MLP_COMPONENT_TO_ATTR,
    TASKS_REQUIRING_INT_BATCH_SIZE,
    Path,
    _append_csv_row,
    _fallback_is_redundant,
    _n_way_average_subcomponent_into_target,
    create_unified_eval_context,
    infer_family_prefix,
    load_ops_csvs_for_target,
    parse_participants_with_layers,
    reorder_ops_log_for_schedule,
    task_prefers_vllm,
)
from merge_tools.micr.eval_harness import EnvironmentManager, TASK_REGISTRY  # type: ignore  # noqa: F401
# run_eval.evaluate_model_for_task reads the optional registry eval cap via
# the `eval_harness` module global; the rebound clone (see
# _rebind_into_this_module) resolves that name in THIS module.
from merge_tools.micr import eval_harness  # type: ignore  # noqa: F401
from merge_tools.micr.eval_retry import (  # type: ignore
    evaluate_baseline_blocking as _evaluate_baseline_blocking,
    evaluate_with_retry as _evaluate_with_retry,
)
# run_eval.evaluate_model_for_task now consults `persistent_eval` (optional
# run-persistent vLLM engine). The rebound clone below resolves globals in THIS
# module, so the name must exist here too or _rebind_into_this_module raises at
# import time. It may be None (import failure disables the fast path).
from merge_tools.micr.run_eval import persistent_eval  # type: ignore  # noqa: F401
# Same rebind requirement for the optional in-process HF evaluator (see the
# persistent_eval note above): evaluate_model_for_task's clone resolves the
# `inprocess_eval` global in THIS module. May be None (import failure simply
# disables the fast path).
from merge_tools.micr.run_eval import inprocess_eval  # type: ignore  # noqa: F401

# The 32B runner recognized a few extra --domain aliases on top of run_eval's
# map. Only "ifeval" is a real redirect; the identity entries are kept for
# documentation (DOMAIN_TO_REGISTRY_TASK.get(x, x) would resolve them anyway).
DOMAIN_TO_REGISTRY_TASK: Dict[str, str] = dict(_run_eval.DOMAIN_TO_REGISTRY_TASK)
DOMAIN_TO_REGISTRY_TASK.update(
    {
        "ifeval": "tinyMMLU",
        "medqa_4options": "medqa_4options",
        "m_mmlu_ru": "m_mmlu_ru",
    }
)


def _rebind_into_this_module(func):
    """
    Clone ``func`` so its global lookups resolve in *this* module.

    The test suites patch ``run_eval_32b.TASK_REGISTRY`` and
    ``run_eval_32b.create_unified_eval_context`` and expect
    ``run_eval_32b.evaluate_model_for_task`` to honor the patches. A plain
    re-export would keep reading ``run_eval``'s globals, so instead we rebind
    the one shared code object onto this module's namespace: the body (and
    therefore the behavior, ``inspect.getsource``, signature and annotations)
    stays run_eval.py's single implementation.
    """
    clone = types.FunctionType(
        func.__code__, globals(), func.__name__, func.__defaults__, func.__closure__
    )
    clone.__kwdefaults__ = None if func.__kwdefaults__ is None else dict(func.__kwdefaults__)
    clone.__annotations__ = dict(func.__annotations__)
    clone.__doc__ = func.__doc__
    clone.__module__ = __name__
    # Fail at import time (not mid-run) if run_eval grows a global this module
    # does not provide. co_names also lists attribute names, so only check the
    # ones neither built in nor defined here against run_eval's namespace.
    for name in func.__code__.co_names:
        if name in globals() or hasattr(builtins, name):
            continue
        if hasattr(_run_eval, name):
            raise RuntimeError(
                f"run_eval.{func.__name__} uses global '{name}' which run_eval_32b "
                f"does not re-export; add it to the imports above."
            )
    return clone


# Single implementation lives in run_eval.py; this copy resolves TASK_REGISTRY,
# create_unified_eval_context and DOMAIN_TO_REGISTRY_TASK in this module.
evaluate_model_for_task = _rebind_into_this_module(_run_eval.evaluate_model_for_task)


# The retry policy lives in micr/eval_retry.py so all runners share one
# definition of what a failed evaluation means. These wrappers look up
# ``evaluate_model_for_task`` at call time so tests can patch it per-module.
def evaluate_with_retry(label: str, **kwargs) -> Optional[float]:
    return _evaluate_with_retry(evaluate_model_for_task, label, **kwargs)


def evaluate_baseline_blocking(label: str, **kwargs) -> float:
    return _evaluate_baseline_blocking(evaluate_model_for_task, label, **kwargs)


def run_single_target_pipeline(
    *,
    ops_csv: Optional[str] = None,
    ops_step_csvs_dir: Optional[str] = None,
    sorted_ops_out: Optional[str],
    label_map_json: str,
    target_label: str,
    target_domain_or_task: str,
    working_root: str,
    results_csv: str,
    output_dir: str,
    gpu_ids: str,
    timeout_minutes: int,
    batch_size: int,
    temperature: float,
    drop_tolerance: float,
    eval_enabled: bool,
    sort_mode: str,
    ignore_other_families: bool,
    eval_split: str = "full",
    merge_device: str = "auto",
    scaling_mode: str = "auto",
) -> None:
    """
    Delegate to ``run_eval.run_single_target_pipeline`` with the 32B choices
    pinned: per-(layer, attn/mlp)-group evaluation blocks and no scaling (the
    merge is a plain elementwise average, identical on cpu and cuda). See the
    module docstring for the full list of behavior notes.

    scaling_mode (--scaling): NOTE that enable_scaling=False below does NOT
    stop run_eval's per-op AUTO-trigger (cross-family / AUTO_SCALE_FAMILIES,
    which matches "qwen" and would scale the Qwen3-32B trio). The set-level
    decision for the 32B same-base finetunes is NO scaling, so run_figures
    passes --scaling off here; the default "auto" preserves the historical
    behavior for direct CLI use.
    """
    # Resolve the 32B-only --domain aliases before delegating; run_eval passes
    # unknown keys straight through to TASK_REGISTRY, so an already-resolved
    # registry key survives its own DOMAIN_TO_REGISTRY_TASK lookup unchanged.
    target_domain_or_task = DOMAIN_TO_REGISTRY_TASK.get(
        target_domain_or_task, target_domain_or_task
    )
    _run_eval.run_single_target_pipeline(
        ops_csv=ops_csv,
        ops_step_csvs_dir=ops_step_csvs_dir,
        sorted_ops_out=sorted_ops_out,
        label_map_json=label_map_json,
        target_label=target_label,
        target_domain_or_task=target_domain_or_task,
        working_root=working_root,
        results_csv=results_csv,
        output_dir=output_dir,
        gpu_ids=gpu_ids,
        timeout_minutes=timeout_minutes,
        batch_size=batch_size,
        temperature=temperature,
        drop_tolerance=drop_tolerance,
        eval_enabled=eval_enabled,
        sort_mode=sort_mode,
        # The old 32B walker evaluated once per contiguous (stage, layer,
        # attn/mlp) block; never bundle a full layer into one eval step.
        bundle_eval_mode="group",
        ignore_other_families=ignore_other_families,
        # The 32B std-rescale is retired (see module docstring). The merge is
        # elementwise, so resolve_merge_device sees enable_scaling=False and
        # `auto` picks cpu -- bitwise-identical to cuda for this arithmetic.
        enable_scaling=False,
        merge_device=merge_device,
        eval_split=eval_split,
        scaling_mode=scaling_mode,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply merge ops for a single target model and evaluate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Legacy mode: single CSV file
  python run_eval.py --ops_csv merge_operations_log.csv --target_label fin-llama3.1-8b --domain finance ...

  # New mode: per-model CSV files (step1 and step2)
  python run_eval.py --ops_step_csvs_dir ./ops_step_csvs --target_label fin-llama3.1-8b --domain finance ...
        """
    )
    csv_group = parser.add_mutually_exclusive_group(required=True)
    csv_group.add_argument("--ops_csv", help="Path to operations CSV (e.g., merge_operations_log.csv) - legacy mode")
    csv_group.add_argument("--ops_step_csvs_dir", help="Directory containing per-model CSV files (ops_step1_*.csv, ops_step2_*.csv) - new mode")
    parser.add_argument("--label_map_json", required=True, help="Path to JSON mapping of label -> model path")
    parser.add_argument("--target_label", required=True, help="Label of the target model to modify/evaluate")
    parser.add_argument("--domain", required=True, help="High-level domain or registry task (e.g., medical, finance, sst2)")
    parser.add_argument("--working_root", default=os.environ.get("MICR_WORKING_ROOT", "/tmp/micr_merged_models"), help="Directory for working copies")
    parser.add_argument("--results_csv", default="./ops_step_csvs/target_steps.csv", help="Per-step CSV output for the target")
    parser.add_argument("--output_dir", default="./evaluation_results", help="Eval outputs directory")
    parser.add_argument("--gpu_ids", default="0", help="CUDA_VISIBLE_DEVICES value for evaluation")
    parser.add_argument("--timeout_minutes", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--merge_device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device for the merge arithmetic (auto = cpu unless scaling; see micr/merge_device.py).",
    )
    parser.add_argument("--drop_tolerance", type=float, default=2.0, help="Allowed absolute drop before rejecting a step")
    parser.add_argument("--sorted_ops_out", default="./sorted_ops.csv", help="Where to export sorted ops CSV")
    parser.add_argument("--no_eval", action="store_true", help="Disable evaluation (accept all steps without scoring)")
    parser.add_argument(
        "--eval_split",
        default="full",
        help=(
            "Name of the evaluation split this run gates on (e.g. 'full', or a "
            "MICR data split). Stored baselines are split-keyed and only reused "
            "on an exact split match; otherwise a fresh baseline is measured on "
            "this split."
        ),
    )
    parser.add_argument(
        "--sort_mode",
        choices=["normal", "separate", "together"],
        default="normal",
        help="Order of operations. 'normal': step1 then step2 as-is. "
             "'separate': sort each step by layer. "
             "'together': combine steps, de-duplicate favoring step2, then sort all by layer."
    )
    parser.add_argument(
        "--ignore-other-families",
        dest="ignore_other_families",
        action="store_true",
        help=(
            "When set, ignore donors whose labels appear to come from a different "
            "model family than the target (e.g., for a deepseek target, keep only "
            "other deepseek-* models). Steps where only a single same-family model "
            "remains are skipped entirely."
        ),
    )
    parser.add_argument(
        "--scaling",
        choices=["auto", "on", "off"],
        default="auto",
        help="Per-run scaling policy (see run_eval.py --scaling). run_figures "
             "passes 'off' for the Qwen3-32B trio: same-base finetunes, so the "
             "AUTO_SCALE_FAMILIES 'qwen' auto-trigger must not fire. Default "
             "'auto' keeps the historical behavior for direct CLI use.",
    )
    args = parser.parse_args()

    eval_enabled = not args.no_eval

    run_single_target_pipeline(
        ops_csv=args.ops_csv,
        ops_step_csvs_dir=args.ops_step_csvs_dir,
        sorted_ops_out=args.sorted_ops_out,
        label_map_json=args.label_map_json,
        target_label=args.target_label,
        target_domain_or_task=args.domain,
        working_root=args.working_root,
        results_csv=args.results_csv,
        output_dir=args.output_dir,
        gpu_ids=args.gpu_ids,
        timeout_minutes=args.timeout_minutes,
        batch_size=args.batch_size,
        temperature=args.temperature,
        drop_tolerance=args.drop_tolerance,
        eval_enabled=eval_enabled,
        sort_mode=args.sort_mode,
        ignore_other_families=args.ignore_other_families,
        eval_split=args.eval_split,
        merge_device=args.merge_device,
        scaling_mode=args.scaling,
    )


if __name__ == "__main__":
    main()
