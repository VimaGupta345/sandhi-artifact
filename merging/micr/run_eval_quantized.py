#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-model merge-and-evaluate runner with support for Quantized (FP8 compressed-tensors) models.

- Reads an ops CSV (merge operations) and optionally exports a sorted order CSV
- Applies only operations that involve the specified target label as a participant
- For each matching MERGE step:
    * builds a candidate for the target by N-way averaging donor parameters into the target
    * evaluates the candidate on the target's task (vLLM by default, safe fallback)
    * if accepted (within drop_tolerance), replaces the target's working directory in place
    * logs per-step results to a CSV (one CSV per target)

Usage example is printed at the bottom when running with -h/--help.
"""

import os
import sys
import json
import csv
import shutil
import re
import argparse
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import types as _types
if "merge_tools" not in sys.modules:
    _mt_pkg = _types.ModuleType("merge_tools")
    _mt_pkg.__path__ = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
    sys.modules["merge_tools"] = _mt_pkg

import pandas as pd
import torch
from transformers import AutoModelForCausalLM

# The evaluation layer now lives in this repo: micr/eval_harness.py. It builds the same
# argv and parses scores with the same regexes as the former unified-llm-eval checkout
# (verified: 9/9 tasks byte-identical), but runs in the current interpreter's environment
# -- no conda env names, no UNIFIED_LLM_EVAL_ROOT. The vendored math-evaluation-harness is
# still required for --domain math; point MICR_MATH_HARNESS_DIR at it.
from merge_tools.micr import eval_harness  # type: ignore
from merge_tools.micr.eval_harness import EnvironmentManager, TASK_REGISTRY  # type: ignore

# Import core helpers from the existing experiment module
from merge_tools.micr.eval_retry import fallback_is_redundant as _fallback_is_redundant  # type: ignore
from merge_tools.micr.eval_retry import (  # type: ignore
    evaluate_baseline_blocking as _evaluate_baseline_blocking,
    evaluate_with_retry as _evaluate_with_retry,
)
from merge_tools.micr.merge_device import describe as _describe_merge_device  # type: ignore
from merge_tools.micr.merge_device import resolve_merge_device  # type: ignore
from merge_tools.micr.baselines import get_baseline  # type: ignore
from merge_tools.micr.top_k_experiment import (  # type: ignore
    _infer_model_dtype as tk_infer_model_dtype,
    _load_model as tk_load_model,
    _get_layer_module as tk_get_layer_module,
    _collect_group_parameters as tk_collect_group_parameters,
)

# ---------------------------------------------------------------------------
# Quantized Model Utilities (from gaussian_profiler - canonical implementation)
# ---------------------------------------------------------------------------

from merge_tools.gaussian_profiler.quantized_utils import (
    is_quantized_model,
    load_quantized_model,
    sanitize_quantization_config_on_disk,
    FP8_DTYPE,
    FP8_MAX,
)


def _load_quantized_with_log(path: str, device_hint: str = "cuda") -> torch.nn.Module:
    """Wrapper that adds [quantized] log prefix for run_eval_quantized."""
    print(f"[quantized] Loading model from {path}...")
    return load_quantized_model(path, device_hint)


def safe_load_model(path: str, device_hint: str = "cuda", dtype_hint: Optional[torch.dtype] = None) -> torch.nn.Module:
    """
    Wrapper to load either standard or quantized models appropriately.
    """
    if is_quantized_model(path):
        return _load_quantized_with_log(path, device_hint)
    return tk_load_model(path, device_hint, dtype_hint)

# ---------------------------------------------------------------------------


# Component to group mapping (collapse subcomponents into attn/mlp groups)
COMPONENT_TO_GROUP: Dict[str, str] = {
    "mlp": "mlp",
    "mlp_gate": "mlp",
    "mlp_up": "mlp",
    "mlp_down": "mlp",
    "attn_q": "attn",
    "attn_k": "attn",
    "attn_v": "attn",
    "attn_o": "attn",
}

# Explicit ordering and attribute names for attention subcomponents.
ATTN_COMPONENT_ORDER: List[str] = ["attn_q", "attn_k", "attn_v", "attn_o"]
ATTN_COMPONENT_TO_ATTR: Dict[str, str] = {
    "attn_q": "q_proj",
    "attn_k": "k_proj",
    "attn_v": "v_proj",
    "attn_o": "o_proj",
}

# Explicit ordering and attribute names for MLP subcomponents.
MLP_COMPONENT_ORDER: List[str] = ["mlp_gate", "mlp_up", "mlp_down"]
MLP_COMPONENT_TO_ATTR: Dict[str, str] = {
    "mlp_gate": "gate_proj",
    "mlp_up": "up_proj",
    "mlp_down": "down_proj",
}

# Domain -> eval_harness task key mapping (override with --registry_task if desired)
# These keys should match TASK_REGISTRY entries in micr/eval_harness.py.
DOMAIN_TO_REGISTRY_TASK: Dict[str, str] = {
    # Existing domains
    "medical": "mmlu_professional_medicine",
    "finance": "mmlu_econometrics",
    "legal": "mmlu_professional_law",
    "toxicity": "sst2",
    "truthfulness": "truthfulqa_mc2",
    # New domains:
    # - Math-oriented models (e.g., Qwen2.5-Math-7B-Instruct): use GSM8K CoT.
    "math": "gsm8k-cot",
    # - Code-oriented models (e.g., Qwen2.5-Coder-7B-Instruct): use HumanEval.
    "code": "humaneval",
    "coder": "humaneval",
    # - Cybersecurity-oriented models (e.g., Foundation-Sec-8B): use WMDP cyber benchmark.
    "cyber": "wmdp_cyber",
    "cybersecurity": "wmdp_cyber",
}

# Baseline scores are looked up from the gaussian profiler output via
# merge_tools.micr.baselines.get_baseline (single source of truth); the former
# HARDCODED_BASELINE_SCORES dict was removed. Baselines are keyed by evaluation
# split (profiler split vs MICR split vs full set) and scores from different
# splits are not comparable, so the lookup only matches this runner's split
# (--eval_split); otherwise the runner measures a fresh baseline on its own
# split (see run_single_target_pipeline).


def _append_csv_row(csv_path: str, row: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path)
    write_header = True
    if file_exists:
        try:
            write_header = os.path.getsize(csv_path) == 0
        except OSError:
            write_header = True
    fieldnames = [
        "timestamp",
        "step_idx",
        "stage",
        "op",
        "component",
        "layer",
        "label",
        "score",
        "threshold",
        "decision",
    ]
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists or write_header:
            writer.writeheader()
        stamped = dict(row)
        stamped["timestamp"] = datetime.now(timezone.utc).isoformat()
        writer.writerow(stamped)


def _save_model_to_temp_dir_for_eval(
    model: torch.nn.Module,
    base_model_path: str,
    tmp_root: str,
) -> str:
    """
    Save the modified model to a temporary directory and copy / recreate
    tokenizer/config assets so that the eval harness (including vLLM) can load it.
    """
    os.makedirs(tmp_root, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=tmp_root)

    # Coerce model tensors to the base model's dtype (helps vLLM + HF loaders)
    try:
        # Don't force dtype if quantized, preserve auto/config
        if not is_quantized_model(base_model_path):
            target_dtype = tk_infer_model_dtype(base_model_path)
            model.to(dtype=target_dtype)
            if hasattr(model, "config"):
                model.config.torch_dtype = target_dtype
    except Exception:
        pass

    # Save model weights
    try:
        model.save_pretrained(tmp_dir, safe_serialization=True)
    except TypeError:
        model.save_pretrained(tmp_dir)

    # Try to recreate tokenizer via AutoTokenizer (prefer fast)
    try:
        from transformers import AutoTokenizer

        model_path_lower = str(base_model_path).lower()
        # Apply Mistral regex fix for affected models. The warning comes from
        # SentencePiece using a legacy regex that splits on digits, which can
        # break down numbers. Qwen models also need this fix.
        should_fix_regex = (
            "mistral" in model_path_lower
            or "calme" in model_path_lower
            or "llama-3" in model_path_lower
            or "llama3" in model_path_lower
            or "qwen" in model_path_lower
            or "safetyguard" in model_path_lower
        )

        if should_fix_regex:
            print(f"[tokenizer-fix] Applying fix_mistral_regex=True for '{base_model_path}'")
            tok = AutoTokenizer.from_pretrained(
                base_model_path,
                use_fast=True,
                trust_remote_code=True,
                fix_mistral_regex=True,
            )
        else:
            tok = AutoTokenizer.from_pretrained(
                base_model_path, use_fast=True, trust_remote_code=True
            )

        tok.save_pretrained(tmp_dir)
        return tmp_dir
    except Exception as e:
        print(f"[tmp-save] AutoTokenizer fallback for '{base_model_path}': {e}")

    # Fallback: copy tokenizer/config artifacts from base_model_path
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "spiece.model",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
        "quantization_config.json",
    ]
    base = Path(base_model_path)
    for fname in tokenizer_files:
        src = base / fname
        if src.exists():
            try:
                shutil.copy(str(src), tmp_dir)
            except Exception:
                pass

    return tmp_dir


def get_model_component_stds(model_path: str) -> Dict[str, float]:
    """
    Load model and compute std dev for all relevant weights.
    Returns dict: "{layer_idx}.{component_name}" -> std_dev
    e.g. "0.attn_q" -> 0.0123
    """
    print(f"[profile] Loading {model_path} to compute original std devs...")
    try:
        model = safe_load_model(model_path, device_hint="cpu")
        stds = {}
        
        # Try to determine number of layers
        num_layers = 0
        if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
            num_layers = model.config.num_hidden_layers
        else:
            # Fallback: try accessing layers until failure
            for i in range(1000):
                try:
                    _ = tk_get_layer_module(model, i)
                    num_layers = i + 1
                except Exception:
                    break
        
        print(f"[profile] Found {num_layers} layers.")

        is_quantized = is_quantized_model(model_path)

        for i in range(num_layers):
            layer = tk_get_layer_module(model, i)
            
            # Attn
            if hasattr(layer, "self_attn"):
                for comp_name, attr_name in ATTN_COMPONENT_TO_ATTR.items():
                    mod = getattr(layer.self_attn, attr_name, None)
                    if mod is not None and hasattr(mod, "weight"):
                        w = mod.weight
                        if is_quantized:
                            w = w.detach().to(torch.float32)
                        stds[f"{i}.{comp_name}"] = w.std().item()

            # MLP
            if hasattr(layer, "mlp"):
                for comp_name, attr_name in MLP_COMPONENT_TO_ATTR.items():
                    mod = getattr(layer.mlp, attr_name, None)
                    if mod is not None and hasattr(mod, "weight"):
                        w = mod.weight
                        if is_quantized:
                            w = w.detach().to(torch.float32)
                        stds[f"{i}.{comp_name}"] = w.std().item()
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return stds
    except Exception as e:
        print(f"[profile] Failed to compute stds: {e}")
        return {}


def _n_way_average_subcomponent_into_target(
    target_model: torch.nn.Module,
    target_model_path: str,
    target_layer_idx: int,
    group_name: str,
    component_attr: str,
    contributors: List[Tuple[torch.nn.Module, int]],
    device: str = "cuda",
    target_std: Optional[float] = None,
) -> None:
    """
    N-way average of a *single subcomponent* (attn q/k/v/o OR mlp gate/up/down) across
    [target + partners] INTO target_model, without touching other subcomponents.
    
    contributors: List of (model, source_layer_idx) tuples. 
                  Includes partners loaded in memory.
    """
    if not component_attr:
        return

    is_quantized = is_quantized_model(target_model_path)
    # If quantized, we operate in higher precision then cast back
    calc_dtype = torch.float32 if is_quantized else tk_infer_model_dtype(target_model_path)

    layer_t = tk_get_layer_module(target_model, target_layer_idx)
    
    # Identify the sub-module (attn or mlp)
    if group_name == "attn":
        sub_mod_t = getattr(layer_t, "self_attn", None)
    elif group_name == "mlp":
        sub_mod_t = getattr(layer_t, "mlp", None)
    else:
        return

    if sub_mod_t is None or not hasattr(getattr(sub_mod_t, component_attr, None), "weight"):
        return
    
    proj_t = getattr(sub_mod_t, component_attr)
    p_t = proj_t.weight

    try:
        contributor_weights: List[torch.Tensor] = []
        
        for m, src_layer_idx in contributors:
            layer_m = tk_get_layer_module(m, src_layer_idx)
            if group_name == "attn":
                sub_mod_m = getattr(layer_m, "self_attn", None)
            else:
                sub_mod_m = getattr(layer_m, "mlp", None)
            
            if sub_mod_m is None or not hasattr(getattr(sub_mod_m, component_attr, None), "weight"):
                 raise RuntimeError(f"Contributor model missing {group_name} subcomponent '{component_attr}' at layer {src_layer_idx}")
            
            proj_m = getattr(sub_mod_m, component_attr)
            contributor_weights.append(proj_m.weight)

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        with torch.no_grad():
            acc = torch.zeros_like(p_t.data, dtype=torch.float32, device=device)
            
            # Sum up all contributors
            for w_m in contributor_weights:
                # Always cast to float32 for accumulation
                acc += w_m.data.to(device=device, dtype=torch.float32)
            
            # N-way average
            k = len(contributor_weights)
            if k > 0:
                acc /= float(k)
                
                # Rescale to match original target std dev if provided
                if target_std is not None:
                    curr_std = acc.std()
                    if curr_std > 1e-9:
                        scale = target_std / curr_std
                        acc *= scale
                
                # If quantized, clamp to valid range and cast to FP8
                if is_quantized:
                    acc.clamp_(-FP8_MAX, FP8_MAX)
                    out_final = acc.to(device=p_t.data.device, dtype=FP8_DTYPE)
                else:
                    out_final = acc.to(device=p_t.data.device, dtype=calc_dtype)
                
                p_t.data.copy_(out_final)
            
            if device == "cuda":
                torch.cuda.synchronize()

        if hasattr(target_model, "config") and not is_quantized:
            target_model.config.torch_dtype = calc_dtype
    except Exception as e:
        print(f"Error in averaging {component_attr}: {e}")
        raise e


def _n_way_average_subcomponent_into_target_streaming(
    target_model: torch.nn.Module,
    target_model_path: str,
    target_layer_idx: int,
    group_name: str,
    component_attr: str,
    participants: List[Tuple[str, int]],
    *,
    target_label: str,
    label_to_path: Dict[str, str],
    working_dir: Path,
    current_stage: int,
    device: str = "cuda",
    target_std: Optional[float] = None,
) -> None:
    """
    Memory-efficient variant of N-way averaging for a single subcomponent.

    Instead of loading *all* contributor models into memory at once, this
    helper streams them one-by-one:
      - The target model stays loaded in memory for the duration of the block.
      - Each donor model is loaded, its contribution accumulated into an
        accumulator tensor, and then immediately freed.

    This keeps peak memory bounded to roughly:
        target_model + 1 donor_model + accumulator tensor.
    """
    if not component_attr:
        return

    is_quantized = is_quantized_model(target_model_path)
    calc_dtype = torch.float32 if is_quantized else tk_infer_model_dtype(target_model_path)

    layer_t = tk_get_layer_module(target_model, target_layer_idx)

    # Identify the sub-module (attn or mlp)
    if group_name == "attn":
        sub_mod_t = getattr(layer_t, "self_attn", None)
    elif group_name == "mlp":
        sub_mod_t = getattr(layer_t, "mlp", None)
    else:
        return

    if sub_mod_t is None or not hasattr(getattr(sub_mod_t, component_attr, None), "weight"):
        return

    proj_t = getattr(sub_mod_t, component_attr)
    p_t = proj_t.weight

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    try:
        with torch.no_grad():
            acc = torch.zeros_like(p_t.data, dtype=torch.float32, device=device)
            k = 0

            for lbl, src_layer in participants:
                model_obj: Optional[torch.nn.Module] = None
                try:
                    # Stage 1 (or any non-2): use in-memory target when it appears
                    if current_stage != 2 and lbl == target_label:
                        model_obj = target_model
                    else:
                        # Donor path: either from label_map or (for target) fallback to working_dir
                        donor_path = label_to_path.get(lbl)
                        if not donor_path and lbl == target_label:
                            donor_path = str(working_dir)
                        if not donor_path:
                            print(f"  [warning] No path for donor label '{lbl}', skipping.")
                            continue
                        
                        # Use safe_load_model to handle quantization
                        model_obj = safe_load_model(
                            donor_path,
                            device_hint="cpu",
                            dtype_hint=calc_dtype if not is_quantized else None,
                        )

                    layer_m = tk_get_layer_module(model_obj, src_layer)
                    if group_name == "attn":
                        sub_mod_m = getattr(layer_m, "self_attn", None)
                    else:
                        sub_mod_m = getattr(layer_m, "mlp", None)

                    if sub_mod_m is None or not hasattr(getattr(sub_mod_m, component_attr, None), "weight"):
                        raise RuntimeError(
                            f"Contributor model '{lbl}' missing {group_name} "
                            f"subcomponent '{component_attr}' at layer {src_layer}"
                        )

                    proj_m = getattr(sub_mod_m, component_attr)
                    acc += proj_m.weight.data.to(device=device, dtype=torch.float32)
                    k += 1
                finally:
                    # Only free donor models, never the in-memory target
                    if model_obj is not None and model_obj is not target_model:
                        del model_obj
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            if k == 0:
                return

            acc /= float(k)

            # Rescale to match original target std dev if provided
            if target_std is not None:
                curr_std = acc.std()
                if curr_std > 1e-9:
                    scale = target_std / curr_std
                    acc *= scale

            if is_quantized:
                acc.clamp_(-FP8_MAX, FP8_MAX)
                out_final = acc.to(device=p_t.data.device, dtype=FP8_DTYPE)
            else:
                out_final = acc.to(device=p_t.data.device, dtype=calc_dtype)
            
            p_t.data.copy_(out_final)

            if device == "cuda":
                torch.cuda.synchronize()

        if hasattr(target_model, "config") and not is_quantized:
            target_model.config.torch_dtype = calc_dtype
    except Exception as e:
        print(
            f"Error in streaming average for {group_name}.{component_attr} "
            f"at layer {target_layer_idx}: {e}"
        )
        raise e


def create_unified_eval_context(
    gpu_ids: Optional[str] = None,
    timeout_minutes: int = 15,
    batch_size: int = 32,
    temperature: float = 0.0,
    output_dir: str = "./evaluation_results",
):
    if EnvironmentManager is None:
        raise RuntimeError(
            "EnvironmentManager could not be imported. "
            "micr/eval_harness.py failed to import."
        )
    if gpu_ids is not None and str(gpu_ids).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids).strip()
    eval_settings = {
        "gpu_ids": "" if gpu_ids is None else str(gpu_ids),
        "timeout_minutes": int(timeout_minutes),
        "batch_size": int(batch_size),
        "temperature": float(temperature),
        "runs_per_eval": 1,
        "output_dir": output_dir,
    }
    env_config = {"math_harness_dir": eval_harness.math_harness_dir()}
    env_manager = EnvironmentManager()
    return env_manager, env_config, eval_settings


def evaluate_model_for_task(
    model_path: str,
    task_or_domain: str,
    *,
    env_manager=None,
    env_config: Optional[dict] = None,
    eval_settings: Optional[dict] = None,
    gpu_ids: Optional[str] = None,
    timeout_minutes: int = 15,
    batch_size: int = 32,
    temperature: float = 0.0,
    output_dir: str = "./evaluation_results",
    registry_task_name: Optional[str] = None,
    allow_fallback: bool = True,
) -> Optional[float]:
    """
    Evaluate a single model on a single task using micr/eval_harness.py.

    Returns the measured score, or None when no score was produced (harness
    unavailable, unknown task, evaluator crash/timeout). A returned 0.0 always
    means the model genuinely scored zero; callers must not conflate the two.
    """
    if TASK_REGISTRY is None:
        print("Unified evaluation harness not available (TASK_REGISTRY is None).")
        return None

    # Build or reuse context on demand
    if env_manager is None or env_config is None or eval_settings is None:
        env_manager, env_config, eval_settings = create_unified_eval_context(
            gpu_ids=gpu_ids,
            timeout_minutes=timeout_minutes,
            batch_size=batch_size,
            temperature=temperature,
            output_dir=output_dir,
        )

    registry_key = registry_task_name or DOMAIN_TO_REGISTRY_TASK.get(task_or_domain, task_or_domain)
    task_info = TASK_REGISTRY.get(registry_key)
    if not task_info:
        print(f"Task '{registry_key}' not in TASK_REGISTRY.")
        return None

    EvaluatorClass = task_info["evaluator"]
    # Prefer vLLM by default with safe fallback
    try:
        eval_settings = dict(eval_settings or {})
    except Exception:
        eval_settings = {}
    eval_settings.setdefault("use_vllm", True)

    # Limit number of evaluation examples for SimpleEvaluator-based tasks (e.g., finance)
    # by passing lm_eval's --limit flag via eval_settings["limit"].
    # SIMPLE_TASKS_WITH_LIMIT = {
    #     "financial_tweets",
    #     "medqa_4options",
    #     "mmlu_international_law",
    #     "sst2",
    #     "truthfulqa_mc2",
    #     "mmlu_professional_medicine",
    #     "mmlu_professional_law",
    #     "mmlu_econometrics",
    # }
    # if registry_key in SIMPLE_TASKS_WITH_LIMIT:
    #     eval_settings.setdefault("limit", 2500)  # cap number of eval examples (lm_eval --limit)
    #     # For SimpleEvaluator tasks, let lm_eval choose the batch size automatically.
    #     eval_settings["batch_size"] = 32
    #     if registry_key == "financial_tweets":
    #         eval_settings["gpu_ids"] = "4,5,6,7"
    evaluator = EvaluatorClass(env_manager, env_config, eval_settings)

    model_path_obj = Path(model_path)
    is_hf_model = "/" in model_path and not model_path_obj.exists()
    if is_hf_model:
        model_config = {"model_name": model_path.split("/")[-1], "path": model_path, "location": "huggingface"}
    else:
        abs_path = model_path_obj.resolve()
        model_config = {"model_name": abs_path.name, "path": str(abs_path), "location": "local"}

    result = evaluator.evaluate(model_config, registry_key, run_id=1)
    score_val: Optional[float] = None
    if result.get("status") == "SUCCESS":
        try:
            score_val = float(str(result["score"]).replace("%", ""))
        except Exception:
            score_val = None
    # A SUCCESS that parsed to exactly 0.0 is a real measurement, not a failure.
    measured_zero = score_val == 0.0
    if score_val is not None and score_val != 0.0:
        return score_val

    # Fallback to non-vLLM -- but only when it would actually run a different command.
    # HarnessEvaluator hardcodes --use_vllm / --model vllm and ignores eval_settings,
    # so its "fallback" re-runs the identical subprocess.
    if (allow_fallback and isinstance(eval_settings, dict)
            and eval_settings.get("use_vllm", False)
            and not _fallback_is_redundant(EvaluatorClass, env_manager, env_config,
                                           eval_settings, model_config["path"], registry_key)):
        eval_settings_retry = dict(eval_settings)
        eval_settings_retry["use_vllm"] = False
        evaluator_retry = EvaluatorClass(env_manager, env_config, eval_settings_retry)
        result2 = evaluator_retry.evaluate(model_config, registry_key, run_id=1)
        if result2.get("status") == "SUCCESS":
            try:
                return float(str(result2["score"]).replace("%", ""))
            except Exception:
                pass

    if measured_zero:
        return 0.0

    print(f"Evaluation failed: {result.get('error_log', 'Unknown error')} (status={result.get('status')})")
    return None


# The retry policy lives in micr/eval_retry.py so all four runners share one
# definition of what a failed evaluation means.
def evaluate_with_retry(label: str, **kwargs) -> Optional[float]:
    return _evaluate_with_retry(evaluate_model_for_task, label, **kwargs)


def evaluate_baseline_blocking(label: str, **kwargs) -> float:
    return _evaluate_baseline_blocking(evaluate_model_for_task, label, **kwargs)


def reorder_ops_log_for_schedule(
    ops_log: pd.DataFrame,
    *,
    num_layers: int = 32,
) -> pd.DataFrame:
    """
    Reorder ops_log using a two-phase schedule; replaces still come last.
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
        return COMPONENT_TO_GROUP.get(comp, comp)

    merges["group"] = merges["component"].apply(_comp_group)
    merges["layer"] = merges["layer"].astype(int)

    # Param counts per layer for attn / mlp
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
        *,
        prefer_adjacent_to: Optional[Set[int]] = None,
    ) -> Tuple[pd.DataFrame, Set[int]]:
        sub = merges[merges["group"] == group_name].copy()
        if sub.empty:
            return sub, set()

        # Group rows per layer; keep original intra-layer order by orig_idx
        layer_to_rows: Dict[int, pd.DataFrame] = {}
        for layer, block in sub.sort_values("orig_idx").groupby("layer"):
            layer_to_rows[int(layer)] = block

        remaining_layers: Set[int] = set(layer_to_rows.keys())
        used_layers: Set[int] = set()
        selected_layers: List[int] = []

        def _is_adjacent_to_used(layer: int) -> bool:
            return any(abs(layer - ul) == 1 for ul in used_layers)

        def _adjacent_to_pref(layer: int) -> bool:
            if not prefer_adjacent_to:
                return False
            return any(abs(layer - pl) == 1 for pl in prefer_adjacent_to)

        def _score(layer: int) -> Tuple[int, float, float]:
            adj_bonus = 1 if _adjacent_to_pref(layer) else 0
            base_p = param_count_by_layer.get(layer, float(layer_to_rows[layer]["n_models"].sum()))
            dist_mid = abs(layer - mid_layer)
            return (adj_bonus, base_p, -dist_mid)

        while remaining_layers:
            candidates = list(remaining_layers)
            non_adj = [L for L in candidates if not _is_adjacent_to_used(L)]
            working = non_adj if non_adj else candidates
            best_layer = max(working, key=_score)
            selected_layers.append(best_layer)
            remaining_layers.remove(best_layer)
            used_layers.add(best_layer)

        ordered = pd.concat([layer_to_rows[L] for L in selected_layers], ignore_index=True)
        return ordered, used_layers

    # Phase 1: attn
    phase1, attn_layers_used = _reorder_one_group("attn", attn_param_by_layer)
    # Phase 2: mlp (prefer layers adjacent to attn layers)
    phase2, _ = _reorder_one_group("mlp", mlp_param_by_layer, prefer_adjacent_to=attn_layers_used)

    reordered = pd.concat(
        [
            phase1[df.columns] if not phase1.empty else phase1,
            phase2[df.columns] if not phase2.empty else phase2,
            replaces,  # original order preserved for replaces
        ],
        ignore_index=True,
    )
    return reordered


def parse_participants_with_layers(models_field: str, default_layer: int) -> List[Tuple[str, int]]:
    """
    Parse 'models' field like 'labelA:12,labelB' -> [('labelA', 12), ('labelB', default_layer)].
    """
    out: List[Tuple[str, int]] = []
    if not isinstance(models_field, str) or not models_field.strip():
        return out
    for tok in models_field.split(","):
        tok = tok.strip()
        if not tok:
            continue
        parts = tok.split(":")
        lbl = parts[0].strip()
        if not lbl:
            continue
        
        layer = default_layer
        if len(parts) > 1 and parts[1].strip().isdigit():
            layer = int(parts[1].strip())
            
        out.append((lbl, layer))
    return out


def infer_family_prefix(label: str) -> str:
    """
    Infer a coarse-grained 'family' prefix from a model label.
    Matches 'deepseek', 'qwen', 'llama' (case-insensitive) anywhere in the string.
    Fallback to first alphabetic run.
    """
    if not isinstance(label, str):
        return ""
    s = label.lower()
    if "deepseek" in s:
        return "deepseek"
    if "qwen" in s:
        return "qwen"
    if "llama" in s:
        return "llama"
    if "calme" in s:
        return "llama"
    if "mistral" in s:
        return "mistral"
    
    m = re.search(r"[a-z]+", s)
    return m.group(0) if m else s


def load_ops_csvs_for_target(
    ops_csv: Optional[str] = None,
    ops_step_csvs_dir: Optional[str] = None,
    target_label: str = "",
) -> pd.DataFrame:
    """
    Load operations CSV(s) for a target model.
    
    Supports two modes:
    1. Single CSV file: if ops_csv is provided, load it directly
    2. Per-model CSVs: if ops_step_csvs_dir is provided, load step1 and step2 CSVs
       for the target model and combine them (step1 first, then step2)
    
    Args:
        ops_csv: Path to a single operations CSV file (legacy mode)
        ops_step_csvs_dir: Directory containing per-model CSV files (new mode)
        target_label: Model label to find the appropriate CSV files
    
    Returns:
        Combined DataFrame with operations in the correct order
    """
    if ops_csv:
        # Legacy mode: single CSV file
        df = pd.read_csv(ops_csv)
        df["stage"] = 1
        return df
    
    if not ops_step_csvs_dir:
        raise ValueError("Either ops_csv or ops_step_csvs_dir must be provided")
    
    if not target_label:
        raise ValueError("target_label is required when using ops_step_csvs_dir")
    
    # Sanitize target_label for filename matching
    safe_label = str(target_label).replace("/", "_")
    
    # Look for step1 and step2 CSV files for the target
    step1_path = Path(ops_step_csvs_dir) / f"ops_step1_{safe_label}.csv"
    step2_path = Path(ops_step_csvs_dir) / f"ops_step2_{safe_label}.csv"
    
    dfs = []
    
    # Load step1 if it exists
    if step1_path.exists():
        df_step1 = pd.read_csv(step1_path)
        if not df_step1.empty:
            df_step1["stage"] = 1
            dfs.append(df_step1)
            print(f"[load] Loaded step1 operations: {len(df_step1)} rows from {step1_path.name}")
    else:
        print(f"[load] Step1 CSV not found: {step1_path}")
    
    # Load step2 if it exists
    if step2_path.exists():
        df_step2 = pd.read_csv(step2_path)
        if not df_step2.empty:
            df_step2["stage"] = 2
            dfs.append(df_step2)
            print(f"[load] Loaded step2 operations: {len(df_step2)} rows from {step2_path.name}")
    else:
        print(f"[load] Step2 CSV not found: {step2_path}")
    
    if not dfs:
        raise FileNotFoundError(
            f"No operations CSV files found for target '{target_label}'. "
            f"Looked for: {step1_path.name} and/or {step2_path.name} in {ops_step_csvs_dir}"
        )
    
    # Combine all operations, preserving order (step1 before step2)
    combined = pd.concat(dfs, ignore_index=True)
    
    print(f"[load] Total operations loaded: {len(combined)} rows")
    
    return combined


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
    initial_baseline: Optional[float] = None,
    force_calc_baseline: bool = False,
    eval_split: str = "full",
    enable_scaling: bool = False,
    merge_device: str = "auto",
) -> None:
    # Load operations CSV(s) - supports both legacy single CSV and new per-model CSVs
    ops_log = load_ops_csvs_for_target(
        ops_csv=ops_csv,
        ops_step_csvs_dir=ops_step_csvs_dir,
        target_label=target_label,
    )
    # Respect the order already present in the CSV(s); do not reorder or re-export.
    ops_sorted = ops_log.copy()

    with open(label_map_json, "r") as f:
        label_to_path: Dict[str, str] = json.load(f)
    if target_label not in label_to_path:
        raise RuntimeError(f"Target label '{target_label}' not found in label map.")

    # Working copy for target
    target_src = Path(label_to_path[target_label]).resolve()
    working_dir = Path(working_root) / target_src.name
    os.makedirs(working_root, exist_ok=True)
    if not working_dir.exists():
        print(f"[init] copy {target_label}: {target_src} -> {working_dir}")
        shutil.copytree(str(target_src), str(working_dir))

    # Sanitize config on disk so vLLM (eval harness) can load quantized models
    if is_quantized_model(str(working_dir)):
        sanitize_quantization_config_on_disk(str(working_dir))
        print(f"[init] Sanitized config.json for vLLM compatibility")

    # Baseline and thresholds for target
    thresholds: Dict[str, float] = {}
    last_score: Dict[str, float] = {}

    thresholds[target_label] = 0.0
    last_score[target_label] = 0.0

    # Calculate original component std devs for scaling (optional)
    merge_device = resolve_merge_device(merge_device, enable_scaling)
    print(_describe_merge_device(merge_device, enable_scaling))

    original_stds: Dict[str, float] = {}
    if enable_scaling and target_src.exists():
        print(f"[profile] Computing original std devs for {target_label} because --enable-scaling is set")
        original_stds = get_model_component_stds(str(target_src))

    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids).strip()

    if eval_enabled:
        # Baseline precedence: explicit --initial_baseline (when given and not
        # forced) > stored baseline from get_baseline(target_label,
        # split=eval_split), which only matches a baseline measured on this
        # runner's own split > fresh measurement on this runner's split.
        # --force-calc-baseline forces a fresh measurement.
        base_acc = initial_baseline
        if force_calc_baseline:
            base_acc = evaluate_baseline_blocking(
                target_label,
                model_path=str(working_dir),
                task_or_domain=target_domain_or_task,
                gpu_ids=gpu_ids,
                timeout_minutes=timeout_minutes,
                batch_size=batch_size,
                temperature=temperature,
                output_dir=os.path.join(output_dir, target_label, "baseline"),
            )
        elif base_acc is None:
            base_acc = get_baseline(target_label, split=eval_split)
            if base_acc is not None:
                print(f"[baseline] {target_label}: using stored baseline (split={eval_split})")
            else:
                base_acc = evaluate_baseline_blocking(
                    target_label,
                    model_path=str(working_dir),
                    task_or_domain=target_domain_or_task,
                    gpu_ids=gpu_ids,
                    timeout_minutes=timeout_minutes,
                    batch_size=batch_size,
                    temperature=temperature,
                    output_dir=os.path.join(output_dir, target_label, "baseline"),
                )
        thresholds[target_label] = base_acc
        last_score[target_label] = base_acc
        print(f"[baseline] {target_label} = {base_acc:.4f}")
    else:
        print(f"[baseline] eval disabled, using default 0.0 for {target_label}")

    # Iterate over ordered MERGE ops; but evaluate once per (layer, group),
    # where group is "attn" (all q/k/v/o) or "mlp" (all gate/up/down).
    tmp_dir_root = os.path.join(working_root, "tmp_eval_single")
    os.makedirs(tmp_dir_root, exist_ok=True)

    merges_only = ops_sorted[ops_sorted["op"] == "merge"].copy()
    if merges_only.empty:
        print("[run] No MERGE operations in ops_csv; nothing to do.")
        return

    # Attach group column (attn/mlp) based on component
    merges_only["group"] = merges_only["component"].apply(
        lambda c: COMPONENT_TO_GROUP.get(str(c), str(c))
    )

    # --- Sorting Logic based on sort_mode ---
    if sort_mode == 'separate':
        print("[sort] Mode: 'separate'. Sorting each stage by layer.")
        step1_df = merges_only[merges_only['stage'] == 1].sort_values(['layer', 'group'], kind='stable')
        step2_df = merges_only[merges_only['stage'] == 2].sort_values(['layer', 'group'], kind='stable')
        merges_only = pd.concat([step1_df, step2_df], ignore_index=True)
    elif sort_mode == 'together':
        print("[sort] Mode: 'together'. Combining stages and sorting by layer.")
        # Sort by stage descending to prioritize step 2, then drop duplicates
        merges_only = merges_only.sort_values('stage', ascending=False, kind='stable')
        # Keep the first occurrence, which will be from step 2 if available
        merges_only = merges_only.drop_duplicates(subset=['layer', 'component'], keep='first')
        # Treat all as stage 1
        merges_only['stage'] = 1
        # Sort the final list by layer and group
        merges_only = merges_only.sort_values(['layer', 'group'], kind='stable')
    elif sort_mode == 'normal':
        print("[sort] Mode: 'normal'. Processing steps in order.")
        # For 'normal' mode, we respect the order from the CSVs.
        # The data is already in order from `load_ops_csvs_for_target`.
        pass
    else:
        # Fallback for any other case, though 'normal' is the default.
        merges_only = merges_only.sort_values(["stage", "layer", "group"], kind="stable")
    
    merges_only = merges_only.reset_index(drop=True)

    # --- Operation Summary ---
    print("\n" + "="*20 + " Operation Summary " + "="*20)
    print(f"Sort Mode: {sort_mode}")
    print(f"Total merge operations to process: {len(merges_only)}")
    if not merges_only.empty:
        print(merges_only[['stage', 'layer', 'group', 'component', 'models']].to_string())
    print("="*59 + "\n")

    step_counter = 0

    # Process merges in the *exact CSV order*, but evaluate once per contiguous
    # (layer, group) block (where group is 'attn' or 'mlp').
    current_layer: Optional[int] = None
    current_group: Optional[str] = None
    current_stage: Optional[int] = None
    current_indices: List[int] = []

    def _process_block(layer: int, group_name: str, indices: List[int], step_counter: int) -> int:
        if not indices:
            return step_counter
        block = merges_only.loc[indices]

        # Check stage of this block (assume homogeneous stage within a block)
        first_row = merges_only.loc[indices[0]]
        current_stage = first_row.get("stage", 1) 

        # Helper: filter participants to those in the same family as the target
        family_prefix = infer_family_prefix(target_label) if ignore_other_families else None

        def _filtered_participants(models_str: str) -> List[Tuple[str, int]]:
            participants = parse_participants_with_layers(str(models_str), default_layer=layer)
            if not family_prefix:
                return participants
            return [
                (lbl, lyr)
                for (lbl, lyr) in participants
                if infer_family_prefix(lbl) == family_prefix
            ]

        # Filter out rows that are just 'mlp' or 'attn' without subcomponents,
        # rows where target doesn't participate, or (when ignore_other_families
        # is enabled) rows where only a single same-family model remains.
        valid_indices = []
        for idx in indices:
            row = merges_only.loc[idx]
            comp = str(row["component"])
            if comp in ["mlp", "attn"]:
                continue
            models_str = str(row.get("models", ""))
            if target_label not in models_str:
                continue

            participants = _filtered_participants(models_str)
            if ignore_other_families:
                # Count unique labels *after* family filtering. If there is only
                # a single model left (typically just the target), then this
                # step would be a no-op merge, so we skip it.
                unique_labels = {lbl for (lbl, _) in participants}
                if len(unique_labels) <= 1:
                    continue

            valid_indices.append(idx)
        
        if not valid_indices:
            print(f"  [skip] No valid operations for {target_label} at layer={layer}, group={group_name} (stage={current_stage})")
            if ignore_other_families:
                print("    Reason: --ignore-other-families left <=1 same-family model in all rows for this block.")
            print(f"    Block had {len(indices)} rows, but all were filtered out")
            # Debug: show why rows were filtered
            for idx in indices[:3]:  # Show first 3 for debugging
                row = merges_only.loc[idx]
                comp = str(row["component"])
                models_str = str(row.get("models", ""))
                participants = _filtered_participants(models_str)
                unique_labels = {lbl for (lbl, _) in participants}
                print(
                    f"      Row {idx}: comp={comp}, "
                    f"target_in_models={target_label in models_str}, "
                    f"same_family_models={list(unique_labels)}"
                )
            return step_counter

        # Use valid_indices for the rest
        block = merges_only.loc[valid_indices]
        
        print(f"\n[step {step_counter}] target={target_label}, group={group_name}, layer={layer} (stage={current_stage})")

        # 1. Identify all unique donor models (Label, Layer) required for this block
        # We need to cache models by Label. Layer access is dynamic.
        required_labels: Set[str] = set()
        
        for _, r in block.iterrows():
            participants = _filtered_participants(str(r.get("models", "")))
            if not participants:
                continue
            for lbl, _ in participants:
                # If stage 2, we treat target as external donor
                if current_stage == 2:
                    required_labels.add(lbl) 
                else:
                    if lbl != target_label:
                        required_labels.add(lbl)

        # 2. Load all required donor models into memory
        loaded_models: Dict[str, torch.nn.Module] = {}
        # Infer dtype from target (unless quantized, handled inside safe_load_model)
        target_dtype = tk_infer_model_dtype(str(working_dir))
        
        try:
            for lbl in required_labels:
                path = label_to_path.get(lbl)
                if not path:
                    # If lbl is target_label and we are in stage 2, path is working_dir
                    if lbl == target_label:
                        path = str(working_dir)
                    else:
                        print(f"  [warning] No path for donor label '{lbl}', skipping.")
                        continue
                if lbl not in loaded_models:
                    print(f"  [load] donor: {lbl}")
                    loaded_models[lbl] = safe_load_model(
                        path, 
                        device_hint="cpu", 
                        dtype_hint=target_dtype
                    )
            
            # Load target model
            model_tgt = safe_load_model(
                str(working_dir),
                device_hint="cpu",
                dtype_hint=target_dtype
            )

            # 3. Apply operations for each subcomponent in order
            if group_name == "attn":
                comp_order = ATTN_COMPONENT_ORDER
                comp_map = ATTN_COMPONENT_TO_ATTR
            elif group_name == "mlp":
                comp_order = MLP_COMPONENT_ORDER
                comp_map = MLP_COMPONENT_TO_ATTR
            else:
                comp_order = []
                comp_map = {}

            for comp in comp_order:
                # Find row for this subcomponent
                sub_rows = block[block["component"] == comp]
                if sub_rows.empty:
                    continue
                
                # Should typically be one row per subcomponent per layer in the block, 
                # but if multiple, we process them (maybe overriding?). 
                # Assuming last one wins or we process all? 
                # Usually standard merge schedule implies unique op per subcomp.
                for _, r in sub_rows.iterrows():
                    participants = _filtered_participants(str(r.get("models", "")))
                    
                    # Check if target is involved *after* family filtering
                    if not any(lbl == target_label for (lbl, _) in participants):
                        continue
                        
                    print(f"  - applying {comp}: {participants}")
                    
                    # Build contributors list: (Model, SourceLayer)
                    contributors: List[Tuple[torch.nn.Module, int]] = []
                    for lbl, src_layer in participants:
                        if current_stage == 2 and lbl == target_label:
                             # Stage 2: treat target as loaded donor
                             if lbl in loaded_models:
                                 contributors.append((loaded_models[lbl], src_layer))
                             else:
                                 # Should have been loaded, but fallback/check
                                 print(f"    [warning] Target {lbl} not found in loaded_models for stage 2 op.")
                        elif lbl == target_label:
                             # Stage 1 / Legacy: use in-memory target
                             contributors.append((model_tgt, src_layer))
                        elif lbl in loaded_models:
                             contributors.append((loaded_models[lbl], src_layer))
                    
                    attr_name = comp_map.get(comp)
                    if attr_name:
                        _n_way_average_subcomponent_into_target(
                            target_model=model_tgt,
                            target_model_path=str(working_dir),
                            target_layer_idx=layer,
                            group_name=group_name,
                            component_attr=attr_name,
                            contributors=contributors,
                            device=merge_device,
                            target_std=original_stds.get(f"{layer}.{comp}")
                        )

            # 4. Save modified target model to temp
            tmp_model_dir = _save_model_to_temp_dir_for_eval(
                model_tgt,
                base_model_path=str(working_dir),
                tmp_root=tmp_dir_root,
            )

        except Exception as e:
            print(f"    ERROR building candidate for {target_label}: {e}")
            try:
                del model_tgt
            except Exception:
                pass
            # Clear cache on error
            loaded_models.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return step_counter

        # Clear memory before evaluation
        try:
            del model_tgt
        except Exception:
            pass
        loaded_models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Evaluate candidate
        if eval_enabled:
            new_acc = evaluate_with_retry(
                f"{target_label} step {step_counter}",
                model_path=tmp_model_dir,
                task_or_domain=target_domain_or_task,
                gpu_ids=gpu_ids,
                timeout_minutes=timeout_minutes,
                batch_size=batch_size,
                temperature=temperature,
                output_dir=os.path.join(output_dir, target_label, f"step_{step_counter}"),
            )
        else:
            new_acc = last_score[target_label]

        if new_acc is None:
            # No score was produced, so this merge was never judged. Make no
            # accept/reject claim: discard the candidate, leave working_dir as it
            # was, and record the failure instead of a fabricated 0.0 rejection.
            print(f"  [decision] {target_label}: EVAL FAILED after retries (model unchanged)")
            try:
                shutil.rmtree(tmp_model_dir)
            except Exception:
                pass
            _append_csv_row(
                results_csv,
                {
                    "step_idx": step_counter,
                    "stage": current_stage,
                    "op": "merge",
                    "component": group_name,
                    "layer": int(layer),
                    "label": target_label,
                    # 0 keeps the column numeric for downstream tooling. It is NOT a
                    # measurement: `decision` says eval_failed, and this value never
                    # reaches the accept/reject comparison -- the step returned early.
                    "score": 0,
                    "threshold": drop_tolerance,
                    "decision": "eval_failed",
                },
            )
            return step_counter + 1

        # Accept/reject
        old_acc = last_score[target_label]
        # drop_tolerance is specified in absolute percentage points (e.g., 1.5)
        allowed_drop = float(drop_tolerance)
        if new_acc + 1e-12 < (old_acc - allowed_drop):
            decision = "rejected"
            print(f"  [decision] {target_label}: REJECT (old={old_acc:.4f} new={new_acc:.4f})")
            try:
                shutil.rmtree(tmp_model_dir)
            except Exception:
                pass
        else:
            decision = "accepted"
            print(f"  [decision] {target_label}: ACCEPT (old={old_acc:.4f} new={new_acc:.4f})")
            _prev = str(working_dir) + ".pre_accept"
            try:
                if os.path.isdir(_prev):
                    shutil.rmtree(_prev)
                if os.path.isdir(str(working_dir)):
                    os.rename(str(working_dir), _prev)   # keep old copy until swap completes
                shutil.move(tmp_model_dir, str(working_dir))
            except Exception as e:
                # Restore the previous working copy so the run's state matches the CSV,
                # then fail loudly: a swallowed error here silently loses the model
                # while the CSV records the step as applied.
                if not os.path.isdir(str(working_dir)) and os.path.isdir(_prev):
                    os.rename(_prev, str(working_dir))
                raise RuntimeError(f"accept-path swap failed for {target_label}: {e}")
            else:
                shutil.rmtree(_prev, ignore_errors=True)
            thresholds[target_label] = min(thresholds[target_label], new_acc)
            # Keep last_score as the minimum of all accepted accuracies so far
            last_score[target_label] = min(last_score[target_label], new_acc) #last_score[target_label] = new_acc

        # Log per-step result
        _append_csv_row(
            results_csv,
            {
                "step_idx": step_counter,
                "stage": current_stage,
                "op": "merge",
                "component": group_name,
                "layer": int(layer),
                "label": target_label,
                "score": new_acc,
                "threshold": drop_tolerance,
                "decision": decision,
            },
        )
        return step_counter + 1

    
    # Walk merges_only in sorted order, chunking contiguous (layer, group) blocks within same stage.
    for idx, row in merges_only.iterrows():
        layer = int(row["layer"])
        group_name = str(row["group"])
        stage = int(row.get("stage", 1))
        
        if current_layer is None:
            current_layer = layer
            current_group = group_name
            current_stage = stage
            current_indices = [idx]
            continue
        if layer == current_layer and group_name == current_group and stage == current_stage:
            current_indices.append(idx)
        else:
            step_counter = _process_block(current_layer, current_group, current_indices, step_counter)
            current_layer = layer
            current_group = group_name
            current_stage = stage
            current_indices = [idx]

    # Process the final block
    if current_layer is not None and current_indices:
        step_counter = _process_block(current_layer, current_group or "", current_indices, step_counter)

    # Final evaluation summary for the target
    if eval_enabled:
        final_acc = evaluate_with_retry(
            f"{target_label} final",
            model_path=str(working_dir),
            task_or_domain=target_domain_or_task,
            gpu_ids=gpu_ids,
            timeout_minutes=timeout_minutes,
            batch_size=batch_size,
            temperature=temperature,
            output_dir=os.path.join(output_dir, target_label, "final"),
        )
        if final_acc is None:
            print(f"\n[final] {target_label}: evaluation failed after retries; no score.")
        else:
            print(f"\n[final] {target_label} score: {final_acc:.4f}")
    else:
        print("\n[final] eval disabled; skipping final score.")


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
    parser.add_argument("--initial_baseline", type=float, default=None, help="Force an initial baseline score (skip calculation/lookup).")
    parser.add_argument("--force-calc-baseline", action="store_true", help="Force calculation of baseline even if hardcoded.")
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
        "--enable-scaling",
        dest="enable_scaling",
        action="store_true",
        help=(
            "Rescale merged subcomponents to match the original per-layer std devs. "
            "Disabled by default to avoid extra profiling."
        ),
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
        initial_baseline=args.initial_baseline,
        force_calc_baseline=args.force_calc_baseline,
        eval_split=args.eval_split,
        enable_scaling=args.enable_scaling,
        merge_device=args.merge_device,
    )


if __name__ == "__main__":
    main()
