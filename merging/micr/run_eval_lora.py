#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-model merge-and-evaluate runner (LoRA Adapter Version).

- Loads a Base Model (Llama-3.1-8B) and multiple LoRA adapters.
- Reads an ops CSV (merge operations).
- Applies merges *to the LoRA adapters* (averaging A and B matrices) with std scaling.
- For evaluation, merges the modified adapter into the base model and evaluates.
"""

import os
import sys
import json
import csv
import shutil
import re
import argparse
import tempfile
import copy
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

from merge_tools.micr.eval_retry import fallback_is_redundant as _fallback_is_redundant  # type: ignore
from merge_tools.micr.eval_retry import (  # type: ignore
    evaluate_baseline_blocking as _evaluate_baseline_blocking,
    evaluate_with_retry as _evaluate_with_retry,
)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# The evaluation layer now lives in this repo: micr/eval_harness.py. It builds the same
# argv and parses scores with the same regexes as the former unified-llm-eval checkout
# (verified: 9/9 tasks byte-identical), but runs in the current interpreter's environment
# -- no conda env names, no UNIFIED_LLM_EVAL_ROOT. The vendored math-evaluation-harness is
# still required for --domain math; point MICR_MATH_HARNESS_DIR at it.
from merge_tools.micr import eval_harness  # type: ignore
from merge_tools.micr.eval_harness import EnvironmentManager, TASK_REGISTRY  # type: ignore
from merge_tools.micr.baselines import get_baseline  # type: ignore

# Constants
# HF repo id resolves via HF_HOME anywhere; override with the env var.
BASE_MODEL_PATH = os.environ.get("MICR_LORA_BASE_MODEL", "meta-llama/Llama-3.1-8B")
ADAPTER_ROOT = os.environ.get("MICR_LORA_ADAPTER_ROOT", "/scratch/shared_dir/lora/llama")

# Mapping from CSV Label -> Adapter Directory Name
LABEL_TO_ADAPTER_DIR = {
    "fin-llama3.1-8b": "finance_sanitized",
    "calme-2.3-legalkit-8b": "legal_sanitized",
    "Llama-3-8B-UltraMedical": "medical_sanitized",
    "Llama-SafetyGuard-Content-Binary": "toxicity_sanitized",
    "Llama-3.1-8B-Instruct-multi-truth-judge": "truthfulness_sanitized",
    "finance": "finance_sanitized",
    "legal": "legal_sanitized",
    "medical": "medical_sanitized",
    "toxicity": "toxicity_sanitized",
    "truthfulness": "truthfulness_sanitized",
    "finance_merged": "finance_merged",
    "legal_merged": "legal_merged",
    "medical_merged": "medical_merged",
    "toxicity_merged": "toxicity_merged",
    "truthfulness_merged": "truthfulness_merged",
}

# Mapping: Component Name -> LoRA Module Name
COMPONENT_TO_LORA_MODULE = {
    "attn_q": "q_proj",
    "attn_k": "k_proj",
    "attn_v": "v_proj",
    "attn_o": "o_proj",
    "mlp_gate": "gate_proj",
    "mlp_up": "up_proj",
    "mlp_down": "down_proj",
}

# Component grouping
COMPONENT_TO_GROUP = {
    "mlp": "mlp", "mlp_gate": "mlp", "mlp_up": "mlp", "mlp_down": "mlp",
    "attn_q": "attn", "attn_k": "attn", "attn_v": "attn", "attn_o": "attn",
}

# Domain -> Task Registry
DOMAIN_TO_REGISTRY_TASK = {
    "medical": "mmlu_professional_medicine",
    "finance": "mmlu_econometrics",
    "legal": "mmlu_professional_law",
    "toxicity": "sst2",
    "truthfulness": "truthfulqa_mc2",
    "math": "gsm8k-cot",
    "code": "humaneval",
    "coder": "humaneval",
}

# Baseline scores are looked up from the gaussian profiler output via
# merge_tools.micr.baselines.get_baseline (single source of truth); the former
# HARDCODED_BASELINE_SCORES dict was removed. Baselines are keyed by evaluation
# split and scores from different splits are not comparable, so lookups only
# match this runner's split (--eval_split).

def sanitize_adapter_name(name: str) -> str:
    """Sanitize adapter name by replacing dots with underscores."""
    return name.replace(".", "_")

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
        "timestamp", "step_idx", "stage", "op", "component", 
        "layer", "label", "score", "threshold", "decision"
    ]
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists or write_header:
            writer.writeheader()
        stamped = dict(row)
        stamped["timestamp"] = datetime.now(timezone.utc).isoformat()
        writer.writerow(stamped)

def resolve_adapter_path(label: str) -> Optional[str]:
    """Resolves a model label to a full adapter path."""
    if label in LABEL_TO_ADAPTER_DIR:
        dirname = LABEL_TO_ADAPTER_DIR[label]
        full_path = os.path.join(ADAPTER_ROOT, dirname)
        if os.path.exists(full_path):
            return full_path
    
    direct_path = os.path.join(ADAPTER_ROOT, label)
    if os.path.exists(direct_path):
        return direct_path
        
    sanitized = f"{label}_sanitized"
    sanitized_path = os.path.join(ADAPTER_ROOT, sanitized)
    if os.path.exists(sanitized_path):
        return sanitized_path

    return None

def load_base_model_with_adapters(
    base_path: str, 
    adapter_labels: Set[str],
    device: str = "auto"
) -> PeftModel:
    """
    Loads the base model and attaches all specified adapters.
    """
    print(f"[load] Loading base model: {base_path} on {device}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    
    first_label = list(adapter_labels)[0]
    first_path = resolve_adapter_path(first_label)
    
    if not first_path:
        raise ValueError(f"Could not resolve adapter path for {first_label}")
        
    print(f"[load] Loading initial adapter: {first_label} from {first_path}")
    model = PeftModel.from_pretrained(
        base_model, 
        first_path, 
        adapter_name=sanitize_adapter_name(first_label),
        is_trainable=True
    )
    
    for lbl in adapter_labels:
        if lbl == first_label:
            continue
        path = resolve_adapter_path(lbl)
        if path:
            print(f"[load] Loading adapter: {lbl} from {path}")
            model.load_adapter(path, adapter_name=sanitize_adapter_name(lbl))
        else:
            print(f"[warning] Could not resolve adapter path for {lbl}, skipping.")

    return model

def get_model_layers(model: PeftModel):
    """
    Robustly retrieve the layers module list from the PeftModel.
    Handles different nesting levels of LlamaForCausalLM / LlamaModel.
    """
    obj = model.base_model
    # Traverse down .model attributes to find the one containing .layers
    # Typical structure: PeftModel -> LoraModel -> LlamaForCausalLM -> LlamaModel -> layers
    
    # Attempt to go down 3 levels max
    for _ in range(3):
        if hasattr(obj, "layers"):
            return obj.layers
        if hasattr(obj, "model"):
            obj = obj.model
        else:
            break
            
    if hasattr(obj, "layers"):
        return obj.layers
        
    raise AttributeError(f"Could not find 'layers' attribute in model of type {type(model)}")

def get_adapter_component_stds(
    model: PeftModel, 
    adapter_name: str
) -> Dict[str, Dict[str, float]]:
    """
    Computes std deviation for LoRA A and B weights across all layers for a given adapter.
    Returns: { "layer_idx.component_name": { "A": std_a, "B": std_b } }
    """
    print(f"[profile] Computing stds for adapter: {adapter_name}")
    stds = {}
    
    try:
        layers = get_model_layers(model)
        
        for i, layer_module in enumerate(layers):
            # Check all supported components
            for comp_name, lora_mod_name in COMPONENT_TO_LORA_MODULE.items():
                # Determine submodule
                if "attn" in comp_name:
                    sub = layer_module.self_attn
                elif "mlp" in comp_name:
                    sub = layer_module.mlp
                else:
                    continue
                
                lora_linear = getattr(sub, lora_mod_name, None)
                if lora_linear and hasattr(lora_linear, "lora_A") and adapter_name in lora_linear.lora_A:
                    wa = lora_linear.lora_A[adapter_name].weight
                    wb = lora_linear.lora_B[adapter_name].weight
                    
                    stds[f"{i}.{comp_name}"] = {
                        "A": wa.std().item(),
                        "B": wb.std().item()
                    }
                    
    except Exception as e:
        print(f"[profile] Error computing stds: {e}")
        import traceback
        traceback.print_exc()
        
    return stds

def _n_way_average_lora_into_target(
    model: PeftModel,
    target_layer_idx: int,
    component_name: str, 
    target_adapter: str,
    donor_adapters: List[str], 
    target_stds: Optional[Dict[str, float]] = None # { "A": float, "B": float }
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Averages LoRA weights (A and B matrices) for a specific component with optional scaling.
    Returns the OLD weights (A, B) for rollback purposes.
    """
    lora_module_name = COMPONENT_TO_LORA_MODULE.get(component_name)
    if not lora_module_name:
        raise ValueError(f"Unknown component: {component_name}")

    try:
        layers = get_model_layers(model)
        layer_module = layers[target_layer_idx]
        
        if "attn" in component_name:
            sub = layer_module.self_attn
        elif "mlp" in component_name:
            sub = layer_module.mlp
        else:
            raise ValueError(f"Unknown component group: {component_name}")

        lora_linear = getattr(sub, lora_module_name, None)
        if lora_linear is None:
            raise ValueError(f"Module {lora_module_name} not found")
            
        if not hasattr(lora_linear, "lora_A") or not hasattr(lora_linear, "lora_B"):
            print(f"[skip] {component_name} at layer {target_layer_idx} is not a LoRA layer")
            # Return dummy tensors if skipped
            return (torch.tensor([]), torch.tensor([]))

        # Save old weights for rollback
        old_A = lora_linear.lora_A[target_adapter].weight.data.clone()
        old_B = lora_linear.lora_B[target_adapter].weight.data.clone()

        # Gather weights
        A_weights = []
        B_weights = []
        
        for donor in donor_adapters:
            if donor not in lora_linear.lora_A:
                print(f"[warning] Donor {donor} missing in lora_A at {target_layer_idx}.{component_name}")
                continue
            
            A_weights.append(lora_linear.lora_A[donor].weight.data)
            B_weights.append(lora_linear.lora_B[donor].weight.data)
            
        if not A_weights:
            return (old_A, old_B)

        # Average
        avg_A = torch.mean(torch.stack(A_weights), dim=0)
        avg_B = torch.mean(torch.stack(B_weights), dim=0)
        
        # Scaling
        if target_stds:
            std_A_tgt = target_stds.get("A")
            std_B_tgt = target_stds.get("B")
            
            if std_A_tgt is not None:
                curr_std_A = avg_A.std()
                if curr_std_A > 1e-9:
                    avg_A = avg_A * (std_A_tgt / curr_std_A)
                    
            if std_B_tgt is not None:
                curr_std_B = avg_B.std()
                if curr_std_B > 1e-9:
                    avg_B = avg_B * (std_B_tgt / curr_std_B)
        
        # Update target
        lora_linear.lora_A[target_adapter].weight.data.copy_(avg_A)
        lora_linear.lora_B[target_adapter].weight.data.copy_(avg_B)
        
        return (old_A, old_B)
        
    except Exception as e:
        print(f"Error averaging LoRA {component_name} at layer {target_layer_idx}: {e}")
        raise e

def restore_lora_weights(
    model: PeftModel,
    target_layer_idx: int,
    component_name: str,
    target_adapter: str,
    old_weights: Tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Restores saved weights to the adapter."""
    if not old_weights or old_weights[0].numel() == 0:
        return

    old_A, old_B = old_weights
    lora_module_name = COMPONENT_TO_LORA_MODULE.get(component_name)
    
    layers = get_model_layers(model)
    layer_module = layers[target_layer_idx]
    
    if "attn" in component_name:
        sub = layer_module.self_attn
    elif "mlp" in component_name:
        sub = layer_module.mlp
    else:
        return

    lora_linear = getattr(sub, lora_module_name, None)
    if lora_linear:
        lora_linear.lora_A[target_adapter].weight.data.copy_(old_A)
        lora_linear.lora_B[target_adapter].weight.data.copy_(old_B)

def save_merged_model_for_eval(
    model: PeftModel,
    target_adapter: str,
    tmp_root: str,
    base_model_path: str
) -> str:
    """
    Merges the target adapter into the base model and saves the FULL model for evaluation.
    """
    os.makedirs(tmp_root, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=tmp_root)
    
    try:
        model.set_adapter(target_adapter)
        model.merge_adapter() 
        
        model.base_model.save_pretrained(tmp_dir)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.save_pretrained(tmp_dir)
        
        model.unmerge_adapter()
        
    except Exception as e:
        print(f"[save] Error during merge/save: {e}")
        try:
            model.unmerge_adapter()
        except:
            pass
        raise e
        
    return tmp_dir

def create_unified_eval_context(gpu_ids=None, timeout_minutes=15, batch_size=32, temperature=0.0, output_dir="./evaluation_results"):
    if EnvironmentManager is None:
        return None, None, None
        
    if gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids).strip()
    
    # Avoid VRAM fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
    eval_settings = {
        "gpu_ids": "" if gpu_ids is None else str(gpu_ids),
        "timeout_minutes": int(timeout_minutes),
        "batch_size": int(batch_size),
        "temperature": float(temperature),
        "runs_per_eval": 1,
        "output_dir": output_dir,
        "use_vllm": True,
        # Restrict vLLM memory usage to avoid OOM on shared GPUs or with high fragmentation
        # 0.6 * 140GB ~= 84GB, plenty for 8B model (needs ~16GB weights + cache)
        "vllm_gpu_memory_utilization": 0.4, 
    }
    
    env_config = {"math_harness_dir": eval_harness.math_harness_dir()}
    env_manager = EnvironmentManager()
    return env_manager, env_config, eval_settings
def evaluate_model_for_task(model_path, task_or_domain, gpu_ids="0", allow_fallback=True, **kwargs):
    """Returns the measured score, or None when no score was produced.

    A returned 0.0 always means the model genuinely scored zero.
    """
    if TASK_REGISTRY is None:
        return None

    env_manager, env_config, eval_settings = create_unified_eval_context(gpu_ids=gpu_ids, **kwargs)
    
    registry_key = DOMAIN_TO_REGISTRY_TASK.get(task_or_domain, task_or_domain)
    task_info = TASK_REGISTRY.get(registry_key)
    if not task_info:
        print(f"Task {registry_key} not found")
        return None
        
    EvaluatorClass = task_info["evaluator"]
    evaluator = EvaluatorClass(env_manager, env_config, eval_settings)
    
    model_config = {"model_name": Path(model_path).name, "path": model_path, "location": "local"}
    
    print(f"[eval] Evaluating on {registry_key} with vLLM...")
    result = evaluator.evaluate(model_config, registry_key, run_id=1)
    
    if result.get("status") == "SUCCESS":
        try:
            return float(str(result["score"]).replace("%", ""))
        except:
            pass # fallback if parsing fails or status misleading
    
    print(f"[eval] vLLM Failed: {result.get('error_log')}")
    
    # Retry without vLLM -- only when the command would actually differ.
    if (allow_fallback and eval_settings.get("use_vllm", False)
            and not _fallback_is_redundant(EvaluatorClass, env_manager, env_config,
                                           eval_settings, model_path, registry_key)):
        print("[eval] Retrying without vLLM (using HF)...")
        eval_settings["use_vllm"] = False
        try:
            evaluator_retry = EvaluatorClass(env_manager, env_config, eval_settings)
            result_retry = evaluator_retry.evaluate(model_config, registry_key, run_id=2)
            
            if result_retry.get("status") == "SUCCESS":
                try:
                    return float(str(result_retry["score"]).replace("%", ""))
                except:
                    pass
            print(f"[eval] Retry Failed: {result_retry.get('error_log')}")
        except Exception as e:
            print(f"[eval] Retry Exception: {e}")

    return None

# The retry policy lives in micr/eval_retry.py so all four runners share one
# definition of what a failed evaluation means.
def evaluate_with_retry(label, *args, **kwargs):
    return _evaluate_with_retry(evaluate_model_for_task, label, *args, **kwargs)


def evaluate_baseline_blocking(label, *args, **kwargs):
    return _evaluate_baseline_blocking(evaluate_model_for_task, label, *args, **kwargs)


def load_ops_csvs_for_target(ops_csv=None, ops_step_csvs_dir=None, target_label=""):
    if ops_csv:
        df = pd.read_csv(ops_csv)
        df["stage"] = 1
        return df
    if ops_step_csvs_dir:
        safe = str(target_label).replace("/", "_")
        s1 = Path(ops_step_csvs_dir) / f"ops_step1_{safe}.csv"
        s2 = Path(ops_step_csvs_dir) / f"ops_step2_{safe}.csv"
        dfs = []
        if s1.exists(): dfs.append(pd.read_csv(s1).assign(stage=1))
        if s2.exists(): dfs.append(pd.read_csv(s2).assign(stage=2))
        if dfs: return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

def run_lora_pipeline(
    ops_csv: str,
    ops_step_csvs_dir: str,
    target_label: str,
    domain: str,
    gpu_ids: str,
    eval_enabled: bool,
    **kwargs
):
    # 1. Load Ops
    ops_df = load_ops_csvs_for_target(ops_csv, ops_step_csvs_dir, target_label)
    if ops_df.empty:
        print("No operations found.")
        return

    merges = ops_df[ops_df["op"] == "merge"].copy()
    if merges.empty:
        print("No merge operations.")
        return

    # 2. Identify all required adapters
    required_labels = set()
    required_labels.add(target_label)
    for _, row in merges.iterrows():
        models_str = str(row.get("models", ""))
        for part in models_str.split(","):
            lbl = part.split(":")[0].strip()
            if lbl:
                required_labels.add(lbl)
    print(f"Required Adapters: {required_labels}")

    # 3. Load Model + Adapters
    # Force CPU to avoid VRAM contention with vLLM during eval
    print("[resource] Loading model to CPU to save VRAM for evaluation...")
    model = load_base_model_with_adapters(BASE_MODEL_PATH, required_labels, device="cpu")
    
    # 4. Profile Target Adapter (for scaling)
    target_stds = get_adapter_component_stds(model, sanitize_adapter_name(target_label))
    
    # 5. Baseline Evaluation (Target Only)
    current_acc = 0.0
    tmp_eval_root = os.path.join(kwargs.get("working_root", "."), "tmp_eval_lora")
    results_csv = kwargs.get("results_csv", "lora_results.csv")
    timeout_minutes = int(kwargs.get("timeout_minutes", 15))
    batch_size = int(kwargs.get("batch_size", 32))
    temperature = float(kwargs.get("temperature", 0.0))
    output_dir = str(kwargs.get("output_dir", "./evaluation_results"))
    eval_split = str(kwargs.get("eval_split", "full"))

    if eval_enabled:
        print(f"[baseline] Measuring baseline for {target_label}...")
        tmp_dir = save_merged_model_for_eval(
            model, 
            sanitize_adapter_name(target_label), 
            tmp_eval_root,
            BASE_MODEL_PATH
        )
        base_acc = evaluate_baseline_blocking(
            target_label,
            tmp_dir, 
            domain, 
            gpu_ids=gpu_ids, 
            timeout_minutes=timeout_minutes,
            batch_size=batch_size,
            temperature=temperature,
            output_dir=os.path.join(output_dir, str(target_label), "baseline"),
        )
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"[baseline] Measured: {base_acc:.4f}")
        current_acc = base_acc
    else:
        # Eval disabled: seed the reported score from a stored baseline measured
        # on this runner's split, if one exists; scores from other splits are
        # not comparable and are ignored.
        _profiler_baseline = get_baseline(target_label, split=eval_split)
        current_acc = float(_profiler_baseline if _profiler_baseline is not None else 0.0)

    # 6. Processing Loop
    step_idx = 0
    merges["group"] = merges["component"].apply(lambda c: COMPONENT_TO_GROUP.get(c, c))
    
    for idx, row in merges.iterrows():
        layer = int(row["layer"])
        comp = str(row["component"])
        stage = row.get("stage", 1)
        
        models_str = str(row.get("models", ""))
        participants = []
        for m in models_str.split(","):
            participants.append(m.split(":")[0].strip())
            
        if target_label not in participants:
            continue
            
        print(f"\n[step {step_idx}] Merge {comp} at Layer {layer} (Stage {stage})")
        
        # Get target scaling stats for this component
        comp_stds = target_stds.get(f"{layer}.{comp}") # { "A": ..., "B": ... }

        sanitized_participants = [sanitize_adapter_name(p) for p in participants]
        sanitized_target = sanitize_adapter_name(target_label)

        # APPLY MERGE & SAVE OLD WEIGHTS
        try:
            old_weights = _n_way_average_lora_into_target(
                model, 
                target_layer_idx=layer, 
                component_name=comp, 
                target_adapter=sanitized_target, 
                donor_adapters=sanitized_participants, 
                target_stds=comp_stds
            )
        except Exception as e:
            print(f"  [error] Merge failed: {e}")
            continue

        # EVALUATE
        decision = "accepted"
        if eval_enabled:
            tmp_dir = save_merged_model_for_eval(
                model, 
                sanitized_target, 
                tmp_eval_root,
                BASE_MODEL_PATH
            )
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            new_acc = evaluate_with_retry(
                f"{target_label} step {step_idx}",
                tmp_dir, 
                domain, 
                gpu_ids=gpu_ids, 
                timeout_minutes=timeout_minutes,
                batch_size=batch_size,
                temperature=temperature,
                output_dir=os.path.join(output_dir, str(target_label), f"step_{step_idx}"),
            )
            
            shutil.rmtree(tmp_dir, ignore_errors=True)

            if new_acc is None:
                # No score was produced, so this merge was never judged. Roll it
                # back and record the failure rather than a fabricated rejection.
                decision = "eval_failed"
                print("  [decision] EVAL FAILED after retries (rolling back; adapter unchanged)")
                restore_lora_weights(
                    model,
                    layer,
                    comp,
                    sanitized_target,
                    old_weights
                )
            else:
                print(f"  [eval] Old: {current_acc:.2f}, New: {new_acc:.2f}")

                drop_tol = kwargs.get("drop_tolerance", 2.0)
                if new_acc < (current_acc - drop_tol):
                    decision = "rejected"
                    print("  [decision] REJECTED (Rolling back)")
                    restore_lora_weights(
                        model,
                        layer,
                        comp,
                        sanitized_target, 
                        old_weights
                    )
                else:
                    current_acc = new_acc
        
        _append_csv_row(results_csv, {
            "step_idx": step_idx,
            "stage": stage,
            "op": "merge",
            "component": comp,
            "layer": layer,
            "label": target_label,
            # 0 on failure keeps the column numeric; `decision` carries the meaning.
            "score": 0 if decision == "eval_failed" else current_acc,
            "decision": decision
        })
        step_idx += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops_csv", help="Operations CSV")
    parser.add_argument("--ops_step_csvs_dir", help="Step CSVs dir")
    parser.add_argument("--target_label", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument(
        "--eval_split",
        default="full",
        help=(
            "Name of the evaluation split this run gates on (e.g. 'full', or a "
            "MICR data split). Stored baselines are split-keyed and only reused "
            "on an exact split match."
        ),
    )
    parser.add_argument("--working_root", default="./lora_work")
    parser.add_argument("--results_csv", default="lora_results.csv")
    args = parser.parse_args()
    
    run_lora_pipeline(
        ops_csv=args.ops_csv,
        ops_step_csvs_dir=args.ops_step_csvs_dir,
        target_label=args.target_label,
        domain=args.domain,
        gpu_ids=args.gpu_ids,
        eval_enabled=not args.no_eval,
        working_root=args.working_root,
        results_csv=args.results_csv,
        eval_split=args.eval_split,
    )

if __name__ == "__main__":
    main()
