import os
import sys
import yaml
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tempfile
import shutil
import csv
from datetime import datetime, timezone
import argparse




from transformers import AutoModelForCausalLM
from transformers import AutoConfig

# MICR imports the tensor helpers below, so this module must not drag a external checkout
# onto sys.path. The evaluation layer lives in micr/eval_harness.py.
_MERGE_TOOLS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import types as _types
if "merge_tools" not in sys.modules:
    _mt_pkg = _types.ModuleType("merge_tools")
    _mt_pkg.__path__ = [_MERGE_TOOLS_ROOT]
    sys.modules["merge_tools"] = _mt_pkg
from merge_tools.micr import eval_harness  # type: ignore
from merge_tools.micr.eval_harness import EnvironmentManager, TASK_REGISTRY  # type: ignore


# --- Core Functions (adapted from gaussian_experiment.py) ---
# Dtype helpers
def _infer_model_dtype(path: str, fallback: torch.dtype = torch.bfloat16) -> torch.dtype:
    try:
        cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
        td = getattr(cfg, "torch_dtype", None)
        if isinstance(td, torch.dtype):
            return td
        if isinstance(td, str):
            td = td.replace("torch.", "")
            return getattr(torch, td) if hasattr(torch, td) else fallback
    except Exception:
        pass
    return fallback

def _load_model(path: str, device_hint: str = "cuda", dtype_hint: Optional[torch.dtype] = None) -> torch.nn.Module:
    target_dtype = dtype_hint or _infer_model_dtype(path)
    device = "cuda" if torch.cuda.is_available() and device_hint == "cuda" else "cpu"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    use_multi_gpu = device == "cuda" and ("," in visible)
    common_kwargs = dict(
        dtype=target_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if use_multi_gpu:
        model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", **common_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(path, **common_kwargs)
        if device == "cuda":
            model.to(device)
    model.eval()
    return model

def _get_layer_module(model: torch.nn.Module, layer_idx: int):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    raise RuntimeError("Unsupported model structure.")

def _collect_group_parameters(layer_module: torch.nn.Module, group: str) -> List[torch.nn.Parameter]:
    attn_module = getattr(layer_module, "self_attn", None)
    mlp_module = getattr(layer_module, "mlp", None)
    params: List[torch.nn.Parameter] = []
    if group in ("attn", "both") and attn_module:
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            if hasattr(getattr(attn_module, name, None), "weight"):
                params.append(getattr(attn_module, name).weight)
    if group in ("mlp", "both") and mlp_module:
        for name in ["gate_proj", "up_proj", "down_proj"]:
            if hasattr(getattr(mlp_module, name, None), "weight"):
                params.append(getattr(mlp_module, name).weight)
    return params

def _generate_noise_like_weight(weight: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        mean_val = weight.mean().item()
        std_val = weight.std().item()
        noise = torch.normal(mean=mean_val, std=std_val, size=weight.shape, device=weight.device)
        return noise.to(dtype=weight.dtype)

def _save_model_to_temp_dir(model: torch.nn.Module, source_model_path: str, tmp_root: str) -> str:
    """
    Saves the modified model to a temporary directory and copies tokenizer files.
    """
    tmp_dir = tempfile.mkdtemp(dir=tmp_root)
    try:
        # Force model tensors to the source model's dtype before saving
        target_dtype = _infer_model_dtype(source_model_path)
        model.to(dtype=target_dtype)
        if hasattr(model, "config"):
            model.config.torch_dtype = target_dtype
        model.save_pretrained(tmp_dir, safe_serialization=True)
    except TypeError:
        model.save_pretrained(tmp_dir)

    # Copy common tokenizer assets so downstream loaders (e.g., vLLM) can initialize
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
    for fname in tokenizer_files:
        src = os.path.join(source_model_path, fname)
        if os.path.exists(src):
            try:
                shutil.copy(src, tmp_dir)
            except Exception:
                pass
    return tmp_dir

def evaluate_model(model_path: str, task_name: str, env_manager, env_config: dict, eval_settings: dict) -> float:
    if TASK_REGISTRY is None or env_manager is None:
        print("Evaluation harness not available.")
        return 0.0
    
    TASK_NAME_TO_REGISTRY_KEY = {"coder": "humaneval", "math": "gsm8k-cot"}
    registry_task_name = TASK_NAME_TO_REGISTRY_KEY.get(task_name, task_name)
    
    task_info = TASK_REGISTRY.get(registry_task_name)
    if not task_info:
        print(f"Task '{registry_task_name}' not in registry.")
        return 0.0

    EvaluatorClass = task_info["evaluator"]
    evaluator = EvaluatorClass(env_manager, env_config, eval_settings)
    
    absolute_model_path = Path(model_path).resolve()
    model_config = {
        "model_name": os.path.basename(absolute_model_path),
        "path": str(absolute_model_path),
        "location": "local",
    }
    
    result_dict = evaluator.evaluate(model_config, registry_task_name, run_id=1)
    if result_dict.get("status") == "SUCCESS":
        try:
            return float(result_dict["score"].replace("%", ""))
        except Exception:
            return 0.0
    print(f"Evaluation failed: {result_dict.get('error_log', 'Unknown error')}")
    return 0.0


def _resolve_env_paths(env_config: dict, root: str) -> dict:
    if not isinstance(env_config, dict):
        return {}
    for key in ["math_harness_dir", "language_eval_dir"]:
        path = env_config.get(key)
        if path and not os.path.isabs(path):
            env_config[key] = os.path.normpath(os.path.join(root, path.replace("./", "")))
    return env_config


# --- Helpers for sweeping and logging ---

def _get_num_layers(model: torch.nn.Module) -> int:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    raise RuntimeError("Unsupported model structure.")


def _write_csv_row(csv_path: str, row: Dict[str, object]) -> None:
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
        "model",
        "task",
        "layer",
        "variant",
        "perturbation",
        "score",
        "decision",
        "threshold",
    ]
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists or write_header:
            writer.writeheader()
        stamped = dict(row)
        stamped["timestamp"] = datetime.now(timezone.utc).isoformat()
        writer.writerow(stamped)


def _build_candidates_from_profile(profile_csv_path: str) -> List[Tuple[int, str, float]]:
    """
    Read a gaussian sanity CSV and derive a candidate list:
    - Keep rows with layer >= 0 and variant in {"attn","mlp"}
    - Sort by score descending (large -> small)
    - Deduplicate by layer: only the first encounter of a given layer is kept
      (regardless of component), subsequent encounters for that layer are ignored
    Returns list of (layer_idx, variant, score).
    """
    candidates: List[Tuple[int, str, float]] = []
    try:
        with open(profile_csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows: List[Dict[str, str]] = list(reader)
    except Exception:
        return candidates
    filtered: List[Tuple[int, str, float]] = []
    for r in rows:
        try:
            layer = int(r.get("layer", "-1"))
            variant = str(r.get("variant", "")).strip()
            score = float(r.get("score", "0"))
            # accept attn/mlp/both; ignore 'none' or other entries
            if layer >= 0 and variant in ("attn", "mlp", "both"):
                filtered.append((layer, variant, score))
        except Exception:
            continue
    # Sort large -> small by score, keep one per (layer, variant) combo
    filtered.sort(key=lambda x: x[2], reverse=True)
    seen_pairs = set()
    for layer, variant, score in filtered:
        key = (layer, variant)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        candidates.append((layer, variant, score))
    return candidates

def _collect_params_for_groups(layer_module: torch.nn.Module, groups: List[str]) -> List[torch.nn.Parameter]:
    """Collect unique parameters across multiple groups for a layer."""
    unique: Dict[int, torch.nn.Parameter] = {}
    for g in groups:
        for p in _collect_group_parameters(layer_module, g):
            unique[id(p)] = p
    return list(unique.values())

def _merge_step_multi_groups_with_rollback(
    target_model_obj: torch.nn.Module,
    target_model_path: str,
    partner_model_path: str,
    layer_idx: int,
    groups: List[str],
    task: str,
    env_manager,
    env_config: dict,
    eval_settings: dict,
    tmp_dir_root: str,
    last_acc: float,
    drop_threshold: float,
) -> Tuple[float, bool]:
    """
    Merge multiple groups (e.g., ['attn','mlp']) into the layer in one step,
    evaluate once, and rollback all if accuracy drop exceeds threshold.
    Returns (new_acc, accepted).
    """
    # Backup params for all requested groups
    layer_module = _get_layer_module(target_model_obj, layer_idx)
    params_before = _collect_params_for_groups(layer_module, groups)
    backups: List[torch.Tensor] = [p.data.cpu().clone() for p in params_before]

    # Apply merges in-place for each group
    for g in groups:
        _average_between_models_inplace(
            model_a=target_model_obj,
            model_a_path=target_model_path,
            model_b_path=partner_model_path,
            layer_idx=layer_idx,
            group=g,
        )

    # Save temp and evaluate once
    tmp_model_dir = _save_model_to_temp_dir(target_model_obj, target_model_path, tmp_dir_root)
    try:
        acc = evaluate_model(tmp_model_dir, task, env_manager, env_config, eval_settings)
        if acc == 0.0 and isinstance(eval_settings, dict) and eval_settings.get("use_vllm", True):
            eval_settings2 = dict(eval_settings)
            eval_settings2["use_vllm"] = False
            acc = evaluate_model(tmp_model_dir, task, env_manager, env_config, eval_settings2)
    finally:
        shutil.rmtree(tmp_model_dir)

    # Decision (treat drop_threshold as fraction-of-100)
    allowed_drop = 100.0 * drop_threshold
    if acc + 1e-12 < (last_acc - allowed_drop):
        # Roll back all touched params
        layer_module_after = _get_layer_module(target_model_obj, layer_idx)
        params_after = _collect_params_for_groups(layer_module_after, groups)
        with torch.no_grad():
            for p, backup in zip(params_after, backups):
                p.data.copy_(backup.to(dtype=p.data.dtype, device=p.data.device))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return acc, False

    return acc, True


def _perturb_and_eval(
    base_model_path: str,
    layer_idx: int,
    group: str,
    perturbation: str,
    task: str,
    env_manager,
    env_config: dict,
    eval_settings: dict,
    tmp_dir_root: str,
) -> float:
    model = _load_model(base_model_path)
    layer_module = _get_layer_module(model, layer_idx)
    params = _collect_group_parameters(layer_module, group)
    with torch.no_grad():
        for p in params:
            noise = _generate_noise_like_weight(p.data)
            if perturbation == "add":
                p.add_(noise)
            elif perturbation == "avg":
                p.data.add_(noise).div_(2.0)
            elif perturbation == "replace":
                p.data.copy_(noise)
            else:
                raise ValueError(f"Unknown perturbation: {perturbation}")
    tmp_model_dir = _save_model_to_temp_dir(model, base_model_path, tmp_dir_root)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        acc = evaluate_model(tmp_model_dir, task, env_manager, env_config, eval_settings)
    finally:
        shutil.rmtree(tmp_model_dir)
    return acc


def _average_between_models_and_eval(
    model_a_path: str,
    model_b_path: str,
    layer_idx: int,
    group: str,
    task: str,
    env_manager,
    env_config: dict,
    eval_settings: dict,
    tmp_dir_root: str,
) -> float:
    """
    Load model A and model B on CPU, average the selected group's parameters
    for the specified layer into model A (compute on GPU in fp32 for stability,
    then downcast back to low dtype), save to a temp dir, and evaluate.
    """
    # Merge on CPU to avoid double GPU memory usage; preserve source dtype
    target_low_dtype = _infer_model_dtype(model_a_path)
    model_a = _load_model(model_a_path, device_hint="cpu", dtype_hint=target_low_dtype)
    model_b = _load_model(model_b_path, device_hint="cpu", dtype_hint=target_low_dtype)
    layer_a = _get_layer_module(model_a, layer_idx)
    layer_b = _get_layer_module(model_b, layer_idx)
    params_a = _collect_group_parameters(layer_a, group)
    params_b = _collect_group_parameters(layer_b, group)
    if len(params_a) != len(params_b):
        raise RuntimeError(f"Param count mismatch for layer {layer_idx}, group {group}: {len(params_a)} vs {len(params_b)}")
    if not torch.cuda.is_available():
        raise RuntimeError("Requested GPU merge but CUDA is not available.")
    with torch.no_grad():
        for pa, pb in zip(params_a, params_b):
            # Stream tensors to GPU in float32, average, then downcast to target low dtype on CPU
            a_cpu = pa.data
            b_cpu = pb.data
            a32 = a_cpu.to(device="cuda", dtype=torch.float32, non_blocking=True)
            b32 = b_cpu.to(device="cuda", dtype=torch.float32, non_blocking=True)
            out32 = 0.5 * (a32 + b32)
            out_low_cpu = out32.to(device=a_cpu.device, dtype=target_low_dtype, non_blocking=True)
            a_cpu.copy_(out_low_cpu)
            # Cleanup GPU memory promptly
            del a32, b32, out32
        torch.cuda.synchronize()
    # Ensure model config records low dtype for vLLM
    if hasattr(model_a, "config"):
        model_a.config.torch_dtype = target_low_dtype
    tmp_model_dir = _save_model_to_temp_dir(model_a, model_a_path, tmp_dir_root)
    del model_a
    del model_b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        acc = evaluate_model(tmp_model_dir, task, env_manager, env_config, eval_settings)
        # Auto-fallback to non-vLLM eval if engine init failed and harness supports the flag
        if acc == 0.0 and isinstance(eval_settings, dict) and eval_settings.get("use_vllm", True):
            eval_settings2 = dict(eval_settings)
            eval_settings2["use_vllm"] = False
            acc = evaluate_model(tmp_model_dir, task, env_manager, env_config, eval_settings2)
    finally:
        shutil.rmtree(tmp_model_dir)
    return acc


# --- GPU-f32 averaging for a single param pair (streams one tensor at a time) ---
def _avg_params_gpu32_(pa: torch.nn.Parameter, pb: torch.nn.Parameter, target_low_dtype: torch.dtype) -> None:
    a_cpu = pa.data
    b_cpu = pb.data
    a32 = a_cpu.to(device="cuda", dtype=torch.float32, non_blocking=True)
    b32 = b_cpu.to(device="cuda", dtype=torch.float32, non_blocking=True)
    out32 = 0.5 * (a32 + b32)
    out_low_cpu = out32.to(device=a_cpu.device, dtype=target_low_dtype, non_blocking=True)
    a_cpu.copy_(out_low_cpu)
    # cleanup GPU memory promptly
    del a32, b32, out32


def _average_between_models_inplace(
    model_a: torch.nn.Module,
    model_a_path: str,
    model_b_path: str,
    layer_idx: int,
    group: str,
) -> None:
    """
    Average params for a single (layer, group) into model_a in-place.
    Loads model_b on CPU at the appropriate dtype; streams only the required
    tensors to GPU in fp32 for numerically stable averaging.
    """
    target_low_dtype = _infer_model_dtype(model_a_path)
    model_b = _load_model(model_b_path, device_hint="cpu", dtype_hint=target_low_dtype)
    layer_a = _get_layer_module(model_a, layer_idx)
    layer_b = _get_layer_module(model_b, layer_idx)
    params_a = _collect_group_parameters(layer_a, group)
    params_b = _collect_group_parameters(layer_b, group)
    if len(params_a) != len(params_b):
        del model_b
        raise RuntimeError(f"Param count mismatch for layer {layer_idx}, group {group}: {len(params_a)} vs {len(params_b)}")
    if not torch.cuda.is_available():
        del model_b
        raise RuntimeError("Requested GPU merge but CUDA is not available.")
    with torch.no_grad():
        for pa, pb in zip(params_a, params_b):
            _avg_params_gpu32_(pa, pb, target_low_dtype)
        torch.cuda.synchronize()
    # Hint dtype on config for downstream loaders
    if hasattr(model_a, "config"):
        model_a.config.torch_dtype = target_low_dtype
    del model_b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- Merge step with rollback if accuracy drop exceeds threshold ---
def _merge_step_eval_with_rollback(
    target_model_obj: torch.nn.Module,
    target_model_path: str,
    partner_model_path: str,
    layer_idx: int,
    group: str,
    task: str,
    env_manager,
    env_config: dict,
    eval_settings: dict,
    tmp_dir_root: str,
    last_acc: float,
    drop_threshold: float,
) -> Tuple[float, bool]:
    """
    Merge one (layer, group) into target_model_obj, evaluate, and rollback if the
    accuracy drops by more than drop_threshold compared to last_acc.
    Returns (new_acc, accepted).
    """
    # Backup current params to allow rollback (CPU clones)
    layer_module = _get_layer_module(target_model_obj, layer_idx)
    params_before = _collect_group_parameters(layer_module, group)
    backups: List[torch.Tensor] = [p.data.cpu().clone() for p in params_before]

    # Apply in-place averaging using partner model weights
    _average_between_models_inplace(
        model_a=target_model_obj,
        model_a_path=target_model_path,
        model_b_path=partner_model_path,
        layer_idx=layer_idx,
        group=group,
    )
    # Save and evaluate
    tmp_model_dir = _save_model_to_temp_dir(target_model_obj, target_model_path, tmp_dir_root)
    try:
        acc = evaluate_model(tmp_model_dir, task, env_manager, env_config, eval_settings)
        # Heuristic fallback if harness supports use_vllm and returned 0.0
        if acc == 0.0 and isinstance(eval_settings, dict) and eval_settings.get("use_vllm", True):
            eval_settings2 = dict(eval_settings)
            eval_settings2["use_vllm"] = False
            acc = evaluate_model(tmp_model_dir, task, env_manager, env_config, eval_settings2)
    finally:
        shutil.rmtree(tmp_model_dir)

    # Decision: accept or rollback (treat drop_threshold as fraction-of-100)
    allowed_drop = 100.0 * drop_threshold
    if acc + 1e-12 < (last_acc - allowed_drop):
        # Roll back layer params
        layer_module_after = _get_layer_module(target_model_obj, layer_idx)
        params_after = _collect_group_parameters(layer_module_after, group)
        with torch.no_grad():
            for p, backup in zip(params_after, backups):
                p.data.copy_(backup.to(dtype=p.data.dtype, device=p.data.device))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return acc, False

    return acc, True


# --- Main Sanity Check Logic ---

def run_sanity_check(
    model_override: Optional[str] = None,
    config_path: str = "experiments/gaussian_experiment.yml",
    output_csv: str = "experiments/gaussian_sanity_results.csv",
    output_csv_a: Optional[str] = None,
    output_csv_b: Optional[str] = None,
    profile_csv_a: Optional[str] = None,
    profile_csv_b: Optional[str] = None,
    start_layer: Optional[int] = None,
    end_layer: Optional[int] = None,
    partner_model_override: Optional[str] = None,
    layers_csv: Optional[str] = None,
    groups_csv: str = "attn,mlp",
    sequential_merge: bool = False,
    attn_first: bool = False,
    drop_threshold: float = 0.025,
    task_a: str = "coder",
    task_b: str = "math",
    use_main_profile_for_both: bool = False,
):
    """
    Merge experiment: For the selected layers and groups, average parameters of
    model A and model B (same layer/group), save merged model, and evaluate.
    """
    print("--- Starting Merge (Layer-wise Averaging) Experiment ---")

    # --- 1. Configuration ---
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    algo_config = config["algorithm_settings"]
    eval_settings = config["evaluation_settings"]
    # Respect GPU selection from config (e.g., "3,4")
    gpu_ids = str(eval_settings.get("gpu_ids", "")).strip()
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        print(f"Using GPUs: {gpu_ids}")
    env_config = _resolve_env_paths(config.get("environment_config", {}), _MERGE_TOOLS_ROOT)
    env_config.setdefault("math_harness_dir", eval_harness.math_harness_dir())

    # --- 2. Setup ---
    if EnvironmentManager is None:
        raise RuntimeError("EnvironmentManager could not be imported.")
    env_manager = EnvironmentManager(env_config.get("harness_env"), env_config.get("languages_env"))
    tmp_dir_root = algo_config.get("tmp_dir", os.environ.get("MICR_TMP_ROOT", "/tmp/micr_tmp"))
    os.makedirs(tmp_dir_root, exist_ok=True)

    tasks = ["coder"] #change to math for math experiment
    variant_order = [v.strip() for v in (groups_csv or "attn,mlp").split(",") if v.strip()]

    # Prepare per-model CSVs (filenames encode threshold and profile basename)
    base_dir, base_file = os.path.split(output_csv)
    base_csv_root, base_ext = os.path.splitext(base_file or "topk.csv")
    threshold_tag = f"drop_{str(drop_threshold).replace('.', 'p')}"
    # Determine which profile path is considered "used" per model
    if use_main_profile_for_both:
        effective_profile_a = profile_csv_a or profile_csv_b or "-"
        effective_profile_b = effective_profile_a
    else:
        effective_profile_a = profile_csv_a or "-"
        effective_profile_b = profile_csv_b or "-"
    # Derive short tags from profile basenames
    def _prof_tag(p: str) -> str:
        if not p or p == "-":
            return "none"
        return os.path.splitext(os.path.basename(p))[0]
    prof_tag_a = _prof_tag(effective_profile_a)
    prof_tag_b = _prof_tag(effective_profile_b)
    # Defaults include threshold and profile tags
    csv_a_default = os.path.join(base_dir, f"{base_csv_root}_{threshold_tag}_profile_{prof_tag_a}{base_ext or '.csv'}")
    csv_b_default = os.path.join(base_dir, f"{base_csv_root}_{threshold_tag}_profile_{prof_tag_b}{base_ext or '.csv'}")
    csv_a_path = output_csv_a or csv_a_default
    csv_b_path = output_csv_b or csv_b_default

    for task in tasks:
        # Determine model path (override applies to both tasks if provided)
        model_a_path = model_override or algo_config["models_to_merge"][task]
        # Partner model path (from override or config under algorithm_settings.merge_partner)
        partner_cfg = (algo_config.get("merge_partner") or {})
        model_b_path = partner_model_override or partner_cfg.get(task)
        if not model_b_path:
            raise RuntimeError("No partner model provided. Use --partner_model or set algorithm_settings.merge_partner.<task> in YAML.")
        print(f"\n=== Task: {task} | Model A: {model_a_path} | Model B: {model_b_path} ===")

        # Baseline evaluation for both models
        print("Baseline: evaluating original models...")
        baseline_acc_a = evaluate_model(model_a_path, task_a, env_manager, env_config, eval_settings)
        print(f"  -> Baseline A: {baseline_acc_a:.4f}")
        _write_csv_row(
            csv_a_path,
            {
                "timestamp": datetime.utcnow().isoformat(),
                "model": model_a_path,
                "task": task_a,
                "layer": -1,
                "variant": "none",
                "perturbation": "baseline_model_a",
                "score": baseline_acc_a,
                "decision": "baseline",
                "threshold": drop_threshold,
            },
        )
        baseline_acc_b = evaluate_model(model_b_path, task_b, env_manager, env_config, eval_settings)
        print(f"  -> Baseline B: {baseline_acc_b:.4f}")
        _write_csv_row(
            csv_b_path,
            {
                "timestamp": datetime.utcnow().isoformat(),
                "model": model_b_path,
                "task": task_b,
                "layer": -1,
                "variant": "none",
                "perturbation": "baseline_model_b",
                "score": baseline_acc_b,
                "decision": "baseline",
                "threshold": drop_threshold,
            },
        )

        # Determine layer sweep range
        probe_model = _load_model(model_a_path)
        total_layers = _get_num_layers(probe_model)
        del probe_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if layers_csv:
            selected_layers = []
            for tok in layers_csv.split(","):
                tok = tok.strip()
                if tok:
                    try:
                        selected_layers.append(int(tok))
                    except ValueError:
                        raise RuntimeError(f"Invalid layer index: {tok}")
            selected_layers = [i for i in selected_layers if 0 <= i < total_layers]
        else:
            # Default to the full range if not provided
            sweep_start = 0 if start_layer is None else max(0, start_layer)
            sweep_end = total_layers - 1 if end_layer is None else min(end_layer, total_layers - 1)
            selected_layers = list(range(sweep_start, sweep_end + 1))

        print(f"Selected layers for merging: {selected_layers} (total_layers={total_layers})")
        # If profiles provided, build candidate sequences for each model
        candidates_a: Optional[List[Tuple[int, str, float]]] = None
        candidates_b: Optional[List[Tuple[int, str, float]]] = None
        if profile_csv_a:
            candidates_a = _build_candidates_from_profile(profile_csv_a)
            print("\nModel A candidates from profile (layer, variant, score) in desc order:")
            for la, va, sa in candidates_a:
                print(f"  {la:>3}  {va:<4}  {sa:.4f}")
        if profile_csv_b:
            candidates_b = _build_candidates_from_profile(profile_csv_b)
            print("\nModel B candidates from profile (layer, variant, score) in desc order:")
            for lb, vb, sb in candidates_b:
                print(f"  {lb:>3}  {vb:<4}  {sb:.4f}")
        # If requested, drive both models' sequences from the main (task_a) profile
        if use_main_profile_for_both:
            if candidates_a is not None:
                candidates_b = candidates_a
                print("\nUsing Model A profile for both A and B candidate sequences.")
            elif candidates_b is not None:
                candidates_a = candidates_b
                print("\nModel A profile not provided; using Model B profile for both sequences.")

        if sequential_merge:
            # Load base model once (CPU, low dtype), then cumulatively merge and evaluate after each step
            base_low_dtype = _infer_model_dtype(model_a_path)
            model_a_obj = _load_model(model_a_path, device_hint="cpu", dtype_hint=base_low_dtype)
            model_b_obj = _load_model(model_b_path, device_hint="cpu", dtype_hint=_infer_model_dtype(model_b_path))
            # Track accepted components per layer per model
            accepted_components_a: Dict[int, set] = {}
            accepted_components_b: Dict[int, set] = {}
            last_tmp_dir: Optional[str] = None
            try:
                # Profile-driven order if provided; else fall back to attn_first/layer order
                if candidates_a or candidates_b:
                    # Process Model A sequence
                    if candidates_a:
                        print("\n== Model A profile-driven sequence ==")
                        for layer_idx, variant, _ in candidates_a:
                            print(f"\n-- Layer {layer_idx} --\nVariant: {variant}")
                            acc_set = accepted_components_a.setdefault(layer_idx, set())
                            # Determine which groups to merge based on variant and what is already accepted
                            if variant == "both":
                                groups_to_merge = []
                                if "attn" not in acc_set:
                                    groups_to_merge.append("attn")
                                if "mlp" not in acc_set:
                                    groups_to_merge.append("mlp")
                            elif variant in ("attn", "mlp"):
                                groups_to_merge = [variant] if variant not in acc_set else []
                            else:
                                groups_to_merge = []
                            if not groups_to_merge:
                                print("  -> Skipping (no remaining components to merge for this layer).")
                                continue
                            if len(groups_to_merge) == 1:
                                new_acc_a, accepted_a = _merge_step_eval_with_rollback(
                                    target_model_obj=model_a_obj,
                                    target_model_path=model_a_path,
                                    partner_model_path=model_b_path,
                                    layer_idx=layer_idx,
                                    group=groups_to_merge[0],
                                    task=task_a,
                                    env_manager=env_manager,
                                    env_config=env_config,
                                    eval_settings=eval_settings,
                                    tmp_dir_root=tmp_dir_root,
                                    last_acc=baseline_acc_a,
                                    drop_threshold=drop_threshold,
                                )
                            else:
                                new_acc_a, accepted_a = _merge_step_multi_groups_with_rollback(
                                    target_model_obj=model_a_obj,
                                    target_model_path=model_a_path,
                                    partner_model_path=model_b_path,
                                    layer_idx=layer_idx,
                                    groups=groups_to_merge,
                                    task=task_a,
                                    env_manager=env_manager,
                                    env_config=env_config,
                                    eval_settings=eval_settings,
                                    tmp_dir_root=tmp_dir_root,
                                    last_acc=baseline_acc_a,
                                    drop_threshold=drop_threshold,
                                )
                            _write_csv_row(
                                csv_a_path,
                                {
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "model": model_a_path,
                                    "task": task_a,
                                    "layer": layer_idx,
                                    "variant": variant,
                                    "perturbation": "avg-merge-seq" if accepted_a else "avg-merge-skip",
                                    "score": new_acc_a,
                                    "decision": "accepted" if accepted_a else "rejected",
                                    "threshold": drop_threshold,
                                },
                            )
                            if accepted_a:
                                baseline_acc_a = new_acc_a
                                for g in groups_to_merge:
                                    acc_set.add(g)
                    # Process Model B sequence
                    if candidates_b:
                        print("\n== Model B profile-driven sequence ==")
                        for layer_idx, variant, _ in candidates_b:
                            print(f"\n-- Layer {layer_idx} --\nVariant: {variant}")
                            acc_set_b = accepted_components_b.setdefault(layer_idx, set())
                            if variant == "both":
                                groups_to_merge_b = []
                                if "attn" not in acc_set_b:
                                    groups_to_merge_b.append("attn")
                                if "mlp" not in acc_set_b:
                                    groups_to_merge_b.append("mlp")
                            elif variant in ("attn", "mlp"):
                                groups_to_merge_b = [variant] if variant not in acc_set_b else []
                            else:
                                groups_to_merge_b = []
                            if not groups_to_merge_b:
                                print("  -> Skipping (no remaining components to merge for this layer).")
                                continue
                            if len(groups_to_merge_b) == 1:
                                new_acc_b, accepted_b = _merge_step_eval_with_rollback(
                                    target_model_obj=model_b_obj,
                                    target_model_path=model_b_path,
                                    partner_model_path=model_a_path,
                                    layer_idx=layer_idx,
                                    group=groups_to_merge_b[0],
                                    task=task_b,
                                    env_manager=env_manager,
                                    env_config=env_config,
                                    eval_settings=eval_settings,
                                    tmp_dir_root=tmp_dir_root,
                                    last_acc=baseline_acc_b,
                                    drop_threshold=drop_threshold,
                                )
                            else:
                                new_acc_b, accepted_b = _merge_step_multi_groups_with_rollback(
                                    target_model_obj=model_b_obj,
                                    target_model_path=model_b_path,
                                    partner_model_path=model_a_path,
                                    layer_idx=layer_idx,
                                    groups=groups_to_merge_b,
                                    task=task_b,
                                    env_manager=env_manager,
                                    env_config=env_config,
                                    eval_settings=eval_settings,
                                    tmp_dir_root=tmp_dir_root,
                                    last_acc=baseline_acc_b,
                                    drop_threshold=drop_threshold,
                                )
                            _write_csv_row(
                                csv_b_path,
                                {
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "model": model_b_path,
                                    "task": task_b,
                                    "layer": layer_idx,
                                    "variant": variant,
                                    "perturbation": "avg-merge-seq" if accepted_b else "avg-merge-skip",
                                    "score": new_acc_b,
                                    "decision": "accepted" if accepted_b else "rejected",
                                    "threshold": drop_threshold,
                                },
                            )
                            if accepted_b:
                                baseline_acc_b = new_acc_b
                                for g in groups_to_merge_b:
                                    acc_set_b.add(g)
                elif attn_first:
                    # For each group (e.g., attn then mlp), walk layers in the given order
                    for variant in variant_order:
                        print(f"\n== Group sequence: {variant} ==")
                        for layer_idx in selected_layers:
                            print(f"\n-- Layer {layer_idx} --")
                            print(f"Variant: {variant}")
                            # Step for Model A (merge with B), rollback if needed
                            new_acc_a, accepted_a = _merge_step_eval_with_rollback(
                                target_model_obj=model_a_obj,
                                target_model_path=model_a_path,
                                partner_model_path=model_b_path,
                                layer_idx=layer_idx,
                                group=variant,
                                task=task_a,
                                env_manager=env_manager,
                                env_config=env_config,
                                eval_settings=eval_settings,
                                tmp_dir_root=tmp_dir_root,
                                last_acc=baseline_acc_a,
                                drop_threshold=drop_threshold,
                            )
                            _write_csv_row(
                                csv_a_path,
                                {
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "model": model_a_path,
                                    "task": task_a,
                                    "layer": layer_idx,
                                    "variant": variant,
                                    "perturbation": "avg-merge-seq" if accepted_a else "avg-merge-skip",
                                    "score": new_acc_a,
                                    "decision": "accepted" if accepted_a else "rejected",
                                    "threshold": drop_threshold,
                                },
                            )
                            if accepted_a:
                                baseline_acc_a = new_acc_a
                            # Step for Model B (merge with A baseline), rollback if needed
                            new_acc_b, accepted_b = _merge_step_eval_with_rollback(
                                target_model_obj=model_b_obj,
                                target_model_path=model_b_path,
                                partner_model_path=model_a_path,
                                layer_idx=layer_idx,
                                group=variant,
                                task=task_b,
                                env_manager=env_manager,
                                env_config=env_config,
                                eval_settings=eval_settings,
                                tmp_dir_root=tmp_dir_root,
                                last_acc=baseline_acc_b,
                                drop_threshold=drop_threshold,
                            )
                            _write_csv_row(
                                csv_b_path,
                                {
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "model": model_b_path,
                                    "task": task_b,
                                    "layer": layer_idx,
                                    "variant": variant,
                                    "perturbation": "avg-merge-seq" if accepted_b else "avg-merge-skip",
                                    "score": new_acc_b,
                                    "decision": "accepted" if accepted_b else "rejected",
                                    "threshold": drop_threshold,
                                },
                            )
                            if accepted_b:
                                baseline_acc_b = new_acc_b
                else:
                    # Default: for each layer, perform groups in the given order
                    for layer_idx in selected_layers:
                        print(f"\n-- Layer {layer_idx} --")
                        for variant in variant_order:
                            print(f"Variant: {variant}")
                            # Model A
                            new_acc_a, accepted_a = _merge_step_eval_with_rollback(
                                target_model_obj=model_a_obj,
                                target_model_path=model_a_path,
                                partner_model_path=model_b_path,
                                layer_idx=layer_idx,
                                group=variant,
                                task=task_a,
                                env_manager=env_manager,
                                env_config=env_config,
                                eval_settings=eval_settings,
                                tmp_dir_root=tmp_dir_root,
                                last_acc=baseline_acc_a,
                                drop_threshold=drop_threshold,
                            )
                            _write_csv_row(
                                csv_a_path,
                                {
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "model": model_a_path,
                                    "task": task_a,
                                    "layer": layer_idx,
                                    "variant": variant,
                                    "perturbation": "avg-merge-seq" if accepted_a else "avg-merge-skip",
                                    "score": new_acc_a,
                                    "decision": "accepted" if accepted_a else "rejected",
                                    "threshold": drop_threshold,
                                },
                            )
                            if accepted_a:
                                baseline_acc_a = new_acc_a
                            # Model B
                            new_acc_b, accepted_b = _merge_step_eval_with_rollback(
                                target_model_obj=model_b_obj,
                                target_model_path=model_b_path,
                                partner_model_path=model_a_path,
                                layer_idx=layer_idx,
                                group=variant,
                                task=task_b,
                                env_manager=env_manager,
                                env_config=env_config,
                                eval_settings=eval_settings,
                                tmp_dir_root=tmp_dir_root,
                                last_acc=baseline_acc_b,
                                drop_threshold=drop_threshold,
                            )
                            _write_csv_row(
                                csv_b_path,
                                {
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "model": model_b_path,
                                    "task": task_b,
                                    "layer": layer_idx,
                                    "variant": variant,
                                    "perturbation": "avg-merge-seq" if accepted_b else "avg-merge-skip",
                                    "score": new_acc_b,
                                    "decision": "accepted" if accepted_b else "rejected",
                                    "threshold": drop_threshold,
                                },
                            )
                            if accepted_b:
                                baseline_acc_b = new_acc_b
            finally:
                del model_a_obj
                del model_b_obj
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            # Independent per-(layer,group) merges against the unmodified base
            for layer_idx in selected_layers:
                print(f"\n-- Layer {layer_idx} --")
                for variant in variant_order:
                    print(f"Variant: {variant}")
                    acc = _average_between_models_and_eval(
                        model_a_path,
                        model_b_path,
                        layer_idx,
                        variant,
                        task_a,
                        env_manager,
                        env_config,
                        eval_settings,
                        tmp_dir_root,
                    )
                    print(f"  Avg-Merge: {acc:.4f}")
                    _write_csv_row(
                        csv_a_path,
                        {
                            "timestamp": datetime.utcnow().isoformat(),
                            "model": model_a_path,
                            "task": task_a,
                            "layer": layer_idx,
                            "variant": variant,
                            "perturbation": "avg-merge",
                            "score": acc,
                            "decision": "accepted",
                            "threshold": drop_threshold,
                        },
                    )
                    acc_b = _average_between_models_and_eval(
                        model_b_path,
                        model_a_path,
                        layer_idx,
                        variant,
                        task_b,
                        env_manager,
                        env_config,
                        eval_settings,
                        tmp_dir_root,
                    )
                    print(f"  Avg-Merge (B): {acc_b:.4f}")
                    _write_csv_row(
                        csv_b_path,
                        {
                            "timestamp": datetime.utcnow().isoformat(),
                            "model": model_b_path,
                            "task": task_b,
                            "layer": layer_idx,
                            "variant": variant,
                            "perturbation": "avg-merge",
                            "score": acc_b,
                            "decision": "accepted",
                            "threshold": drop_threshold,
                        },
                    )

    print(f"\n--- Sweep complete. Results written to: {output_csv} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layer-wise averaging (merge) experiment between two models.")
    parser.add_argument("--config", type=str, default="experiments/gaussian_experiment.yml", help="Path to config YAML")
    parser.add_argument("--model", type=str, default=None, help="Override base (model A) path for tasks")
    parser.add_argument("--partner_model", type=str, default=None, help="Override partner (model B) path")
    parser.add_argument("--output_csv", type=str, default="experiments/gaussian_sanity_results.csv", help="CSV file to append results to")
    parser.add_argument("--output_csv_a", type=str, default=None, help="CSV path for model A logs (defaults to output_csv with _a suffix)")
    parser.add_argument("--output_csv_b", type=str, default=None, help="CSV path for model B logs (defaults to output_csv with _b suffix)")
    parser.add_argument("--profile_csv_a", type=str, default=None, help="Gaussian sanity CSV to derive candidate order for model A")
    parser.add_argument("--profile_csv_b", type=str, default=None, help="Gaussian sanity CSV to derive candidate order for model B")
    parser.add_argument("--start_layer", type=int, default=None, help="Start layer index (inclusive, used if --layers not provided)")
    parser.add_argument("--end_layer", type=int, default=None, help="End layer index (inclusive, used if --layers not provided)")
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated list of layer indices to merge (e.g., '1,2,3')")
    parser.add_argument("--groups", type=str, default="attn,mlp", help="Comma-separated groups to merge: attn,mlp,both")
    parser.add_argument("--sequential_merge", action="store_true", help="Cumulatively merge layers in the given order and evaluate after each step")
    parser.add_argument("--attn_first", action="store_true", help="In sequential mode, do all attention merges first, then MLPs")
    parser.add_argument("--drop_threshold", type=float, default=0.025, help="Rollback a step if accuracy drops more than this amount")
    parser.add_argument("--task_a", type=str, default="coder", help="Evaluation task for model A (e.g., coder or math)")
    parser.add_argument("--task_b", type=str, default="math", help="Evaluation task for model B (e.g., math or coder)")
    parser.add_argument("--use_main_profile_for_both", action="store_true", help="Use task_a's profile to drive both A and B candidate sequences")
    args = parser.parse_args()
    run_sanity_check(
        model_override=args.model,
        config_path=args.config,
        output_csv=args.output_csv,
        output_csv_a=args.output_csv_a,
        output_csv_b=args.output_csv_b,
        profile_csv_a=args.profile_csv_a,
        profile_csv_b=args.profile_csv_b,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        partner_model_override=args.partner_model,
        layers_csv=args.layers,
        groups_csv=args.groups,
        sequential_merge=args.sequential_merge,
        attn_first=args.attn_first,
        drop_threshold=args.drop_threshold,
        task_a=args.task_a,
        task_b=args.task_b,
        use_main_profile_for_both=args.use_main_profile_for_both,
    )


