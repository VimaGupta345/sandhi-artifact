import os
import sys
import time
import hashlib
import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tempfile
import shutil
import csv
from datetime import datetime
import argparse

MERGE_TOOLS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Register the in-repo `merge_tools` package keyed to THIS file's location, so
# imports work regardless of what the repository directory is named (main
# checkout, worktree, or renamed artifact copy) and a same-named sibling
# checkout on sys.path can never shadow it (sys.modules wins over path search).
import types as _types
if "merge_tools" not in sys.modules:
    _mt_pkg = _types.ModuleType("merge_tools")
    _mt_pkg.__path__ = [MERGE_TOOLS_ROOT]
    sys.modules["merge_tools"] = _mt_pkg

from transformers import AutoModelForCausalLM
from merge_tools.micr import eval_harness  # noqa: E402
from merge_tools.micr import eval_splits  # noqa: E402
from merge_tools.micr.eval_harness import EnvironmentManager, TASK_REGISTRY  # noqa: E402

# Delta-shard candidate saves (production-verified in MICR): per cell, rewrite
# only the shard(s) holding the <=7 perturbed tensors and hardlink every other
# shard from a pristine base dir written once at run start. Strictly optional --
# when unavailable or disabled (MICR_INCREMENTAL_SAVE=0) every cell keeps the
# legacy full save, byte-for-byte unchanged.
try:
    from merge_tools.micr import save_utils  # noqa: E402
except Exception:  # pragma: no cover - delta saving is strictly optional
    save_utils = None

# Profiler task aliases -> eval_harness registry keys.
TASK_NAME_TO_REGISTRY_KEY = {"coder": "humaneval", "math": "gsm8k-cot"}

# Registry tasks whose evals run on a vLLM backend end-to-end -- the only ones
# persistent_eval will actually serve (everything else it declines, and the
# eval falls back to the subprocess path). Mirrored literally from
# merge_tools.micr.run_eval.PERSISTENT_VLLM_TASKS -- keep the two in sync --
# so the profiler does not have to import run_eval's heavy module graph.
PERSISTENT_VLLM_TASKS = {"gsm8k-cot", "gsm8k-pal", "humaneval", "ifeval"}

# Optional run-persistent vLLM engine (one engine reused across cells, weights
# reloaded per candidate). Wired here behind an env flag (see _persistent_enabled)
# so the default path stays the cold-boot subprocess path, byte-for-byte unchanged.
try:
    from merge_tools.micr import persistent_eval  # noqa: E402
except Exception:  # pragma: no cover - persistent path is strictly optional
    persistent_eval = None

# Optional in-process HF evaluator for MC/loglikelihood tasks (opt-in via
# MICR_INPROCESS_HF_EVAL, default OFF): scores the RESIDENT model object with
# lm_eval's Python API -- no candidate save, no subprocess, no checkpoint
# re-load. Any unsupported case falls back to the unchanged subprocess path.
try:
    from merge_tools.micr import inprocess_eval  # noqa: E402
except Exception:  # pragma: no cover - in-process path is strictly optional
    inprocess_eval = None

# Quantized model utilities
from quantized_utils import (
    is_quantized_model,
    load_quantized_model,
    generate_fp8_noise,
    apply_fp8_perturbation,
    quantize_model_dir_to_4bit_bnb,
)


# --- Core Functions ---

def _load_model(path: str, device_hint: str = "cuda") -> torch.nn.Module:
    device = "cuda" if torch.cuda.is_available() and device_hint == "cuda" else "cpu"
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype="auto", low_cpu_mem_usage=True)
    if device == "cuda":
        model.to(device)
    model.eval()
    # Newer transformers VALIDATE GenerationConfig on save_pretrained and raise
    # if e.g. temperature is set while do_sample=False. Some shipped configs
    # (T-pro-it-2.0, observed) are invalid that way, which would crash the very
    # first cell save. Sanitize the in-memory object once here (same logic as
    # run_eval's tmp-save sanitizer); eval dirs get the ORIGINAL config file
    # back via the source-asset overlay, so nothing downstream changes.
    try:
        gen_cfg = getattr(model, "generation_config", None)
        if gen_cfg is not None and getattr(gen_cfg, "do_sample", False) is False:
            for attr, neutral in (("temperature", 1.0), ("top_p", 1.0), ("top_k", 50)):
                val = getattr(gen_cfg, attr, None)
                if val not in (None, neutral):
                    print(f"[load] sanitizing generation_config.{attr} {val} -> {neutral} "
                          f"(do_sample=False; transformers save validation)")
                    setattr(gen_cfg, attr, neutral)
    except Exception as e:  # pragma: no cover - sanitize is best-effort
        print(f"[load] generation_config sanitize skipped: {e}")
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

def _seed_for_cell(seed: int, layer_idx: int, variant: str, perturbation: str) -> None:
    """
    Deterministically seed the torch RNG for one (layer, variant, perturbation)
    cell before its noise is drawn.

    The seed is derived from a stable string via SHA-256 (NOT Python's builtin
    hash(), which is process-salted), so the noise for a given cell is:
      (a) reproducible across runs (same --seed => same noise),
      (b) identical whether the model was reloaded from disk or reverted in place
          (we reseed immediately before drawing, so prior RNG state is irrelevant),
      (c) independent of the order cells are visited (each cell reseeds afresh).

    Only the RNG *state* is set here; the distribution/formula is unchanged.
    """
    key = f"{seed}|{layer_idx}|{variant}|{perturbation}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    seed32 = int(digest[:8], 16)  # stable 32-bit integer in [0, 2**32)
    torch.manual_seed(seed32)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed32)


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
    ]
    for fname in tokenizer_files:
        src = os.path.join(source_model_path, fname)
        if os.path.exists(src):
            try:
                shutil.copy(src, tmp_dir)
            except Exception:
                pass
    return tmp_dir


# Same tokenizer/generation assets _save_model_to_temp_dir copies; the delta-save
# path needs them too because save_utils.save_candidate only hardlinks the
# index/marker/config/generation_config from the base dir.
_TOKENIZER_ASSET_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "spiece.model",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
]


def _copy_tokenizer_assets_unlink_first(source_model_path: str, dest_dir: str) -> None:
    """
    Copy tokenizer assets into ``dest_dir``, unlinking any existing destination
    file first. The unlink is load-bearing: after save_candidate() the dest may
    hold HARDLINKS into the pristine base dir (e.g. generation_config.json), and
    shutil.copy would write through the link and corrupt the base for every
    later cell. Unlinking only drops the name; the base inode is untouched.
    """
    for fname in _TOKENIZER_ASSET_FILES:
        src = os.path.join(source_model_path, fname)
        if os.path.exists(src):
            try:
                dst = os.path.join(dest_dir, fname)
                if os.path.exists(dst):
                    os.unlink(dst)
                shutil.copy(src, dst)
            except Exception:
                pass


def _changed_state_dict_keys(model: torch.nn.Module, layer_idx: int, group: str) -> Optional[List[str]]:
    """
    Exact state-dict keys of the parameters a cell perturbs, derived from the
    SAME selection _perturb_and_eval uses (_collect_group_parameters at
    layer_idx/group) and mapped to names by parameter identity against
    model.named_parameters() -- so the names are right for any architecture
    prefix (e.g. 'model.layers.<L>.mlp.up_proj.weight'). Every name is verified
    to exist in model.state_dict(); returns None whenever the mapping is not
    airtight, in which case the caller must full-save.
    """
    try:
        params = _collect_group_parameters(_get_layer_module(model, layer_idx), group)
        wanted = {id(p) for p in params}
        names = [n for n, p in model.named_parameters() if id(p) in wanted]
        if not names or len(names) != len(params):
            return None
        sd_keys = set(model.state_dict().keys())
        if any(n not in sd_keys for n in names):
            return None
        return names
    except Exception:
        return None


def _prepare_delta_base_dir(model: torch.nn.Module, source_model_path: str, tmp_root: str) -> Optional[str]:
    """
    Full-save the PRISTINE model once into a stable base dir under tmp_root
    (a pure full_save; tokenizer assets are overlaid per-cell post-verify). Every
    later cell delta-saves against it: dirty shards rewritten, clean shards
    hardlinked. The base dir is never written again, so it stays pristine for
    the next cell (per-cell dirs are rmtree'd, which only unlinks names).

    Returns None (=> cells use the legacy full save) when save_utils is
    unavailable, MICR_INCREMENTAL_SAVE=0, the checkpoint is single-shard
    (no index => save_candidate would full-save anyway), or anything raises.
    """
    if save_utils is None or not save_utils.incremental_enabled():
        return None
    base_dir = None
    try:
        base_dir = tempfile.mkdtemp(prefix="gp_delta_base_", dir=tmp_root)
        save_utils.full_save(model, base_dir)
        # NOTE: no tokenizer-asset copies here. The base must stay a PURE
        # full_save so save_candidate's byte-verify (delta dir vs a fresh
        # full_save) compares like with like; source assets (which differ
        # bytewise from save_pretrained's normalized output, e.g.
        # generation_config.json) are overlaid per-cell AFTER the verify.
        # Observed failure otherwise: VERIFY FAILED on ['generation_config.json'].
        if not os.path.exists(os.path.join(base_dir, save_utils.INDEX_NAME)):
            print("[save] checkpoint is single-shard (no index); delta-shard saving disabled, "
                  "cells will use the legacy full save")
            shutil.rmtree(base_dir, ignore_errors=True)
            return None
        print(f"[save] delta-shard saving: ON (pristine base dir: {base_dir})")
        return base_dir
    except Exception as e:
        print(f"[save] could not prepare delta base dir ({type(e).__name__}: {e}); "
              "cells will use the legacy full save")
        if base_dir:
            shutil.rmtree(base_dir, ignore_errors=True)
        return None


def _save_model_delta_to_temp_dir(
    model: torch.nn.Module,
    source_model_path: str,
    tmp_root: str,
    base_dir: str,
    changed_keys: Optional[List[str]],
) -> str:
    """
    Delta-shard variant of _save_model_to_temp_dir: save_utils.save_candidate
    rewrites only the shard(s) holding ``changed_keys`` and hardlinks the rest
    from ``base_dir``, then the usual tokenizer assets are copied in. Falls back
    to the existing full save on ANY precondition failure or exception, with the
    reason logged. Returns the temp dir either way.
    """
    reason = None
    if save_utils is None:
        reason = "save_utils unavailable"
    elif not save_utils.incremental_enabled():
        reason = "incremental saving disabled (MICR_INCREMENTAL_SAVE=0 or a prior verify failure)"
    elif not changed_keys:
        reason = "could not map the perturbed parameters to state_dict keys"
    elif not os.path.exists(os.path.join(base_dir, save_utils.MARKER)):
        reason = f"base dir is missing {save_utils.MARKER}"
    else:
        try:
            with open(os.path.join(base_dir, save_utils.MARKER)) as f:
                if json.load(f) != save_utils._layout_signature():
                    reason = "base dir shard-layout signature mismatch"
        except Exception:
            reason = "unreadable shard-layout marker"
    if reason is not None:
        print(f"  [save] delta save skipped ({reason}); using full save")
        return _save_model_to_temp_dir(model, source_model_path, tmp_root)

    tmp_dir = tempfile.mkdtemp(dir=tmp_root)
    try:
        deltas_before = save_utils.stats()["delta"]
        save_utils.save_candidate(model, base_dir, tmp_dir, changed_keys)
        if save_utils.stats()["delta"] == deltas_before:
            # save_candidate's own preconditions failed (layout/key-set/shard
            # mismatch); it already fell back to a full save inside tmp_dir.
            print("  [save] save_candidate declined the delta path (precondition failed "
                  "inside save_utils); it performed a full save instead")
    except Exception as e:
        print(f"  [save] save_candidate failed ({type(e).__name__}: {e}); falling back to full save")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return _save_model_to_temp_dir(model, source_model_path, tmp_root)

    _copy_tokenizer_assets_unlink_first(source_model_path, tmp_dir)
    return tmp_dir


def _save_model_to_fixed_dir(model: torch.nn.Module, source_model_path: str, dest_dir: str) -> str:
    """
    Save the model to a fixed (non-random) directory, overwriting any prior
    contents. Used for the persistent-eval candidate directory, which the reused
    engine reloads from a stable path every cell.

    Same serialization + tokenizer-asset copy as _save_model_to_temp_dir; only
    the destination is fixed instead of mkdtemp'd.
    """
    if os.path.isdir(dest_dir):
        shutil.rmtree(dest_dir, ignore_errors=True)
    os.makedirs(dest_dir, exist_ok=True)
    try:
        model.save_pretrained(dest_dir, safe_serialization=True)
    except TypeError:
        model.save_pretrained(dest_dir)

    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "spiece.model",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
    ]
    for fname in tokenizer_files:
        src = os.path.join(source_model_path, fname)
        if os.path.exists(src):
            try:
                shutil.copy(src, dest_dir)
            except Exception:
                pass
    return dest_dir


def _persistent_enabled() -> bool:
    """Persistent-eval (one resident vLLM engine per run, ~2s weight reload
    instead of a ~26s cold boot per cell). DEFAULT ON ("1") -- fastest-eval-
    path-by-default policy, matching what the fig5 driver already pins. The
    persistent engine carries the same small run-to-run wobble (~0.1-0.3) as a
    cold subprocess boot (inherent to vLLM engines, NOT added by persistence),
    so it is speed-neutral on reproducibility. Set
    GAUSSIAN_PROFILER_PERSISTENT_EVAL=0 to restore the cold-boot subprocess path
    byte-for-byte (strict-reproduction escape hatch)."""
    if persistent_eval is None:
        return False
    return os.environ.get("GAUSSIAN_PROFILER_PERSISTENT_EVAL", "1").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _inprocess_eligible(task: str, eval_settings: dict) -> bool:
    """True when the in-process HF fast path may serve this task's evals:
    MICR_INPROCESS_HF_EVAL on, MC/loglikelihood registry task, and eval
    settings that build an `lm_eval --model hf` command (no use_vllm). A cheap
    pre-gate so _perturb_and_eval can skip the candidate save entirely;
    inprocess_eval.evaluate re-checks the constructed argv before serving."""
    if inprocess_eval is None or not inprocess_eval.enabled():
        return False
    registry_key = TASK_NAME_TO_REGISTRY_KEY.get(task, task)
    return inprocess_eval.eligible(registry_key, eval_settings)


def evaluate_model(model_path: str, task_name: str, env_manager, env_config: dict, eval_settings: dict,
                   resident_model=None) -> Optional[float]:
    if TASK_REGISTRY is None or env_manager is None:
        print("Evaluation harness not available.")
        return None
    
    registry_task_name = TASK_NAME_TO_REGISTRY_KEY.get(task_name, task_name)

    task_info = TASK_REGISTRY.get(registry_task_name)
    if not task_info:
        print(f"Task '{registry_task_name}' not in registry.")
        return None

    EvaluatorClass = task_info["evaluator"]

    # Optional per-task example cap from the model registry (entry field
    # 'eval_limit' in models_download/hf_repos.json; see
    # eval_harness.registry_eval_limit for the exact --limit semantics).
    # Applied identically to the baseline and every cell, so scores within a
    # run stay comparable. No registry field -> byte-identical behavior. A
    # caller-provided eval_settings['limit'] wins over the registry.
    _registry_limit = eval_harness.registry_eval_limit(registry_task_name)
    if _registry_limit is not None and "limit" not in eval_settings:
        eval_settings = dict(eval_settings)
        eval_settings["limit"] = _registry_limit
        print(f"  [limit] registry eval_limit for '{registry_task_name}': "
              f"{_registry_limit} (models_download/hf_repos.json)")

    # Optional PER-MODEL chat-template flag from the registry (entry field
    # 'apply_chat_template'; see eval_harness.registry_apply_chat_template).
    # Resolved by MODEL (eval_settings['registry_label'], recovered from --model
    # in run_sanity_check) falling back to per-task, so it survives the
    # medqa_4options collision. Applied identically to the profiler's baseline
    # and every cell (they share this eval_settings), so profile scores stay
    # mutually comparable -- the CONSISTENCY INVARIANT for the profiler. No
    # registry field -> None -> nothing set -> byte-identical. A caller-provided
    # value wins. The union with the per-task LM_EVAL_CHAT_TEMPLATE_TASKS lives
    # in the evaluator (BaseEvaluator._wants_chat_template).
    if "apply_chat_template" not in eval_settings:
        _registry_tmpl = eval_harness.registry_apply_chat_template(
            registry_task_name,
            registry_label=eval_settings.get("registry_label"),
        )
        if _registry_tmpl is not None:
            eval_settings = dict(eval_settings)
            eval_settings["apply_chat_template"] = bool(_registry_tmpl)
            print(f"  [chat-template] registry apply_chat_template for "
                  f"'{eval_settings.get('registry_label') or registry_task_name}': "
                  f"{bool(_registry_tmpl)} (models_download/hf_repos.json)")

    # ifeval / Light-IF-32B (decided 2026-07): (1) vLLM backend -- the
    # profiler's default eval_settings carry no use_vllm, which would route
    # ifeval to the hf subprocess and outside the
    # persistent-eval-for-all-vllm-tasks decision; (2) the SAME 10% example
    # cap MICR's runner forces (the profiler otherwise runs the task
    # uncapped, and profile scores must be comparable with the gating
    # scores); (3) gpu_memory_utilization 0.55 so the eval engine coexists
    # with the profiler's resident 32B model on the same GPU(s) (the default
    # 0.8 cannot boot next to it). Work on a COPY: eval_settings is shared
    # across cells/tasks and other tasks must see it untouched. setdefault
    # for the memory knob lets an explicit caller override survive.
    # The chat template + think-strip handling lives in the eval commands
    # themselves (eval_harness routes ifeval to the local *_nothink / _P / _M
    # variants with --apply_chat_template).
    if registry_task_name == "ifeval":
        eval_settings = dict(eval_settings or {})
        eval_settings["use_vllm"] = True
        eval_settings["limit"] = 0.1
        eval_settings.setdefault("vllm_gpu_memory_utilization", 0.55)

    # Persistent-eval fast path (opt-in). One vLLM engine is reused across cells
    # and only reloads weights per candidate. It is gated to the fixed candidate
    # directory (persistent_eval verifies the basename) and returns None on any
    # unsupported case (quantized checkpoint, unknown backend, exception), in
    # which case we fall through to the identical subprocess path below. The
    # returned score matches the subprocess parser (math acc is already a
    # percentage; lm_eval metrics are scaled *100), so numbers do not move.
    # Eval splits: no longer a reason to skip. The persistent lm_eval path
    # replays the evaluator's own argv -- --tasks <task>_<SPLIT> and
    # --include_path <configs dir> included -- and hands simple_evaluate a
    # TaskManager built over that dir, so it resolves the generated split
    # variants exactly like the subprocess CLI does. The math path replays
    # --split the same way. Split or full, every vLLM-backed task is eligible.
    if (_persistent_enabled()
            and persistent_eval is not None
            and Path(model_path).name == getattr(persistent_eval, "CANDIDATE_DIRNAME", None)):
        try:
            p_score = persistent_eval.evaluate(
                EvaluatorClass, env_manager, env_config, eval_settings,
                model_path, registry_task_name,
            )
        except Exception:
            p_score = None
        if p_score is not None:
            try:
                return float(p_score)
            except Exception:
                pass

    # In-process HF fast path (opt-in via MICR_INPROCESS_HF_EVAL): score the
    # caller's RESIDENT model object -- the CURRENT (possibly perturbed)
    # weights -- directly through lm_eval's Python API. When the caller passes
    # resident_model, the candidate was never saved to disk, so this function
    # must NOT fall through to the subprocess evaluator below: model_path is
    # the SOURCE checkpoint (tokenizer provenance), not the candidate, and a
    # subprocess would score pristine weights. On any failure we return None
    # and the CALLER falls back to its own save + subprocess path.
    if resident_model is not None:
        score = None
        if inprocess_eval is not None and inprocess_eval.enabled():
            try:
                score = inprocess_eval.evaluate(
                    EvaluatorClass, env_manager, env_config, eval_settings,
                    resident_model, model_path, registry_task_name,
                )
            except Exception as e:
                print(f"  [inprocess-eval] failed ({type(e).__name__}: {e})")
                score = None
        if score is None:
            return None
        try:
            return float(score)
        except Exception:
            return None

    evaluator = EvaluatorClass(env_manager, env_config, eval_settings)

    absolute_model_path = Path(model_path).resolve()
    model_config = {
        "model_name": os.path.basename(absolute_model_path),
        "path": str(absolute_model_path),
        "location": "local",
    }
    
    # Retry policy mirrors micr/eval_retry: a failed eval must yield None (no
    # measurement), NEVER 0.0 -- a fabricated zero here poisons the noise
    # profile (cell rows falsely un-avgable; a zero BASELINE would make the
    # avgability threshold baseline-5 < everything, marking every component
    # mergeable). Same bug class that corrupted MICR CSVs before eval_retry.
    from merge_tools.micr import eval_retry as _eval_retry
    _retries = _eval_retry.eval_retries()
    _backoff = _eval_retry.eval_retry_backoff_s()
    for _attempt in range(_retries + 1):
        result_dict = evaluator.evaluate(model_config, registry_task_name, run_id=1)
        if result_dict.get("status") == "SUCCESS":
            try:
                score_val = result_dict["score"]
                if isinstance(score_val, str):
                    score_val = score_val.replace("%", "")
                return float(score_val)
            except Exception:
                return None
        print(f"Evaluation failed: {result_dict.get('error_log', 'Unknown error')}")
        if _attempt < _retries:
            print(f"  [eval-retry] attempt {_attempt + 1}/{_retries + 1} failed; "
                  f"retrying in {_backoff * (2 ** _attempt):.0f}s")
            time.sleep(_backoff * (2 ** _attempt))
    return None


# --- Helpers for sweeping and logging ---

def _get_num_layers(model: torch.nn.Module) -> int:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    raise RuntimeError("Unsupported model structure.")


def _row_split(eval_settings: Dict[str, object]) -> str:
    """Split label recorded with every CSV row ("full" when no split is active)."""
    v = eval_settings.get("eval_split") if isinstance(eval_settings, dict) else None
    return str(v) if v else "full"


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
        # Which evaluation split this row was scored on ("P"/"M"/"full").
        # baselines.py keys lookups by split; rows from files predating this
        # column are treated as "full".
        "split",
    ]
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists or write_header:
            writer.writeheader()
        writer.writerow(row)


def _perturb_and_eval(
    model: torch.nn.Module,
    base_model_path: str,
    layer_idx: int,
    group: str,
    perturbation: str,
    task: str,
    env_manager,
    env_config: dict,
    eval_settings: dict,
    tmp_dir_root: str,
    seed: int,
    eval_dir: Optional[str] = None,
    delta_base_dir: Optional[str] = None,
) -> float:
    """
    Perturb one (layer, group) in the already-loaded model IN PLACE, save + eval
    out of process, then restore the pristine weights.

    Only the (<=7) targeted .weight tensors are mutated. They are cloned first and
    restored inside a try/finally so the model is byte-identical to before the cell
    even if evaluation raises. This replaces the previous revert-by-reload; the
    arithmetic (mean/std/normal draw and add/avg/replace) is unchanged.

    delta_base_dir: pristine full save of this model (see _prepare_delta_base_dir).
    When set (and eval_dir is not in play), the per-cell save rewrites only the
    shard(s) holding the perturbed tensors and hardlinks the rest; any
    precondition failure falls back to the legacy full save with a logged reason.
    """
    layer_module = _get_layer_module(model, layer_idx)
    params = _collect_group_parameters(layer_module, group)
    backups = [p.detach().clone() for p in params]
    try:
        _seed_for_cell(seed, layer_idx, group, perturbation)
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
        acc = None
        if _inprocess_eligible(task, eval_settings):
            # In-process fast path: score the perturbed RESIDENT model
            # directly. No candidate save at all -- in this sweep the save
            # exists only to feed the eval subprocess (verified: the tmp dir
            # is rmtree'd right after evaluate_model and has no other
            # consumer). base_model_path is passed for the tokenizer.
            acc = evaluate_model(base_model_path, task, env_manager, env_config,
                                 eval_settings, resident_model=model)
            if acc is None:
                print("  [inprocess-eval] no score from the in-process fast "
                      "path; falling back to the save + subprocess eval for "
                      "this cell (NOTE: a subprocess score can carry a small "
                      "engine offset against an in-process baseline)")
        if acc is not None:
            pass  # scored in-process above; no candidate dir was ever written
        elif eval_dir is not None:
            # Persistent-eval candidate: fixed path the reused engine reloads from.
            tmp_model_dir = _save_model_to_fixed_dir(model, base_model_path, eval_dir)
            acc = evaluate_model(tmp_model_dir, task, env_manager, env_config, eval_settings)
        else:
            if delta_base_dir is not None:
                # Same selection as `params` above, mapped to state_dict names.
                changed_keys = _changed_state_dict_keys(model, layer_idx, group)
                tmp_model_dir = _save_model_delta_to_temp_dir(
                    model, base_model_path, tmp_dir_root, delta_base_dir, changed_keys
                )
            else:
                tmp_model_dir = _save_model_to_temp_dir(model, base_model_path, tmp_dir_root)
            try:
                acc = evaluate_model(tmp_model_dir, task, env_manager, env_config, eval_settings)
            finally:
                shutil.rmtree(tmp_model_dir, ignore_errors=True)
    finally:
        with torch.no_grad():
            for p, b in zip(params, backups):
                p.data.copy_(b)
    return acc


def _perturb_and_eval_fp_then_4bit_eval(
    model: torch.nn.Module,
    base_model_path: str,
    layer_idx: int,
    group: str,
    perturbation: str,
    task: str,
    env_manager,
    env_config: dict,
    eval_settings: dict,
    tmp_dir_root: str,
    seed: int,
    eval_dir: Optional[str] = None,
    delta_base_dir: Optional[str] = None,
    *,
    quant_backend: str = "bnb",
    quant_type: str = "nf4",
    debug: bool = False,
) -> float:
    """
    Full-precision perturbation + 4-bit evaluation.

    delta_base_dir is accepted for signature parity but IGNORED: this path
    quantizes the FP checkpoint after saving, so it keeps its existing
    full-save behavior.

    Flow:
      1) Use the already-loaded full-precision model (cloned/restored per cell).
      2) Apply perturbation in full precision.
      3) Save a temporary *full-precision* checkpoint.
      4) Quantize that checkpoint to 4-bit (second temporary directory).
      5) Evaluate the quantized checkpoint.
      6) Delete both temporary directories and restore the pristine weights.

    This matches the "operate on regular weights, but quantize and save a temp
    version before evaluation" workflow. eval_dir is ignored here: a 4-bit temp
    checkpoint is quantized and never served by the persistent engine.
    """
    if quant_backend != "bnb":
        raise ValueError(
            f"Unsupported 4-bit backend: {quant_backend}. Only 'bnb' is implemented."
        )

    layer_module = _get_layer_module(model, layer_idx)
    params = _collect_group_parameters(layer_module, group)
    backups = [p.detach().clone() for p in params]

    try:
        _seed_for_cell(seed, layer_idx, group, perturbation)
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

        tmp_fp_dir = _save_model_to_temp_dir(model, base_model_path, tmp_dir_root)

        tmp_q_dir = None
        try:
            if debug:
                print(f"[DEBUG] 4-bit quantization backend: {quant_backend} ({quant_type})")
                print(f"[DEBUG] Saving FP checkpoint: {tmp_fp_dir}")
                print(f"[DEBUG] Quantizing FP checkpoint to 4-bit...")

            tmp_q_dir = quantize_model_dir_to_4bit_bnb(
                tmp_fp_dir,
                tmp_dir_root,
                source_assets_dir=base_model_path,
                quant_type=quant_type,
            )

            if debug:
                print(f"[DEBUG] Quantized checkpoint ready: {tmp_q_dir}")
                print(f"[DEBUG] Evaluating quantized checkpoint...")

            acc = evaluate_model(tmp_q_dir, task, env_manager, env_config, eval_settings)
        finally:
            if tmp_q_dir and os.path.isdir(tmp_q_dir):
                shutil.rmtree(tmp_q_dir, ignore_errors=True)
            if os.path.isdir(tmp_fp_dir):
                shutil.rmtree(tmp_fp_dir, ignore_errors=True)
    finally:
        with torch.no_grad():
            for p, b in zip(params, backups):
                p.data.copy_(b)
    return acc


def _perturb_and_eval_quantized(
    model: torch.nn.Module,
    base_model_path: str,
    layer_idx: int,
    group: str,
    perturbation: str,
    task: str,
    env_manager,
    env_config: dict,
    eval_settings: dict,
    tmp_dir_root: str,
    seed: int,
    eval_dir: Optional[str] = None,
    delta_base_dir: Optional[str] = None,
) -> float:
    """
    Quantized-model variant of _perturb_and_eval.

    delta_base_dir is accepted for signature parity but IGNORED: the quantized
    path keeps its existing full-save behavior.

    Perturbs FP8 weights directly on the already-loaded model:
    1. Use the loaded FP8 quantized model (cloned/restored per cell).
    2. Cast targeted weights to bfloat16 for arithmetic.
    3. Apply Gaussian noise perturbation.
    4. Clamp to FP8 range and cast back. Scales are untouched.
    5. Save, evaluate, clean up, then restore the pristine FP8 weights.

    eval_dir is ignored: persistent_eval refuses quantized checkpoints, so the
    quantized path always uses the subprocess evaluator.
    """
    layer_module = _get_layer_module(model, layer_idx)
    params = _collect_group_parameters(layer_module, group)
    backups = [p.detach().clone() for p in params]

    try:
        _seed_for_cell(seed, layer_idx, group, perturbation)
        with torch.no_grad():
            for p in params:
                noise = generate_fp8_noise(p)
                apply_fp8_perturbation(p, noise, perturbation)

        tmp_model_dir = _save_model_to_temp_dir(model, base_model_path, tmp_dir_root)
        try:
            acc = evaluate_model(tmp_model_dir, task, env_manager, env_config, eval_settings)
        finally:
            shutil.rmtree(tmp_model_dir, ignore_errors=True)
    finally:
        with torch.no_grad():
            for p, b in zip(params, backups):
                p.data.copy_(b)
    return acc


# --- Main Sanity Check Logic ---

def run_sanity_check(
    model_path: str,
    tasks: List[str],
    output_csv: str,
    gpu_ids: Optional[str] = None,
    tmp_dir: Optional[str] = None,
    start_layer: Optional[int] = None,
    end_layer: Optional[int] = None,
    debug: bool = False,
    perturbations: List[str] = ["avg"],
    groups: Optional[List[str]] = None,
    seed: int = 1234,
    quantized: bool = False,
    eval_4bit: bool = False,
    eval_4bit_backend: str = "bnb",
    eval_4bit_quant_type: str = "nf4",
    eval_split: Optional[str] = None,
):
    """
    Evaluate tasks across a sweep of layers and variants.

    eval_split: 'P'/'M' scores the baseline and every profiled cell on that
    materialized half of each task's eval set (micr/eval_splits.py); the
    intended profiler split is 'P'. None/'full' (default) uses the full set
    and leaves every eval command byte-identical to before this option.
    """
    print("--- Starting Gaussian Noise Sanity Sweep ---")

    # --- 1. Setup ---
    # GPU setup
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids.strip()
        print(f"Using GPUs: {gpu_ids}")

    # Eval settings
    eval_settings = {"gpu_ids": gpu_ids} if gpu_ids else {}
    # Recover the per-model registry LABEL for --model once, so evaluate_model
    # can resolve the optional per-model chat-template flag by MODEL (not just by
    # task): two registry models can share a task (UltraMedical / MedGo on
    # medqa_4options) yet need different prompting. Carried in eval_settings and
    # read in evaluate_model; None (path not in the registry) -> task-only
    # resolution -> no change. Ignored by the eval command builders.
    _reg_label = eval_harness.registry_label_for_path(model_path)
    if _reg_label is not None:
        eval_settings["registry_label"] = _reg_label
    eval_split = eval_splits.normalize_eval_split(eval_split)
    if eval_split is not None:
        eval_settings["eval_split"] = eval_split
        print(f"Eval split: '{eval_split}' -- baseline and every profiled cell are scored on "
              f"this materialized half of the eval set (see merge_tools/micr/eval_splits.py)")
        for task in tasks:
            _reg = TASK_NAME_TO_REGISTRY_KEY.get(task, task)
            _spec = eval_splits.load_recorded_spec(_reg, eval_split)
            if _spec is not None:
                print(f"  eval-split spec [{_reg}]: {json.dumps(_spec, sort_keys=True)}")
            else:
                print(f"  eval-split spec [{_reg}]: NOT MATERIALIZED -- run: "
                      f"python merge_tools/micr/eval_splits.py --task {_reg} --splits {eval_split}")
    env_config = {}  # Empty config, relying on defaults/env vars

    if EnvironmentManager is None:
        raise RuntimeError("EnvironmentManager could not be imported.")
    
    # Initialize EnvironmentManager with defaults (None)
    env_manager = EnvironmentManager(None, None)
    
    # FIXME: tmp dir should be updated for each user.
    tmp_dir_root = tmp_dir if tmp_dir else os.environ.get("MICR_TMP_ROOT", "/tmp/micr_tmp")
    os.makedirs(tmp_dir_root, exist_ok=True)

    # Which parameter groups to sweep, in order. Default stays exactly attn then
    # mlp; "both" is an opt-in ADDITIONAL element (attn+mlp perturbed together).
    variant_order = list(groups) if groups else ["attn", "mlp"]
    print(f"Groups (in order): {variant_order}")
    print(f"Gaussian noise seed: {seed}")

    # Auto-detect quantized model if --quantized not explicitly set
    if not quantized and is_quantized_model(model_path):
        print("Auto-detected quantized model (compressed-tensors). Enabling quantized mode.")
        quantized = True

    if quantized and eval_4bit:
        raise ValueError(
            "Cannot combine --quantized (FP8 compressed-tensors) with --eval_4bit. "
            "Use --quantized for FP8 models, or --eval_4bit for full-precision models."
        )

    eval_4bit_backend = (eval_4bit_backend or "").lower()
    eval_4bit_quant_type = (eval_4bit_quant_type or "").lower()
    if eval_4bit:
        if eval_4bit_backend != "bnb":
            raise ValueError(
                f"Unsupported --eval_4bit_backend: {eval_4bit_backend}. Currently supported: bnb"
            )
        if eval_4bit_quant_type not in ("nf4", "fp4"):
            raise ValueError(
                f"Unsupported --eval_4bit_quant_type: {eval_4bit_quant_type}. Expected nf4 or fp4."
            )

    if quantized:
        print(f"Quantized mode: ON (direct FP8 perturbation, scales untouched)")
        perturb_fn = _perturb_and_eval_quantized
        eval_4bit = False
    else:
        if eval_4bit:
            if debug:
                print(
                    f"[DEBUG] 4-bit eval mode: ON (backend={eval_4bit_backend}, quant_type={eval_4bit_quant_type})"
                )

            def perturb_fn(*args, **kwargs):
                return _perturb_and_eval_fp_then_4bit_eval(
                    *args,
                    **kwargs,
                    quant_backend=eval_4bit_backend,
                    quant_type=eval_4bit_quant_type,
                    debug=debug,
                )
        else:
            perturb_fn = _perturb_and_eval

    # Load the model ONCE for the whole sweep. Every cell perturbs it in place,
    # saves + evaluates out of process, then restores the pristine weights, so a
    # single resident copy serves every (layer, group, perturbation) cell. This
    # replaces the previous per-cell revert-by-reload.
    print("Loading model once for the full sweep...")
    if quantized:
        model = load_quantized_model(model_path)
    else:
        model = _load_model(model_path)
    total_layers = _get_num_layers(model)

    # Persistent-eval candidate directory (opt-in). Only the plain full-precision
    # path is eligible: the quantized/4-bit paths serve quantized checkpoints,
    # which persistent_eval refuses. When disabled, eval_dir stays None and each
    # cell mkdtemp's a throwaway dir exactly as before.
    eval_dir = None
    if _persistent_enabled() and not quantized and not eval_4bit:
        eval_dir = persistent_eval.candidate_dir(tmp_dir_root)
        print(f"Persistent-eval: ON (candidate dir: {eval_dir})")

    # Delta-shard saving (opt-out via MICR_INCREMENTAL_SAVE=0): full-save the
    # pristine model ONCE; each cell then rewrites only the shard(s) holding its
    # <=7 perturbed tensors and hardlinks the rest. Only the plain FP subprocess
    # path is eligible -- the quantized/4-bit paths keep their full saves, and
    # the persistent-eval path saves into its own fixed dir.
    delta_base_dir = None
    if not quantized and not eval_4bit and eval_dir is None:
        if tasks and all(_inprocess_eligible(t, eval_settings) for t in tasks):
            # Every task's cells will be scored in-process on the resident
            # model: no candidate dirs are needed at all, so skip the
            # full-model delta-save base (62GB per 32B job). A cell that ever
            # falls back to the subprocess path uses a legacy full save for
            # that cell.
            print("[save] in-process HF eval covers every task: skipping the "
                  "delta-save base dir (no candidate saves needed)")
        else:
            delta_base_dir = _prepare_delta_base_dir(model, model_path, tmp_dir_root)

    for task in tasks:
        print(f"\n=== Task: {task} | Model: {model_path} ===")

        # Baseline evaluation (once per task)
        print("Baseline: evaluating original model...")
        baseline_model_path = model_path
        baseline_tmp_q_dir = None
        if eval_dir is not None:
            # ENGINE-CONSISTENCY: the cells below are scored by the resident
            # engine, which carries a small CONSTANT per-task config offset
            # against subprocess cold boots (measured: humaneval 82.32/82.32
            # subprocess vs 81.71/81.71 persistent, bit-stable within each
            # path). The baseline must ride the SAME engine, or that offset
            # shifts every (baseline - cell) drop in the noise profile --
            # 0.61 points of a 5.0 threshold on humaneval. Mirror the
            # unmodified model into the candidate dir (hardlinks; identical
            # bytes): evaluate_model's persistent fast path then scores it
            # like any cell. Tasks the persistent module declines (e.g.
            # lm_eval --model hf backends) fall back to a subprocess eval of
            # the same bytes -- which is also how their cells are scored, so
            # both sides of the comparison always take the same path.
            #
            # The mirror itself is only worth that copy when the engine WILL
            # serve the task. persistent_eval declines every non-vLLM-backed
            # task (lm_eval --model hf), so for those the unconditional mirror
            # burned a full checkpoint copy per task (62GB at 32B -- observed
            # filling /scratch) only for the subprocess to score identical
            # bytes from a different path. Gate it on the vLLM-backed task
            # set; declined tasks evaluate the source model path directly,
            # which is byte-identical input to the same subprocess engine
            # their cells use.
            _registry_task = TASK_NAME_TO_REGISTRY_KEY.get(task, task)
            if _registry_task in PERSISTENT_VLLM_TASKS:
                baseline_model_path = persistent_eval.mirror_into_candidate(
                    model_path, tmp_dir_root
                )
                print(f"  [persistent-eval] baseline routed through the resident "
                      f"engine ({baseline_model_path})")
            else:
                print(f"  [persistent-eval] task '{_registry_task}' is not "
                      f"vLLM-backed (persistent_eval declines it); baseline "
                      f"evaluates the source model path directly -- mirror "
                      f"copy skipped")
        if eval_4bit:
            if debug:
                print(f"[DEBUG] Quantizing baseline model to 4-bit before evaluation...")
            baseline_tmp_q_dir = quantize_model_dir_to_4bit_bnb(
                model_path,
                tmp_dir_root,
                source_assets_dir=model_path,
                quant_type=eval_4bit_quant_type,
            )
            baseline_model_path = baseline_tmp_q_dir

        baseline_resident = None
        if (baseline_tmp_q_dir is None and not quantized
                and _inprocess_eligible(task, eval_settings)):
            # PATH-CONSISTENCY (mirror of the ENGINE-CONSISTENCY rule above):
            # this task's cells will be scored by the in-process fast path, so
            # the baseline must be scored the same in-process way or any
            # engine offset shifts every (baseline - cell) drop. The resident
            # model is pristine here -- it is loaded once per sweep and every
            # cell restores its weights.
            baseline_resident = model
            print("  [inprocess-eval] baseline scored in-process on the "
                  "resident model (path-consistency with the cells)")
        try:
            baseline_acc = evaluate_model(baseline_model_path, task, env_manager, env_config, eval_settings,
                                          resident_model=baseline_resident)
            if baseline_resident is not None and baseline_acc is None:
                # Consistency beats speed (same policy as run_eval's
                # persistent-eval baseline): if the in-process path cannot
                # serve the baseline, disable it for the whole run so every
                # cell takes the same subprocess path, then measure the
                # baseline there.
                if inprocess_eval is not None:
                    inprocess_eval.disable(
                        "baseline could not be scored in-process; cells must "
                        "take the same path as the baseline")
                baseline_acc = evaluate_model(baseline_model_path, task,
                                              env_manager, env_config, eval_settings)
        finally:
            if baseline_tmp_q_dir and os.path.isdir(baseline_tmp_q_dir):
                shutil.rmtree(baseline_tmp_q_dir, ignore_errors=True)

        if baseline_acc is None:
            raise RuntimeError(
                f"baseline evaluation FAILED after retries for task '{task}' -- aborting this "
                f"task. A fabricated baseline would corrupt every avgability decision "
                f"(threshold = baseline - drop)."
            )
        print(f"  -> Baseline Accuracy: {baseline_acc:.4f}")
        _write_csv_row(
            output_csv,
            {
                "timestamp": datetime.utcnow().isoformat(),
                "model": model_path,
                "task": task,
                "layer": -1,
                "variant": "none",
                "perturbation": "baseline",
                "score": baseline_acc,
                "split": _row_split(eval_settings),
            },
        )

        # Determine layer sweep range (total_layers is from the resident model)
        sweep_start = 0 if start_layer is None else max(0, start_layer)
        sweep_end = total_layers - 1 if end_layer is None else min(end_layer, total_layers - 1)

        print(f"Sweeping layers {sweep_start}..{sweep_end} ({total_layers} total)")

        # --- Progress tracking (per task) ---
        n_layers_sweep = max(0, sweep_end - sweep_start + 1)
        cells_total = n_layers_sweep * len(variant_order) * len(perturbations)
        cells_done = 0
        sweep_t0 = time.time()

        for layer_idx in range(sweep_start, sweep_end + 1):
            print(f"\n-- Layer {layer_idx} --")
            for variant in variant_order:
                print(f"Variant: {variant}")

                for pert in perturbations:
                    if debug:
                        print(f"[DEBUG] Testing perturbation: Layer {layer_idx}, {variant}, {pert}")
                        print(f"[DEBUG] State: Loading fresh baseline model...")
                        if quantized:
                            print(f"[DEBUG] Quantized path: cast→perturb→clamp→cast (scales untouched)")
                        elif eval_4bit:
                            print(f"[DEBUG] 4-bit eval path: perturb FP → save FP temp → quantize temp → evaluate quantized → cleanup")
                    
                    layer_pos = layer_idx - sweep_start + 1
                    cell_t0 = time.time()
                    print(
                        f"[layer {layer_idx} ({layer_pos}/{n_layers_sweep}) "
                        f"· {variant} · {pert}] running (task {task})..."
                    )

                    acc = perturb_fn(
                        model,
                        model_path,
                        layer_idx,
                        variant,
                        pert,
                        task,
                        env_manager,
                        env_config,
                        eval_settings,
                        tmp_dir_root,
                        seed,
                        eval_dir=eval_dir,
                        delta_base_dir=delta_base_dir,
                    )

                    # Display name formatting. acc may be None (eval failed after
                    # retries) -- must not hit the :.4f format before the None
                    # branch below, or the whole model sweep dies on a TypeError.
                    display_name = "Average" if pert == "avg" else ("Replace" if pert == "replace" else pert.capitalize())
                    print(f"  {display_name}: {'FAILED (no score)' if acc is None else format(acc, '.4f')}")

                    # --- Progress: elapsed + rough ETA + running done/total ---
                    cells_done += 1
                    cell_dt = time.time() - cell_t0
                    elapsed = time.time() - sweep_t0
                    avg_per_cell = elapsed / cells_done if cells_done else 0.0
                    remaining = max(0, cells_total - cells_done)
                    eta = avg_per_cell * remaining
                    print(
                        f"  [progress] {cells_done}/{cells_total} cells "
                        f"| cell {cell_dt:.1f}s | elapsed {elapsed:.1f}s "
                        f"| ETA ~{eta:.1f}s"
                    )

                    if debug and acc is not None:
                        print(f"[DEBUG] Step Result: Score {acc:.4f} recorded. (Profiler mode: Step accepted for log, model reset to baseline)")
                    
                    if acc is None:
                        print(f"  [eval-failed] layer {layer_idx} {variant} {pert}: no score "
                              f"after retries; ROW SKIPPED (clustering treats the missing row "
                              f"as not-avgable -- honest, never a fabricated 0.0)")
                    else:
                        _write_csv_row(
                            output_csv,
                            {
                                "timestamp": datetime.utcnow().isoformat(),
                                "model": model_path,
                                "task": task,
                                "layer": layer_idx,
                                "variant": variant,
                                "perturbation": pert,
                                "score": acc,
                                "split": _row_split(eval_settings),
                            },
                        )

    # Release the resident model and tear down any persistent engine.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if delta_base_dir is not None:
        if save_utils is not None:
            print(f"[save] stats: {save_utils.stats()}")
        shutil.rmtree(delta_base_dir, ignore_errors=True)
    if eval_dir is not None and persistent_eval is not None:
        try:
            persistent_eval.shutdown()
        except Exception:
            pass
        shutil.rmtree(eval_dir, ignore_errors=True)

    print(f"\n--- Sweep complete. Results written to: {output_csv} ---")

    # Completion sentinel for the driver's reuse check. Row count alone cannot
    # distinguish a finished sweep from an interrupted one: rows may
    # legitimately be missing (eval-failure skips are honest holes, treated as
    # not-avgable downstream). This marker asserts the sweep ran to the end.
    try:
        n_rows = max(0, sum(1 for _ in open(output_csv)) - 1)
        with open(str(output_csv) + ".complete", "w") as fh:
            json.dump({"completed": True, "rows_in_csv": n_rows,
                       "timestamp": datetime.utcnow().isoformat()}, fh)
    except OSError as e:
        print(f"[warn] could not write completion sentinel: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gaussian noise sanity sweep across layers and tasks.")
    parser.add_argument("--model", type=str, required=True, help="Path to the model to profile")
    parser.add_argument("--tasks", type=str, default="math,coder", help="Comma-separated list of tasks (e.g., 'math,coder')")
    parser.add_argument("--output_csv", type=str, default="output/gaussian_sanity_results.csv", help="CSV file to append results to")
    parser.add_argument("--start_layer", type=int, default=None, help="Start layer index (inclusive)")
    parser.add_argument("--end_layer", type=int, default=None, help="End layer index (inclusive)")
    parser.add_argument("--gpus", type=str, default=None, help="CUDA_VISIBLE_DEVICES string (e.g., '0,1')")
    parser.add_argument("--tmp_dir", type=str, default=None, help="Temporary directory for model saves")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--perturbation", type=str, default="avg", help="Comma-separated list of perturbations (avg, replace, add). Default: avg")
    parser.add_argument("--groups", type=str, default="attn,mlp", help="Comma-separated parameter groups to sweep, in order. Default 'attn,mlp'. 'both' is an opt-in additional element (attn+mlp perturbed together).")
    parser.add_argument("--seed", type=int, default=1234, help="Base seed for deterministic per-cell Gaussian noise. Default: 1234")
    parser.add_argument("--quantized", action="store_true", help="Enable quantized model mode (direct FP8 perturbation). Auto-detected if not set.")
    parser.add_argument("--eval_4bit", action="store_true", help="Evaluate using a temporary 4-bit quantized checkpoint (perturb in FP, quantize for eval).")
    parser.add_argument("--eval_4bit_backend", type=str, default="bnb", help="4-bit backend for --eval_4bit. Currently supported: bnb")
    parser.add_argument("--eval_4bit_quant_type", type=str, default="nf4", help="4-bit quant type for --eval_4bit (nf4 or fp4)")
    parser.add_argument(
        "--eval_split",
        choices=["P", "M", "full"],
        default="P",
        help="Score baseline and all profiled cells on a materialized half of each task's eval "
             "set ('P' is the profiler half; materialize it first with "
             "merge_tools/micr/eval_splits.py). Default 'full' uses the full set and leaves "
             "every eval command byte-identical to before this flag existed.",
    )

    args = parser.parse_args()
    
    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    pert_list = [p.strip() for p in args.perturbation.split(",") if p.strip()]
    group_list = [g.strip() for g in args.groups.split(",") if g.strip()]

    _ALLOWED_GROUPS = {"attn", "mlp", "both"}
    bad_groups = [g for g in group_list if g not in _ALLOWED_GROUPS]
    if bad_groups:
        parser.error(
            f"--groups contains unsupported value(s): {bad_groups}. "
            f"Allowed: {sorted(_ALLOWED_GROUPS)}"
        )
    if not group_list:
        group_list = ["attn", "mlp"]

    run_sanity_check(
        model_path=args.model,
        tasks=task_list,
        output_csv=args.output_csv,
        gpu_ids=args.gpus,
        tmp_dir=args.tmp_dir,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        debug=args.debug,
        perturbations=pert_list,
        groups=group_list,
        seed=args.seed,
        quantized=args.quantized,
        eval_4bit=args.eval_4bit,
        eval_4bit_backend=args.eval_4bit_backend,
        eval_4bit_quant_type=args.eval_4bit_quant_type,
        eval_split=(None if args.eval_split == "full" else args.eval_split),
    )
