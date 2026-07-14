#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-model merge-and-evaluate runner.

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
import time
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

# The evaluation layer now lives in this repo: micr/eval_harness.py. It builds the same
# argv and parses scores with the same regexes as the former unified-llm-eval checkout
# (verified: 9/9 tasks byte-identical), but runs in the current interpreter's environment
# -- no conda env names, no UNIFIED_LLM_EVAL_ROOT. The vendored math-evaluation-harness is
# still required for --domain math; point MICR_MATH_HARNESS_DIR at it.
from merge_tools.micr import eval_harness  # type: ignore
from merge_tools.micr.eval_harness import EnvironmentManager, TASK_REGISTRY  # type: ignore

# Import core helpers from the existing experiment module
from merge_tools.micr.top_k_experiment import (  # type: ignore
    _infer_model_dtype as tk_infer_model_dtype,
    _load_model as tk_load_model,
    _get_layer_module as tk_get_layer_module,
    _collect_group_parameters as tk_collect_group_parameters,
)
from merge_tools.micr.merge_device import describe as _describe_merge_device  # type: ignore
from merge_tools.micr.merge_device import resolve_merge_device  # type: ignore
from merge_tools.micr import save_utils  # type: ignore
from merge_tools.micr import eval_splits  # type: ignore
from merge_tools.micr.baselines import get_baseline  # type: ignore
from merge_tools.micr.eval_retry import fallback_is_redundant as _fallback_is_redundant  # type: ignore
from merge_tools.micr.eval_retry import (  # type: ignore
    evaluate_baseline_blocking as _evaluate_baseline_blocking,
    evaluate_with_retry as _evaluate_with_retry,
)

# Optional run-persistent vLLM engine: one engine boots per run and reloads
# weights from a fixed candidate dir per step (~2s) instead of a ~26-28s cold
# subprocess boot per eval. Gated by MICR_PERSISTENT_EVAL (read inside
# persistent_eval.enabled(); default ON, run_figures pins it per job). Any
# unsupported case returns None and the subprocess path below runs unchanged.
try:
    from merge_tools.micr import persistent_eval  # type: ignore
except Exception:  # pragma: no cover - persistent path is strictly optional
    persistent_eval = None

# Optional in-process HF evaluator for MC/loglikelihood tasks: scores a
# candidate through lm_eval's Python API inside this process instead of a cold
# `lm_eval --model hf` subprocess that re-loads the checkpoint from disk.
# Gated by MICR_INPROCESS_HF_EVAL (default OFF) + task-is-MC + backend-is-hf;
# any unsupported case returns None and the subprocess path runs unchanged.
try:
    from merge_tools.micr import inprocess_eval  # type: ignore
except Exception:  # pragma: no cover - in-process path is strictly optional
    inprocess_eval = None


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
FULL_LAYER_COMPONENT_ORDER: List[str] = ATTN_COMPONENT_ORDER + MLP_COMPONENT_ORDER

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

TASKS_REQUIRING_INT_BATCH_SIZE: Set[str] = {
    "gsm8k-cot",
    "gsm8k-pal",
}

# Registry tasks whose per-step evals run on a vLLM backend end-to-end: the
# vendored math harness (gsm8k) and lm_eval --model vllm (humaneval, ifeval).
# Only these route through the persistent engine; every other task keeps its
# unmodified lm_eval --model hf subprocess path (candidates for them are still
# saved into throwaway mkdtemp dirs, so persistent_eval's basename gate never
# even fires).
PERSISTENT_VLLM_TASKS: Set[str] = {"gsm8k-cot", "gsm8k-pal", "humaneval", "ifeval"}


# Env strings that read as "on" (mirrors inprocess_eval.enabled / persistent_eval.enabled).
_ENV_TRUE: Set[str] = {"1", "true", "yes", "on"}


def task_prefers_vllm(registry_key: str) -> bool:
    """Per-task DEFAULT eval backend: True -> vLLM, False -> lm_eval hf.

    This replaces the old blanket ``eval_settings.setdefault("use_vllm", True)``
    with a principled per-task decision so that every task takes its fastest
    SAFE path by default (fastest-eval-path-by-default policy):

      * Generative tasks -- the vendored math harness (gsm8k-cot/pal) and the
        lm_eval ``--model vllm`` tasks (humaneval, ifeval), i.e. exactly
        ``PERSISTENT_VLLM_TASKS`` -- run end-to-end on vLLM and always prefer
        it (that is where the persistent-engine win lives).

      * MC / loglikelihood (SimpleEvaluator) tasks prefer the lm_eval ``hf``
        backend, because that is the backend the in-process fast path
        (``inprocess_eval``) serves, and it is the project's canonical
        loglikelihood backend (the profiler already scores these tasks on hf,
        and hf in-process is bit-identical to hf subprocess). The switch is made
        ONLY when the in-process fast path is configured on AND the task is one
        it recognizes; otherwise the historical ``use_vllm=True`` default is
        kept.

    ESCAPE HATCH / byte-identity: the MC->hf switch keys on the SAME env flag as
    the in-process path (``MICR_INPROCESS_HF_EVAL``, default "1"). With
    ``MICR_INPROCESS_HF_EVAL=0`` every task keeps ``use_vllm=True`` -- exactly
    the pre-fast-path default -- so the whole runner is byte-identical to today.

    WITHIN-RUN CONSISTENCY: the decision keys on the configured env flag rather
    than ``inprocess_eval.enabled()`` on purpose. ``enabled()`` also goes False
    once the in-process path auto-disables after a systematic failure; if the
    backend followed that, a mid-run disable would flip MC evals from hf back to
    vLLM and break the (baseline vs candidate) path-consistency the drop
    computation relies on. Keying on the env flag keeps the backend fixed at hf
    for the whole run -- an auto-disable then falls back to the hf SUBPROCESS
    (same backend, bit-identical), never to vLLM.
    """
    if registry_key in PERSISTENT_VLLM_TASKS:
        return True
    inproc_on = os.environ.get("MICR_INPROCESS_HF_EVAL", "1").strip().lower() in _ENV_TRUE
    if (inproc_on and inprocess_eval is not None
            and registry_key in getattr(inprocess_eval, "MC_HF_TASKS", frozenset())):
        return False
    return True


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
        # Whether any op in this step's block used row-affine scaling. Every
        # audited consumer (plot_steps pd.read_csv, build_operating_points /
        # run_figures csv.DictReader, replay's pd.read_csv) reads columns by
        # name, so the extra column is backward-compatible.
        "scaled",
    ]
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists or write_header:
            writer.writeheader()
        stamped = dict(row)
        stamped.setdefault("scaled", False)
        stamped["timestamp"] = datetime.now(timezone.utc).isoformat()
        writer.writerow(stamped)


def _is_full_layer_bundle(block_df: pd.DataFrame) -> bool:
    if block_df is None or block_df.empty:
        return False
    components = [str(c) for c in block_df["component"].tolist()]
    return components == FULL_LAYER_COMPONENT_ORDER


# Families whose members have divergent weight distributions even WITHIN the
# family, so their merges are scaled too: the DeepSeek pair (coder / math) and
# the Qwen models (2.5-coder / 2.5-math / Qwen3-32B tunes) are different
# pretrains or bases, unlike the Llama pool (same-base fine-tunes, near-
# duplicate weights). Matches the historical runs: DeepSeek MICR used
# --enable-scaling and the old 32B runner always rescaled.
AUTO_SCALE_FAMILIES = {"deepseek", "qwen"}


def op_needs_scaling(participants, enable_scaling: bool, auto_scale: bool = True,
                     mode: str = "auto") -> bool:
    """Scaling policy for one merge op.

    ``mode`` (CLI --scaling) is the per-run override, decided per model SET:
      - "off":  never scale. Chosen for single-family same-base finetune sets
                (llama5 AND the qwen3-32B trio) -- this deliberately OVERRIDES
                the AUTO_SCALE_FAMILIES "qwen" auto-trigger for the 32B set.
      - "on":   scale every op (equivalent to --enable-scaling). Chosen for
                different-pretrain sets (deepseek pair, qwen2.5-7B pair).
      - "auto": exact historical behavior below (default; keeps replay of OLD
                recorded runs driven by --enable-scaling byte-compatible).

    In "auto": --enable-scaling forces it on for every op. Otherwise scaling
    auto-enables when the op's participants (a) span more than one family, or
    (b) include a family in AUTO_SCALE_FAMILIES -- both are
    distribution-mismatch cases where an unscaled average lets the large-std
    member dominate (verified: scale alignment fixes relF 1.54 -> 0.71 for the
    mismatched member). Pure-Llama ops stay unscaled, preserving the validated
    bitwise-stable default.
    """
    if mode == "off":
        return False
    if mode == "on":
        return True
    if enable_scaling:
        return True
    if not auto_scale:
        return False
    families = {infer_family_prefix(lbl) for lbl, _ in participants}
    families.discard("")
    if len(families) > 1:
        return True
    return bool(families & AUTO_SCALE_FAMILIES)


# ---------------------------------------------------------------------------
# Scaling-factor sidecar (transform "row_affine_v1").
#
# A scaled merge bakes a row-wise affine into the target's weights; at serving
# time ONE UNSCALED canonical average tensor is shared by all group members and
# each member recovers its own distribution from its factors. The sidecar's
# PRIMARY content is each member's DISTRIBUTION IDENTITY: the per-row (mu,
# sigma) of the member's ORIGINAL on-disk tensor. Row-affines compose, so the
# reconstruction rule needs only destination stats:
#
#     member = (X - rowmean(X)) * (sigma / rowstd(X)) + mu
#
# where X is ANY stored representation of the merged slot (the unscaled
# canonical average OR another member's baked scaled tensor) and rowmean/rowstd
# are computed from X's stored bytes at load (float32, stats over dim 1,
# unbiased=False). SECONDARY (present only where they diverge from the
# identity): the ACHIEVED row stats of the just-baked tensor, flagged with the
# cause -- "clamp" (the merge's scale clamp bound, so the affine could not
# reach the identity), "overwrite" (multi-stage last-write-wins re-merge of the
# slot; MICR validated the trajectory state, not the original), or
# "unexplained". Reconstructing with achieved stats reproduces the tensor MICR
# actually validated.
#
# Only ACCEPTED merges persist (rejected candidates are reverted), aggregated
# last-write-wins across stages exactly like the recipe semantics. Format .npz
# (NOT .safetensors: run_figures's finaleval stage deletes *.safetensors under
# the variant dirs, and the replay handoff sidecar lives exactly there).
# ---------------------------------------------------------------------------
SCALING_SIDECAR_BASENAME = "scaling_factors.npz"
SCALING_TRANSFORM = "row_affine_v1"
_SCALING_EPS = 1e-6
_SCALING_CLAMP = (0.1, 10.0)

# ops component name -> merge-spec output name, matching
# scripts/formatter.OPS_COMPONENT_TO_OUTPUT so the operating-point tooling can
# join sidecar keys directly against B/C.jsonl entries.
_SIDECAR_COMPONENT_NAME: Dict[str, str] = {
    "attn_q": "attn.q_proj", "attn_k": "attn.k_proj", "attn_v": "attn.v_proj",
    "attn_o": "attn.o_proj",
    "mlp_gate": "mlp.gate_proj", "mlp_up": "mlp.up_proj", "mlp_down": "mlp.down_proj",
}
_SIDECAR_STATE_DICT_PATH: Dict[str, str] = {
    "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj", "attn_o": "self_attn.o_proj",
    "mlp_gate": "mlp.gate_proj", "mlp_up": "mlp.up_proj", "mlp_down": "mlp.down_proj",
}


def scaling_sidecar_metadata(target_label: str, source_model: str, scaling_mode: str) -> Dict[str, object]:
    return {
        "format": "micr_scaling_factors",
        "version": 1,
        "transform": SCALING_TRANSFORM,
        # THE reconstruction rule (see the block comment above). Keep this text
        # in sync with scripts/build_operating_points.py's point sidecar.
        "reconstruction": (
            "member = (X - rowmean(X)) * (sigma / rowstd(X)) + mu, per output "
            "row (stats over dim 1, float32, unbiased=False), then cast to "
            "dtype_policy. X is ANY stored representation of the merged slot "
            "-- the unscaled canonical average or another member's baked "
            "scaled tensor -- with rowmean/rowstd computed from X's stored "
            "bytes at load. mu/sigma are this member's ORIGINAL on-disk row "
            "stats (distribution identity). Where achieved_mu/achieved_sigma "
            "are present, use them instead to reproduce the exact tensor MICR "
            "validated (per-slot cause flags in metadata['slots'])."
        ),
        "keys": "L{layer}.{component}.{mu|sigma|achieved_mu|achieved_sigma}",
        "components": sorted(_SIDECAR_COMPONENT_NAME.values()),
        "dtype_policy": "bfloat16",
        "merge_time_transform": {
            "eps": _SCALING_EPS,
            "clamp": list(_SCALING_CLAMP),
            "note": ("at merge time MICR applied scale = clamp(sigma_target / "
                     "(sigma_avg + eps), *clamp) to the fp32 average; the "
                     "runtime rule above intentionally omits eps/clamp -- the "
                     "achieved_* stats carry any clamp effect instead"),
        },
        "target_label": target_label,
        "source_model": source_model,
        "scaling_mode": scaling_mode,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def _original_row_stats(model_dir: str, layer: int, comp: str):
    """Per-row (mu, sigma) of the ORIGINAL on-disk tensor for (layer, comp).

    Reads only the shard holding the tensor (safetensors lazy load, CPU).
    Returns (mu, sigma) float32 CPU tensors, or None if the tensor cannot be
    located. MUST be computed from the pristine source model dir -- the working
    copy's tensor reflects trajectory state, not the distribution identity.
    """
    try:
        from safetensors import safe_open
    except Exception:
        return None
    rel = _SIDECAR_STATE_DICT_PATH.get(comp)
    if rel is None:
        return None
    key = f"model.layers.{int(layer)}.{rel}.weight"
    d = Path(model_dir)
    try:
        idx = d / "model.safetensors.index.json"
        if idx.exists():
            shard = json.load(open(idx)).get("weight_map", {}).get(key)
            files = [d / shard] if shard else []
        elif (d / "model.safetensors").exists():
            files = [d / "model.safetensors"]
        else:
            files = sorted(d.glob("*.safetensors"))
        for f in files:
            with safe_open(str(f), framework="pt", device="cpu") as sf:
                if key in sf.keys():
                    t = sf.get_tensor(key).to(torch.float32)
                    if t.ndim < 2:
                        return None
                    return t.mean(dim=1), t.std(dim=1, unbiased=False)
        # pytorch .bin checkpoints (e.g. deepseek-math ships no safetensors --
        # and the deepseek pair is exactly a scaling-ON set). mmap avoids
        # loading the whole shard into RAM where torch supports it.
        bin_idx = d / "pytorch_model.bin.index.json"
        bin_files = []
        if bin_idx.exists():
            shard = json.load(open(bin_idx)).get("weight_map", {}).get(key)
            bin_files = [d / shard] if shard else []
        elif (d / "pytorch_model.bin").exists():
            bin_files = [d / "pytorch_model.bin"]
        for f in bin_files:
            try:
                sd = torch.load(str(f), map_location="cpu", weights_only=True, mmap=True)
            except TypeError:
                sd = torch.load(str(f), map_location="cpu", weights_only=True)
            t = sd.get(key)
            if t is None or t.ndim < 2:
                continue
            t = t.to(torch.float32)
            return t.mean(dim=1), t.std(dim=1, unbiased=False)
    except Exception as e:
        print(f"  [scaling-sidecar] original stats read failed for {key}: "
              f"{type(e).__name__}: {e}")
    return None


def _write_scaling_sidecar(path: str, factors: Dict[Tuple[int, str], dict],
                           meta: Dict[str, object]) -> None:
    """Atomically (re)write the run's sidecar npz from the aggregate dict."""
    import numpy as np
    arrays: Dict[str, object] = {}
    slot_meta: Dict[str, dict] = {}
    for (layer, comp), rec in sorted(factors.items()):
        prefix = f"L{int(layer)}.{_SIDECAR_COMPONENT_NAME.get(comp, comp)}"
        arrays[prefix + ".mu"] = rec["mu"].numpy()
        arrays[prefix + ".sigma"] = rec["sigma"].numpy()
        sm: dict = {"step_idx": rec.get("step_idx")}
        if rec.get("achieved_cause"):
            arrays[prefix + ".achieved_mu"] = rec["achieved_mu"].numpy()
            arrays[prefix + ".achieved_sigma"] = rec["achieved_sigma"].numpy()
            sm["achieved_cause"] = list(rec["achieved_cause"])
        slot_meta[prefix] = sm
    if not arrays:
        return
    header = dict(meta)
    header["slots"] = slot_meta
    arrays["__metadata__"] = np.frombuffer(
        json.dumps(header, sort_keys=True).encode("utf-8"), dtype=np.uint8
    )
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, str(path))


def _plan_eval_blocks(merges_only: pd.DataFrame, bundle_eval_mode: str) -> List[Dict[str, object]]:
    """Plan evaluation blocks while preserving CSV row order.

    Modes:
      - group: evaluate per contiguous (stage, layer, group) block
      - layer_full: evaluate per contiguous (stage, layer) block
      - auto: if a contiguous (stage, layer) block is exactly the 7-component
        layer bundle used by clustering approach 4, evaluate once for the full
        layer; otherwise fall back to group-wise evaluation.
    """
    blocks: List[Dict[str, object]] = []
    if merges_only is None or merges_only.empty:
        return blocks

    n_rows = len(merges_only)
    i = 0
    while i < n_rows:
        first = merges_only.iloc[i]
        stage = int(first.get("stage", 1))
        layer = int(first["layer"])

        j = i
        same_layer_indices: List[int] = []
        while j < n_rows:
            row = merges_only.iloc[j]
            if int(row.get("stage", 1)) != stage or int(row["layer"]) != layer:
                break
            same_layer_indices.append(j)
            j += 1

        layer_block = merges_only.iloc[same_layer_indices].reset_index(drop=True)
        use_layer_bundle = bundle_eval_mode == "layer_full" or (
            bundle_eval_mode == "auto" and _is_full_layer_bundle(layer_block)
        )

        if use_layer_bundle:
            blocks.append(
                {
                    "stage": stage,
                    "layer": layer,
                    "group": "layer_full",
                    "indices": same_layer_indices[:],
                }
            )
        else:
            k = i
            while k < j:
                row_k = merges_only.iloc[k]
                group = str(row_k["group"])
                group_indices: List[int] = []
                while k < j:
                    row_inner = merges_only.iloc[k]
                    if str(row_inner["group"]) != group:
                        break
                    group_indices.append(k)
                    k += 1
                blocks.append(
                    {
                        "stage": stage,
                        "layer": layer,
                        "group": group,
                        "indices": group_indices,
                    }
                )

        i = j

    return blocks


def _save_model_to_temp_dir_for_eval(
    model: torch.nn.Module,
    base_model_path: str,
    tmp_root: str,
    changed_keys: Optional[Set[str]] = None,
    fixed_dir: Optional[str] = None,
) -> str:
    """
    Save the modified model to a temporary directory and copy / recreate
    tokenizer/config assets so that the eval harness (including vLLM) can load it.

    When ``changed_keys`` names the state_dict entries this block mutated and
    ``base_model_path`` is a safetensors dir written by this process, only the shard(s)
    holding those tensors are re-serialized; the rest are hardlinked. See micr/save_utils.py.

    ``fixed_dir``: save into this exact directory (recreated empty) instead of a
    fresh mkdtemp. Used for the persistent-eval candidate: the resident engine
    reloads weights from the path it booted with, so the candidate must live at
    one stable location for the whole run (cf. gaussian_profiler's
    _save_model_to_fixed_dir / persistent_eval.CANDIDATE_DIRNAME).
    """
    os.makedirs(tmp_root, exist_ok=True)
    if fixed_dir is not None:
        if os.path.isdir(fixed_dir):
            shutil.rmtree(fixed_dir, ignore_errors=True)
        os.makedirs(fixed_dir, exist_ok=True)
        tmp_dir = fixed_dir
    else:
        tmp_dir = tempfile.mkdtemp(dir=tmp_root)

    # Coerce model tensors to the base model's dtype (helps vLLM + HF loaders)
    try:
        target_dtype = tk_infer_model_dtype(base_model_path)
        model.to(dtype=target_dtype)
        if hasattr(model, "config"):
            model.config.torch_dtype = target_dtype
    except Exception:
        pass

    # Newer versions of transformers validate GenerationConfig on save and
    # will raise if, for example, temperature is set while do_sample=False.
    # Some shipped configs (e.g., T-pro-it-2.0) have temperature=0.6 without
    # explicitly setting do_sample, which defaults to False and triggers:
    #   "GenerationConfig is invalid: `temperature`: `do_sample` is set to `False` ..."
    # We sanitize that here so that save_pretrained() succeeds.
    # (Ported from the former run_eval_32b.py, where large models first hit this.)
    try:
        gen_cfg = getattr(model, "generation_config", None)
        # Avoid importing GenerationConfig just for isinstance checks; we only
        # need attribute access.
        if gen_cfg is not None:
            do_sample = getattr(gen_cfg, "do_sample", False)
            temperature = getattr(gen_cfg, "temperature", None)
            # If we're in greedy mode (do_sample=False), reset temperature to
            # the neutral default 1.0 when it's set to a non-default value.
            if do_sample is False and temperature not in (None, 1.0):
                print(
                    f"[tmp-save] Adjusting generation_config.temperature from "
                    f"{temperature} to 1.0 to satisfy transformers validation"
                )
                try:
                    gen_cfg.temperature = 1.0
                except Exception:
                    # If we can't modify it, just continue and let save fail if it must.
                    pass
    except Exception as e:
        print(f"[tmp-save] Failed to sanitize generation_config: {e}")

    # Save model weights: delta-shard when provably byte-identical, else full save.
    if changed_keys:
        save_utils.save_candidate(model, str(base_model_path), tmp_dir, changed_keys)
    else:
        save_utils.full_save(model, tmp_dir)

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





def _n_way_average_subcomponent_into_target(
    target_model: torch.nn.Module,
    target_model_path: str,
    target_layer_idx: int,
    group_name: str,
    component_attr: str,
    contributors: List[Tuple[torch.nn.Module, int]],
    device: str = "cuda",
    target_std: Optional[float] = None,
    enable_scaling: bool = False,
    factors_out: Optional[dict] = None,
) -> None:
    """
    N-way average of a *single subcomponent* (attn q/k/v/o OR mlp gate/up/down) across
    [target + partners] INTO target_model, without touching other subcomponents.

    contributors: List of (model, source_layer_idx) tuples.
                  Includes partners loaded in memory.

    factors_out: when a dict is passed AND enable_scaling is True, the sidecar
    capture (read-only, fp32 CPU copies; zero effect on the merge arithmetic)
    records the AT-MERGE-TIME row stats of the pre-merge target (trajectory
    state -- for slots overwritten by later stages these differ from the
    original disk weights, and recomputing from originals would be wrong),
    whether the scale clamp bound, and the ACHIEVED row stats of the
    just-baked (bf16-cast) tensor. None (the default) captures nothing.
    """
    if not component_attr:
        return

    target_low_dtype = tk_infer_model_dtype(target_model_path)
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
        # Add target's own weight first? 
        # CAUTION: The contributor list constructed in _process_block includes the target 
        # if it was listed in the CSV models list. We should rely on that list.
        # However, p_t is the weight tensor of the target *in place*. 
        # We need to collect weights from the contributor models.
        # If a contributor is the target model (same object), we can use p_t (or a clone).
        
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
            device = resolve_merge_device("cuda", enable_scaling)

        with torch.no_grad():
            acc = torch.zeros_like(p_t.data, dtype=torch.float32, device=device)
            target_weight = None
            if enable_scaling:
                target_weight = p_t.data.to(device=device, dtype=torch.float32).clone()
            
            # Sum up all contributors
            for w_m in contributor_weights:
                acc += w_m.data.to(device=device, dtype=torch.float32)
            
            # N-way average
            k = len(contributor_weights)
            if k > 0:
                acc /= float(k)

                # Match each output row's first two moments to the target weight.
                if enable_scaling and target_weight is not None:
                    eps = 1e-6
                    if acc.ndim >= 2:
                        mu_t = target_weight.mean(dim=1, keepdim=True)
                        std_t = target_weight.std(dim=1, keepdim=True, unbiased=False)

                        mu_m = acc.mean(dim=1, keepdim=True)
                        std_m = acc.std(dim=1, keepdim=True, unbiased=False)

                        # (scale_raw kept separate only so the sidecar capture
                        # can flag rows where the clamp bound; the arithmetic
                        # is op-for-op identical to the previous inline form.)
                        scale_raw = std_t / (std_m + eps)
                        scale = torch.clamp(scale_raw, 0.1, 10.0)
                        acc = (acc - mu_m) * scale + mu_t
                        if factors_out is not None:
                            factors_out["merge_mu"] = (
                                mu_t.detach().reshape(-1).to("cpu", torch.float32).clone())
                            factors_out["merge_sigma"] = (
                                std_t.detach().reshape(-1).to("cpu", torch.float32).clone())
                            factors_out["clamp_bound"] = bool(
                                torch.ne(scale, scale_raw).any().item())
                            factors_out["eps"] = float(eps)
                            factors_out["clamp"] = (0.1, 10.0)
                    else:
                        mu_t = target_weight.mean()
                        std_t = target_weight.std(unbiased=False)
                        mu_m = acc.mean()
                        std_m = acc.std(unbiased=False)
                        scale = torch.clamp(std_t / (std_m + eps), 0.1, 10.0)
                        acc = (acc - mu_m) * scale + mu_t

                out_low = acc.to(device=p_t.data.device, dtype=target_low_dtype)
                p_t.data.copy_(out_low)
                if factors_out is not None and enable_scaling and acc.ndim >= 2:
                    # ACHIEVED row stats of the just-baked tensor (post bf16
                    # cast): what the affine actually produced. Diverges from
                    # the original identity when the clamp bound or the slot
                    # was previously overwritten; the caller diffs and stores
                    # these only for diverging slots.
                    baked32 = out_low.detach().to(dtype=torch.float32)
                    factors_out["achieved_mu"] = baked32.mean(dim=1).to("cpu").clone()
                    factors_out["achieved_sigma"] = (
                        baked32.std(dim=1, unbiased=False).to("cpu").clone())
            
            if device == "cuda":
                torch.cuda.synchronize()

        if hasattr(target_model, "config"):
            target_model.config.torch_dtype = target_low_dtype
    except Exception as e:
        print(f"Error in averaging {component_attr}: {e}")
        raise e


def create_unified_eval_context(
    gpu_ids: Optional[str] = None,
    timeout_minutes: int = 15,
    batch_size: str | int = "auto",
    temperature: float = 0.0,
    output_dir: str = "./evaluation_results",
    eval_split: Optional[str] = None,
):
    if EnvironmentManager is None:
        raise RuntimeError(
            "EnvironmentManager could not be imported. "
            "micr/eval_harness.py failed to import."
        )
    if gpu_ids is not None and str(gpu_ids).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids).strip()
    batch_size_value: str | int
    if isinstance(batch_size, str) and batch_size.strip().lower() == "auto":
        batch_size_value = "auto"
    else:
        batch_size_value = int(batch_size)
    eval_settings = {
        "gpu_ids": "" if gpu_ids is None else str(gpu_ids),
        "timeout_minutes": int(timeout_minutes),
        "batch_size": batch_size_value,
        "temperature": float(temperature),
        "runs_per_eval": 1,
        "output_dir": output_dir,
    }
    # Optional: gate MICR on a random subset of the eval set (profiler/MICR data split).
    _nts = os.environ.get("MICR_NUM_TEST_SAMPLE")
    if _nts:
        eval_settings["num_test_sample"] = int(_nts)
        eval_settings["seed"] = int(os.environ.get("MICR_SEED", "0"))
    # Optional: evaluate on a materialized P/M half of the eval set
    # (micr/eval_splits.py). None/"full" (the default) changes nothing.
    _split = eval_splits.normalize_eval_split(eval_split)
    if _split is not None:
        eval_settings["eval_split"] = _split
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
    batch_size: str | int = "auto",
    temperature: float = 0.0,
    output_dir: str = "./evaluation_results",
    registry_task_name: Optional[str] = None,
    allow_fallback: bool = True,
    eval_split: Optional[str] = None,
    allow_inprocess: bool = True,
    registry_label: Optional[str] = None,
) -> Optional[float]:
    """
    Evaluate a single model on a single task using micr/eval_harness.py.

    Returns the measured score, or None when no score was produced (harness
    unavailable, unknown task, evaluator crash/timeout). A returned 0.0 always
    means the model genuinely scored zero; callers must not conflate the two.

    eval_split: 'P'/'M' scores on that materialized half of the eval set
    (micr/eval_splits.py); None/'full' (default) scores on the full set. Only
    consulted when the eval context is built here; a caller-provided
    eval_settings dict already carries (or omits) its own "eval_split".
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
            eval_split=eval_split,
        )

    registry_key = registry_task_name or DOMAIN_TO_REGISTRY_TASK.get(task_or_domain, task_or_domain)
    task_info = TASK_REGISTRY.get(registry_key)
    if not task_info:
        print(f"Task '{registry_key}' not in TASK_REGISTRY.")
        return None

    EvaluatorClass = task_info["evaluator"]
    # Per-task DEFAULT backend (was: unconditional setdefault True). Generative
    # tasks keep vLLM; MC/loglikelihood tasks default to the lm_eval hf backend,
    # which unlocks the in-process fast path. See task_prefers_vllm for the full
    # policy and the MICR_INPROCESS_HF_EVAL=0 byte-identity escape hatch. An
    # explicit caller-provided use_vllm still wins (setdefault semantics kept).
    try:
        eval_settings = dict(eval_settings or {})
    except Exception:
        eval_settings = {}
    if "use_vllm" not in eval_settings:
        eval_settings["use_vllm"] = task_prefers_vllm(registry_key)
    batch_size_value = eval_settings.get("batch_size", batch_size)
    if isinstance(batch_size_value, str) and batch_size_value.strip().lower() == "auto":
        if registry_key in TASKS_REQUIRING_INT_BATCH_SIZE:
            eval_settings["batch_size"] = 32
            print(f"[batch_size] Task '{registry_key}' does not support 'auto'; using 32")
        else:
            eval_settings["batch_size"] = "auto"

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

    # Forced example limits for the extra 32B-era tasks (ported verbatim from the
    # former run_eval_32b.py). The evaluator holds eval_settings by reference, so
    # mutating after construction still takes effect at evaluate() time.
    if registry_key == "ifeval":
        eval_settings["limit"] = 0.1
        print("[limit] Forcing ifeval limit to 50 examples")
        # Light-IF-32B co-residency: the eval engine shares the GPU(s) with
        # MICR's resident working model; the default 0.8 cannot boot next to
        # a 32B parent. setdefault: an explicit caller override survives.
        # (The chat template + think-strip variants are wired inside
        # eval_harness: ifeval -> ifeval_nothink / ifeval_P / ifeval_M with
        # --apply_chat_template; the registry key stays "ifeval" everywhere,
        # so the final full-set eval automatically scores ifeval_nothink.)
        eval_settings.setdefault("vllm_gpu_memory_utilization", 0.55)
    if registry_key == "m_mmlu_ru":
        eval_settings["limit"] = 0.01
        print("[limit] Forcing m_mmlu_ru limit to 0.01")
    if registry_key == "medqa_4options":
        eval_settings["limit"] = 0.25
        print("[limit] Forcing medqa_4options limit to 0.25")
    if registry_key == "tinyMMLU":
        # No limit is applied; kept only for log parity with the old 32B runner.
        print("[limit] Forcing tinyMMLU limit to 1.0")

    # Optional per-task example cap from the model registry (entry field
    # 'eval_limit' in models_download/hf_repos.json; see
    # eval_harness.registry_eval_limit for the exact --limit semantics). The
    # registry is where the user pins per-task values, so it OVERRIDES the
    # forced parity defaults above. Only SimpleEvaluator commands consume
    # eval_settings['limit']; no registry field -> nothing changes. (The
    # evaluator holds eval_settings by reference -- see the note above -- so
    # mutating here still takes effect at evaluate() time.)
    _registry_limit = eval_harness.registry_eval_limit(registry_key)
    if _registry_limit is not None:
        eval_settings["limit"] = _registry_limit
        print(f"[limit] registry eval_limit for '{registry_key}': {_registry_limit} "
              f"(models_download/hf_repos.json)")

    # Optional PER-MODEL chat-template flag from the same registry (entry field
    # 'apply_chat_template'; see eval_harness.registry_apply_chat_template).
    # Resolved per MODEL (registry_label, threaded from the pipeline's
    # target_label so two models that share a task -- UltraMedical / MedGo on
    # medqa_4options -- can disagree), falling back to per-task. Applied here on
    # eval_settings, which is the same dict the baseline, every per-step
    # candidate AND the final full-set eval build their command from within one
    # run, so all sides prompt the model identically (CONSISTENCY INVARIANT). No
    # registry field -> None -> nothing set -> byte-identical. A caller-provided
    # eval_settings['apply_chat_template'] wins. The union with the per-task
    # LM_EVAL_CHAT_TEMPLATE_TASKS (ifeval forced-on) lives in the evaluator
    # (BaseEvaluator._wants_chat_template); every path -- subprocess hf/vllm,
    # persistent, in-process -- reads it from the argv this produces.
    if "apply_chat_template" not in eval_settings:
        _registry_tmpl = eval_harness.registry_apply_chat_template(
            registry_key, registry_label=registry_label)
        if _registry_tmpl is not None:
            eval_settings["apply_chat_template"] = bool(_registry_tmpl)
            print(f"[chat-template] registry apply_chat_template for "
                  f"'{registry_label or registry_key}': {bool(_registry_tmpl)} "
                  f"(models_download/hf_repos.json)")

    model_path_obj = Path(model_path)
    is_hf_model = "/" in model_path and not model_path_obj.exists()
    if is_hf_model:
        model_config = {"model_name": model_path.split("/")[-1], "path": model_path, "location": "huggingface"}
    else:
        abs_path = model_path_obj.resolve()
        model_config = {"model_name": abs_path.name, "path": str(abs_path), "location": "local"}

    # Persistent-eval fast path: reuse the run-resident vLLM engine (weight
    # reload, ~2s) instead of a cold subprocess boot (~26-28s). It only ever
    # fires for candidates saved into the stable candidate dir --
    # persistent_eval.evaluate verifies the basename is CANDIDATE_DIRNAME --
    # so baselines (working_dir) and the final full-set eval stay on the
    # subprocess path. It builds its command from the same EvaluatorClass /
    # eval_settings as the subprocess below (limits, splits, gen_kwargs
    # included) and returns None for any unsupported case (HF backend,
    # quantized checkpoint, exception), in which case we fall through to the
    # unchanged subprocess path.
    #
    # KNOWN CAVEAT (decided: enable anyway): the persistent engine's scores
    # are NOT guaranteed bit-stable against a cold boot -- gsm8k measured
    # 81.2 / 81.1 under the persistent engine vs a bit-stable 81.3 from the
    # subprocess path. That wobble can flip MICR accept/reject decisions near
    # the drop_tolerance boundary. MICR_PERSISTENT_EVAL=0 restores the
    # bit-stable subprocess behavior. (Full caveat at the candidate-dir setup
    # in run_single_target_pipeline.)
    if (persistent_eval is not None and persistent_eval.enabled()
            and Path(model_config["path"]).name
            == getattr(persistent_eval, "CANDIDATE_DIRNAME", None)):
        try:
            _p_score = persistent_eval.evaluate(
                EvaluatorClass, env_manager, env_config, eval_settings,
                model_config["path"], registry_key,
            )
        except Exception:
            _p_score = None
        if _p_score is not None:
            try:
                return float(_p_score)
            except Exception:
                pass

    # In-process HF fast path (opt-in via MICR_INPROCESS_HF_EVAL, default
    # OFF): MC/loglikelihood tasks on the lm_eval hf backend are scored with
    # lm_eval's Python API in this process, skipping the subprocess
    # interpreter + imports + checkpoint re-load. PATH-CONSISTENCY: the gates
    # depend only on (env flag, task, backend), so within a run the baseline
    # (evaluate_baseline_blocking routes through here) and every candidate
    # take the same path. In-process scores can differ from subprocess scores
    # (batch composition under different free GPU memory) but are
    # self-consistent within a run; the final full-set eval stays on the
    # subprocess path (allow_inprocess=False there) so headline numbers remain
    # comparable project-wide. Any exception or None falls through to the
    # unchanged subprocess path below.
    if (allow_inprocess
            and inprocess_eval is not None and inprocess_eval.enabled()
            and not eval_settings.get("use_vllm", False)):
        try:
            _ip_score = inprocess_eval.evaluate(
                EvaluatorClass, env_manager, env_config, eval_settings,
                model_config["path"], model_config["path"], registry_key,
            )
        except Exception:
            _ip_score = None
        if _ip_score is not None:
            try:
                return float(_ip_score)
            except Exception:
                pass

    result = evaluator.evaluate(model_config, registry_key, run_id=1)
    score_val: Optional[float] = None
    if result.get("status") == "SUCCESS":
        try:
            score_val = float(str(result["score"]).replace("%", ""))
        except Exception:
            score_val = None
    # A SUCCESS that parsed to exactly 0.0 is a real measurement, not a failure.
    # Remember it: the retry below fires for it (as it always has), but if the
    # retry produces nothing we must still report the measured zero.
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
    batch_size: str | int,
    temperature: float,
    drop_tolerance: float,
    eval_enabled: bool,
    sort_mode: str,
    bundle_eval_mode: str,
    ignore_other_families: bool,
    initial_baseline: Optional[float] = None,
    force_calc_baseline: bool = False,
    enable_scaling: bool = False,
    merge_device: str = "auto",
    replay_steps_csv: Optional[str] = None,
    replay_cutoff: Optional[int] = None,
    replay_cutoff_mode: str = "step_idx",
    save_variant_dir: Optional[str] = None,
    eval_split: Optional[str] = None,
    scaling_mode: str = "auto",
) -> None:
    replay_mode = replay_steps_csv is not None
    # Per-run scaling policy (CLI --scaling): "auto" = exact historical
    # behavior; "on"/"off" force it per model set (see op_needs_scaling).
    if scaling_mode not in ("auto", "on", "off"):
        raise ValueError(f"scaling_mode must be auto|on|off, got {scaling_mode!r}")
    if scaling_mode == "off" and enable_scaling:
        raise ValueError("scaling_mode='off' contradicts enable_scaling=True")
    # 'P'/'M' scores the baseline and every per-step gating eval on that
    # materialized half of the eval set; the final summary eval always runs on
    # the full set. None/'full' (default) scores everything on the full set.
    eval_split = eval_splits.normalize_eval_split(eval_split)
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
    if replay_mode and working_dir.exists():
        # A replay must start from the pristine target; a leftover working copy from an
        # earlier run would silently stack merges on top of merges.
        print(f"[replay] discarding stale working copy {working_dir}")
        shutil.rmtree(str(working_dir))
    if not working_dir.exists():
        print(f"[init] copy {target_label}: {target_src} -> {working_dir}")
        shutil.copytree(str(target_src), str(working_dir))

    # Baseline and thresholds for target
    thresholds: Dict[str, float] = {}
    last_score: Dict[str, float] = {}

    thresholds[target_label] = 0.0
    last_score[target_label] = 0.0

    merge_device = resolve_merge_device(merge_device, enable_scaling)
    print(_describe_merge_device(merge_device, enable_scaling))

    if eval_split is not None:
        _split_task = DOMAIN_TO_REGISTRY_TASK.get(target_domain_or_task, target_domain_or_task)
        print(f"[eval-split] baseline + per-step gating on split '{eval_split}' of "
              f"task '{_split_task}'; the final eval runs on the FULL set.")
        _split_spec = eval_splits.load_recorded_spec(_split_task, eval_split)
        if _split_spec is not None:
            print(f"[eval-split] spec: {json.dumps(_split_spec, sort_keys=True)}")
        else:
            print(f"[eval-split] spec: NOT MATERIALIZED -- run: "
                  f"python merge_tools/micr/eval_splits.py --task {_split_task} --splits {eval_split}")

    if enable_scaling:
        print(f"[scale] Enabled row-wise affine moment matching for {target_label}")
    if scaling_mode != "auto":
        print(f"[scale] scaling mode '{scaling_mode}' (--scaling): "
              + ("every merge op is scaled"
                 if scaling_mode == "on"
                 else "scaling disabled, including cross-family/auto-family triggers"))

    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids).strip()

    # ---- scaling-factor sidecar state (row_affine_v1; see the block comment
    # at SCALING_SIDECAR_BASENAME). Only ACCEPTED (or replay-promoted) scaled
    # merges persist, keyed (layer, ops-component), last-write-wins across
    # stages exactly like the recipe semantics. When no op ever scales
    # (pure-llama runs, --scaling off) nothing is captured and NO sidecar file
    # is created -- the run's outputs are unchanged.
    scaling_factors_run: Dict[Tuple[int, str], dict] = {}
    accepted_slots: Set[Tuple[int, str]] = set()
    scaling_sidecar_path = os.path.join(
        os.path.dirname(os.path.abspath(results_csv)) or ".", SCALING_SIDECAR_BASENAME
    )

    def _persist_scaling_sidecar(extra_dir: Optional[str] = None) -> None:
        """(Re)write the run sidecar; optionally mirror it into extra_dir
        (replay's save_variant_dir, next to micr_replay.json). No-op while
        nothing has been captured."""
        if not scaling_factors_run:
            return
        meta = scaling_sidecar_metadata(target_label, str(target_src), scaling_mode)
        try:
            _write_scaling_sidecar(scaling_sidecar_path, scaling_factors_run, meta)
            if extra_dir:
                _write_scaling_sidecar(
                    os.path.join(extra_dir, SCALING_SIDECAR_BASENAME),
                    scaling_factors_run, meta,
                )
        except Exception as e:
            print(f"  [scaling-sidecar] write failed ({type(e).__name__}: {e}); "
                  f"the factors of this run may be incomplete on disk")

    def _commit_scaling_factors(block_factors: Dict[Tuple[int, str], dict],
                                step_idx: int) -> None:
        """Fold a just-ACCEPTED block's captured factors into the run aggregate.

        MUST be called BEFORE accepted_slots is updated with this block (the
        overwrite-cause check needs the set of slots written by EARLIER
        accepts). PRIMARY stats = the member's original on-disk row stats
        (distribution identity, cutoff-independent); ACHIEVED stats are stored
        only where they diverge from that identity, with the cause named.
        """
        if not block_factors:
            return
        try:
            for (layer, comp), rec in block_factors.items():
                causes: List[str] = []
                if rec.get("clamp_bound"):
                    causes.append("clamp")
                if (layer, comp) in accepted_slots:
                    causes.append("overwrite")
                orig = _original_row_stats(str(target_src), layer, comp)
                if orig is None:
                    # Identity unavailable: fall back to the merge-time
                    # trajectory stats so the slot is still reconstructable,
                    # and say so.
                    print(f"  [scaling-sidecar] original stats unavailable for "
                          f"L{layer}.{comp}; storing merge-time stats as the identity")
                    mu_o, sig_o = rec["merge_mu"], rec["merge_sigma"]
                    causes.append("original_stats_unavailable")
                else:
                    mu_o, sig_o = orig
                    if "overwrite" not in causes and not (
                        torch.allclose(mu_o, rec["merge_mu"], rtol=1e-4, atol=1e-7)
                        and torch.allclose(sig_o, rec["merge_sigma"], rtol=1e-4, atol=1e-7)
                    ):
                        # Bookkeeping missed it (e.g. resumed working copy that
                        # already carried merges): the merge-time trajectory
                        # stats visibly differ from the disk identity.
                        causes.append("overwrite")
                if not causes and not (
                    torch.allclose(mu_o, rec["achieved_mu"], rtol=1e-3, atol=1e-6)
                    and torch.allclose(sig_o, rec["achieved_sigma"], rtol=1e-3, atol=1e-6)
                ):
                    # Safety net for divergence we did not model explicitly.
                    causes.append("unexplained")
                out = {"mu": mu_o, "sigma": sig_o, "step_idx": int(step_idx)}
                if causes:
                    out["achieved_mu"] = rec["achieved_mu"]
                    out["achieved_sigma"] = rec["achieved_sigma"]
                    out["achieved_cause"] = causes
                scaling_factors_run[(int(layer), comp)] = out  # last write wins
        except Exception as e:
            # Sidecar problems must never take down a merge run; the factors
            # can be regenerated by replaying the finished steps.csv.
            print(f"  [scaling-sidecar] capture failed ({type(e).__name__}: {e}); "
                  f"slot(s) of step {step_idx} not recorded")
        _persist_scaling_sidecar()

    # ---- replay: decisions come from a previous run's steps.csv, not from an evaluator ----
    replay_rows: List[dict] = []
    if replay_mode:
        _steps = pd.read_csv(replay_steps_csv)
        if replay_cutoff is not None and replay_cutoff_mode == "line":
            # scripts/formatter.py counts 1-based CSV lines (header = line 1) and row 1 of
            # the data is the baseline (step_idx == -1), so line == step_idx + 3.
            replay_cutoff = replay_cutoff - 3
        replay_rows = _steps[_steps["step_idx"] >= 0].to_dict("records")
        _n_acc = sum(
            1 for r in replay_rows
            if str(r["decision"]).strip().lower() == "accepted"
            and (replay_cutoff is None or int(r["step_idx"]) <= replay_cutoff)
        )
        print(
            f"[replay] {replay_steps_csv}: {len(replay_rows)} recorded steps, "
            f"{_n_acc} accepted at or before cutoff={replay_cutoff} (step_idx)"
        )
        print("[replay] per-step evaluation is disabled; decisions are taken from the CSV.")

    # Temp/candidate dir root + persistent-eval setup. This must precede the
    # baseline measurement: when the persistent engine is active, the baseline
    # itself is routed through it (ENGINE-CONSISTENCY below).
    tmp_dir_root = os.path.join(working_root, "tmp_eval_single")
    os.makedirs(tmp_dir_root, exist_ok=True)

    # ---- persistent eval (vLLM-backed tasks only) -------------------------
    # When enabled, every candidate is saved into ONE stable directory
    # (<tmp_dir_root>/candidate_eval) and evaluate_model_for_task routes the
    # eval through the run-resident engine, which reloads weights from that
    # path (~2s) instead of cold-booting a subprocess (~26-28s). Rejected /
    # accepted candidates recreate the dir next step; the engine only touches
    # it during evaluate(). Replay mode never evaluates candidates, so it
    # keeps mkdtemp saves.
    #
    # KNOWN CAVEAT (decided: enable anyway). The persistent engine is NOT
    # guaranteed score-identical to the subprocess path on every task.
    # Measured on this machine:
    #   - gsm8k: small run-to-run wobble (~0.1-0.2 points) observed on BOTH
    #     sides across measurement campaigns (81.2/81.1 persistent vs 81.3
    #     subprocess in one; persistent bit-stable 90.1x3 while subprocess
    #     cold boots wobbled 89.9/90.1 in another).
    #   - humaneval: a CONSTANT per-path config offset -- subprocess
    #     82.32/82.32 vs persistent 81.71/81.71 (exactly one problem),
    #     bit-stable within each path.
    # The wobble can flip individual accept/reject decisions near the
    # drop_tolerance boundary. The constant offset is neutralized by
    # ENGINE-CONSISTENCY: the baseline is measured through the same resident
    # engine as the candidates (see below), so it cancels out of every
    # (baseline - candidate) drop. Export MICR_PERSISTENT_EVAL=0 to restore
    # the pure subprocess behavior.
    persist_candidate_dir: Optional[str] = None
    _persist_registry_task = DOMAIN_TO_REGISTRY_TASK.get(
        target_domain_or_task, target_domain_or_task
    )
    if (eval_enabled and not replay_mode
            and persistent_eval is not None and persistent_eval.enabled()
            and _persist_registry_task in PERSISTENT_VLLM_TASKS):
        persist_candidate_dir = persistent_eval.candidate_dir(tmp_dir_root)
        print(f"[persistent-eval] ON for task '{_persist_registry_task}' "
              f"(candidate dir: {persist_candidate_dir}; MICR_PERSISTENT_EVAL=0 disables)")

    if eval_enabled and not replay_mode:
        # Baseline precedence: --force-calc-baseline forces a fresh measurement;
        # else --initial_baseline (explicit) wins; else a stored baseline from
        # get_baseline(target_label, split=eval_split) is reused ONLY on an exact
        # split match (scores from different splits are not comparable); else a
        # fresh baseline is measured on this runner's own split.
        #
        # ENGINE-CONSISTENCY (persistent eval active): the resident engine
        # carries a small constant per-task config offset against subprocess
        # cold boots (measured: humaneval 82.32 subprocess vs 81.71 persistent,
        # bit-stable on both sides). Candidates are scored by the engine, so
        # the baseline MUST be measured by the same engine, or that offset
        # contaminates every drop computed against it (0.61 of the 2.0
        # tolerance on humaneval). Persistent-enabled runs therefore always
        # MEASURE their baseline through the engine: both the stored baseline
        # (measured by subprocess cold boots) and an explicit
        # --initial_baseline (provenance unknown, almost certainly subprocess)
        # are ignored, loudly.
        base_acc = None if force_calc_baseline else initial_baseline
        if persist_candidate_dir is not None and base_acc is not None:
            print(f"[persistent-eval] engine-consistency: ignoring "
                  f"--initial_baseline={base_acc} for {target_label}; the baseline "
                  f"must be measured by the same resident engine that scores the "
                  f"candidates (subprocess-measured numbers carry a per-task "
                  f"engine offset)")
            base_acc = None
        if base_acc is None and not force_calc_baseline:
            if persist_candidate_dir is not None:
                print(f"[persistent-eval] engine-consistency: skipping stored-baseline "
                      f"lookup for {target_label} (store values were measured by "
                      f"subprocess cold boots; not comparable at engine precision)")
            else:
                base_acc = get_baseline(target_label, split=eval_split)
                if base_acc is not None:
                    print(f"[baseline] {target_label}: using stored baseline "
                          f"(split={eval_split or 'full'}) = {base_acc:.4f}")
        if base_acc is None:
            baseline_eval_path = str(working_dir)
            if persist_candidate_dir is not None:
                # Mirror the UNMODIFIED working model into the candidate dir
                # (hardlinks; identical bytes) so evaluate_model_for_task's
                # persistent fast path scores it with the engine that will
                # score every candidate step.
                baseline_eval_path = persistent_eval.mirror_into_candidate(
                    str(working_dir), tmp_dir_root
                )
                print(f"[persistent-eval] baseline for {target_label} routed "
                      f"through the resident engine ({baseline_eval_path})")
            base_acc = evaluate_baseline_blocking(
                target_label,
                model_path=baseline_eval_path,
                task_or_domain=target_domain_or_task,
                gpu_ids=gpu_ids,
                timeout_minutes=timeout_minutes,
                batch_size=batch_size,
                temperature=temperature,
                output_dir=os.path.join(output_dir, target_label, "baseline"),
                eval_split=eval_split,
                # Per-model chat-template resolution keyed on the target model,
                # identical to the per-step and final calls below: baseline,
                # candidates and final must all prompt the model the same way.
                registry_label=target_label,
            )
            if persist_candidate_dir is not None and not persistent_eval.session_active():
                # Defense in depth: an engine-served eval always leaves the
                # session resident, so no session here proves the baseline
                # number came from the subprocess fallback inside
                # evaluate_model_for_task (boot declined or failed). The
                # candidates must then be scored the same way, or the constant
                # engine offset re-enters every (baseline - candidate) drop.
                # Consistency beats speed: turn the persistent path off for
                # the rest of this run.
                print(f"[persistent-eval] engine did not serve the baseline for "
                      f"{target_label}; disabling persistent eval for this run so "
                      f"candidates are scored by the same (subprocess) path")
                try:
                    persistent_eval.disable(
                        "baseline fell back to subprocess; engine-consistency "
                        "requires candidates to match"
                    )
                except Exception:
                    pass
                shutil.rmtree(persist_candidate_dir, ignore_errors=True)
                persist_candidate_dir = None
        thresholds[target_label] = base_acc
        last_score[target_label] = base_acc
        print(f"[baseline] {target_label} = {base_acc:.4f}")
        _append_csv_row(
            results_csv,
            {
                "step_idx": -1,
                "stage": 0,
                "op": "baseline",
                "component": "baseline",
                "layer": -1,
                "label": target_label,
                "score": base_acc,
                "threshold": 0.0,
                "decision": "baseline",
            },
        )
    else:
        print(f"[baseline] eval disabled, using default 0.0 for {target_label}")

    # Iterate over ordered MERGE ops without changing CSV order. Evaluation
    # can happen per group block or per full layer bundle, depending on
    # bundle_eval_mode. (tmp_dir_root and the persistent-eval setup moved
    # above the baseline block: engine-consistency requires them there.)
    merges_only = ops_sorted[ops_sorted["op"] == "merge"].copy()
    if merges_only.empty:
        print("[run] No MERGE operations in ops_csv; nothing to do.")
        # The baseline above may have booted the resident engine; this return
        # skips the end-of-run teardown, so release it here.
        if persistent_eval is not None:
            try:
                persistent_eval.shutdown()
            except Exception:
                pass
        if persist_candidate_dir is not None:
            shutil.rmtree(persist_candidate_dir, ignore_errors=True)
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

    # Replay must follow the recorded steps.csv (greedy accept) order, NOT the
    # ops-CSV order. The 7B clustering ops already happen to be in greedy order,
    # but the 32B ops are in plain layer order, so 'normal' sort would misalign
    # planned block k against replay_rows[k] and abort ("misaligned plan" -- the
    # B1 32B-replay gap). Reorder merges_only by each block's recorded step_idx;
    # every row sharing a (stage, group, layer) block gets the same rank, so
    # _plan_eval_blocks still groups them into one block (stable sort keeps the
    # within-block component order). No-op for runs whose ops already match.
    if replay_mode and replay_rows:
        _step_of: dict = {}
        for _r in replay_rows:
            _key = (int(_r["stage"]), str(_r["component"]), int(_r["layer"]))
            _step_of.setdefault(_key, int(_r["step_idx"]))
        _ranks = [
            _step_of.get((int(row.stage), str(row.group), int(row.layer)), 1 << 30)
            for row in merges_only.itertuples(index=False)
        ]
        merges_only = (merges_only.assign(_replay_rank=_ranks)
                       .sort_values("_replay_rank", kind="stable")
                       .drop(columns="_replay_rank").reset_index(drop=True))

    # --- Operation Summary ---
    print("\n" + "="*20 + " Operation Summary " + "="*20)
    print(f"Sort Mode: {sort_mode}")
    print(f"Total merge operations to process: {len(merges_only)}")
    if not merges_only.empty:
        print(merges_only[['stage', 'layer', 'group', 'component', 'models']].to_string())
    print("="*59 + "\n")

    planned_blocks = _plan_eval_blocks(merges_only, bundle_eval_mode=bundle_eval_mode)
    print(f"[bundle] Mode: {bundle_eval_mode}. Planned evaluation blocks: {len(planned_blocks)}")

    if replay_mode and len(planned_blocks) != len(replay_rows):
        raise RuntimeError(
            f"replay: the ops CSVs plan {len(planned_blocks)} blocks but "
            f"{replay_steps_csv} records {len(replay_rows)} steps. These describe different "
            f"runs, or --bundle_eval_mode/--sort_mode differ from the original."
        )

    step_counter = 0

    def _process_block(layer: int, group_name: str, indices: List[int], step_counter: int) -> int:
        if not indices:
            return step_counter
        block = merges_only.loc[indices]

        # Check stage of this block (assume homogeneous stage within a block)
        first_row = merges_only.loc[indices[0]]
        current_stage = first_row.get("stage", 1)

        replay_row = None
        if replay_mode:
            if step_counter >= len(replay_rows):
                raise RuntimeError(
                    f"replay: block {step_counter} has no matching row in {replay_steps_csv} "
                    f"(CSV has {len(replay_rows)} steps). The ops CSVs and the steps CSV "
                    f"describe different runs."
                )
            replay_row = replay_rows[step_counter]
            recorded = (int(replay_row["stage"]), str(replay_row["component"]), int(replay_row["layer"]))
            planned = (int(current_stage), str(group_name), int(layer))
            if recorded != planned:
                raise RuntimeError(
                    f"replay: block {step_counter} is {planned} but steps.csv records "
                    f"{recorded}. Refusing to build a model from a misaligned plan."
                )
            decision = str(replay_row["decision"]).strip().lower()
            past_cutoff = replay_cutoff is not None and int(replay_row["step_idx"]) > replay_cutoff
            if decision != "accepted" or past_cutoff:
                why = "past cutoff" if past_cutoff else decision
                print(f"[step {step_counter}] skip ({why}): {group_name} layer={layer}")
                return step_counter + 1

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
        
        scaling_status = "enabled" if enable_scaling else "disabled"

        if not valid_indices:
            print(
                f"  [skip] No valid operations for {target_label} at layer={layer}, "
                f"group={group_name} (stage={current_stage}, scaling={scaling_status})"
            )
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
        
        print(
            f"\n[step {step_counter}] target={target_label}, group={group_name}, "
            f"layer={layer} (stage={current_stage}, scaling={scaling_status})"
        )

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
                    loaded_models[lbl] = tk_load_model(
                        path,
                        device_hint="cpu",
                        dtype_hint=target_dtype,
                    )
            
            # Load target model
            model_tgt = tk_load_model(
                str(working_dir),
                device_hint="cpu",
                dtype_hint=target_dtype
            )

            # 3. Apply operations for each subcomponent in order
            if group_name == "attn":
                comp_order = ATTN_COMPONENT_ORDER
            elif group_name == "mlp":
                comp_order = MLP_COMPONENT_ORDER
            elif group_name == "layer_full":
                comp_order = FULL_LAYER_COMPONENT_ORDER
            else:
                comp_order = []

            applied_components: Dict[str, str] = {}
            # Sidecar capture staging: factors of THIS block's scaled ops.
            # Persisted only if the block is accepted/promoted; a reject or
            # eval failure simply drops them with the candidate.
            block_factors: Dict[Tuple[int, str], dict] = {}
            block_used_scaling = False
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
                    
                    if comp in ATTN_COMPONENT_TO_ATTR:
                        effective_group_name = "attn"
                        attr_name = ATTN_COMPONENT_TO_ATTR.get(comp)
                    elif comp in MLP_COMPONENT_TO_ATTR:
                        effective_group_name = "mlp"
                        attr_name = MLP_COMPONENT_TO_ATTR.get(comp)
                    else:
                        effective_group_name = group_name
                        attr_name = None
                    if attr_name:
                        applied_components[comp] = attr_name
                        # Replay must reproduce the recorded run's arithmetic exactly:
                        # auto-scaling is disabled there; only the explicit flag
                        # (--enable-scaling, or --scaling on) applies.
                        op_scaled = op_needs_scaling(
                            participants, enable_scaling,
                            auto_scale=not replay_mode,
                            mode=scaling_mode,
                        )
                        if op_scaled and not enable_scaling and scaling_mode == "auto":
                            print(f"    [scale] cross-family op detected "
                                  f"({sorted({infer_family_prefix(l) for l, _ in participants})}): "
                                  f"scaling ON for {comp}@L{layer}")
                        op_device = merge_device
                        if op_scaled and merge_device == "cpu" and torch.cuda.is_available():
                            # scaling is device-dependent; prefer cuda for scaled ops
                            op_device = "cuda"
                        if op_scaled:
                            block_used_scaling = True
                        _factors: Optional[dict] = {} if op_scaled else None
                        _n_way_average_subcomponent_into_target(
                            target_model=model_tgt,
                            target_model_path=str(working_dir),
                            target_layer_idx=layer,
                            group_name=effective_group_name,
                            component_attr=attr_name,
                            contributors=contributors,
                            device=op_device,
                            target_std=None,
                            enable_scaling=op_scaled,
                            factors_out=_factors,
                        )
                        if _factors:
                            block_factors[(int(layer), comp)] = _factors

            # 4. Save modified target model to temp (or, when persistent eval
            # is active, to the stable candidate dir the resident engine
            # reloads from).
            tmp_model_dir = _save_model_to_temp_dir_for_eval(
                model_tgt,
                base_model_path=str(working_dir),
                tmp_root=tmp_dir_root,
                changed_keys=save_utils.changed_keys_for(int(layer), applied_components),
                fixed_dir=persist_candidate_dir,
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

        # Replay: the step was already judged, in a previous run. Promote the candidate
        # without scoring it. The decision trace already exists in replay_steps_csv, so
        # nothing is logged here.
        if replay_mode:
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
                raise RuntimeError(f"replay: promote swap failed for {target_label}: {e}")
            else:
                shutil.rmtree(_prev, ignore_errors=True)
            # A promoted replay block is an accepted merge: persist its scaled
            # factors (commit BEFORE updating accepted_slots -- the overwrite
            # cause is judged against slots written by EARLIER accepts).
            _commit_scaling_factors(block_factors, step_counter)
            accepted_slots.update((int(layer), c) for c in applied_components)
            print(f"[step {step_counter}] apply: {group_name} layer={layer} "
                  f"(recorded score {float(replay_row['score']):.4f})")
            return step_counter + 1

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
                eval_split=eval_split,
                # Same per-model chat-template resolution as the baseline/final.
                registry_label=target_label,
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
                    "scaled": block_used_scaling,
                },
            )
            return step_counter + 1

        # Accept/reject
        old_acc = last_score[target_label]
        # drop_tolerance is specified in absolute percentage points (e.g., 1.5)
        allowed_drop = float(drop_tolerance)
        if new_acc + 1e-12 < (old_acc - allowed_drop):
            decision = "rejected"
            progress_tally["rejected"] += 1
            print(f"  [decision] {target_label}: REJECT (old={old_acc:.4f} new={new_acc:.4f})")
            try:
                shutil.rmtree(tmp_model_dir)
            except Exception:
                pass
        else:
            decision = "accepted"
            progress_tally["accepted"] += 1
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
            # Accepted: persist the block's scaled factors (commit BEFORE
            # updating accepted_slots -- the overwrite cause is judged against
            # slots written by EARLIER accepts), then record the slots this
            # block wrote (scaled or not: any accepted merge moves the slot's
            # trajectory off the original disk weights).
            _commit_scaling_factors(block_factors, step_counter)
            accepted_slots.update((int(layer), c) for c in applied_components)
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
                "scaled": block_used_scaling,
            },
        )
        return step_counter + 1

    
    progress_tally = {"accepted": 0, "rejected": 0}
    total_blocks = len(planned_blocks)
    run_start = time.monotonic()
    for block_no, block in enumerate(planned_blocks, start=1):
        step_counter = _process_block(
            int(block["layer"]),
            str(block["group"]),
            list(block["indices"]),
            step_counter,
        )
        elapsed = time.monotonic() - run_start
        print(
            f"[progress] block {block_no}/{total_blocks} | elapsed {elapsed:.1f}s | "
            f"accepted {progress_tally['accepted']} rejected {progress_tally['rejected']}"
        )

    if replay_mode and save_variant_dir:
        if os.path.exists(save_variant_dir):
            shutil.rmtree(save_variant_dir)
        os.makedirs(os.path.dirname(save_variant_dir.rstrip("/")) or ".", exist_ok=True)
        shutil.copytree(str(working_dir), save_variant_dir)
        with open(os.path.join(save_variant_dir, "micr_replay.json"), "w") as f:
            json.dump(
                {
                    "target_label": target_label,
                    "source_model": str(target_src),
                    "steps_csv": os.path.abspath(replay_steps_csv),
                    "cutoff_step_idx": replay_cutoff,
                    "blocks_applied": sum(
                        1 for r in replay_rows
                        if str(r["decision"]).strip().lower() == "accepted"
                        and (replay_cutoff is None or int(r["step_idx"]) <= replay_cutoff)
                    ),
                    "enable_scaling": enable_scaling,
                    "scaling_mode": scaling_mode,
                    "merge_device": merge_device,
                    # additive key; every consumer reads this file with .get()
                    "scaling_factors": (SCALING_SIDECAR_BASENAME
                                        if scaling_factors_run else None),
                },
                f,
                indent=2,
            )
        # Scaling-factor handoff: the replay re-executed every accepted op, so
        # the captured factors are complete for this variant. Ship them next to
        # micr_replay.json -- this is the finaleval handoff point. (.npz on
        # purpose: run_figures's finaleval stage deletes *.safetensors under
        # these dirs after recording the score; the sidecar must survive that.)
        _persist_scaling_sidecar(extra_dir=save_variant_dir)
        print(f"[replay] saved variant -> {save_variant_dir}")

    # Tear down the resident engine (if any) BEFORE the final full-set eval:
    # the final eval scores working_dir through a fresh subprocess, and a
    # still-resident engine holding ~90% of GPU memory would OOM it. Also
    # drop the now-stale candidate dir. (Mirrors gaussian_profiler's
    # end-of-sweep persistent_eval.shutdown().)
    #
    # The final eval staying on the SUBPROCESS path is deliberate, not an
    # engine-consistency violation: it runs on the FULL set and its number is
    # never compared against the in-run split-gating scores (baseline /
    # per-step, which are engine-consistent among themselves), so the
    # engine-vs-subprocess offset cannot contaminate any accept/reject
    # decision. Keeping it subprocess also keeps every headline full-set
    # number in the project (baselines stores, replay finals, paper tables)
    # measured the same way and mutually comparable.
    if persistent_eval is not None:
        try:
            persistent_eval.shutdown()
        except Exception:
            pass
    if persist_candidate_dir is not None:
        shutil.rmtree(persist_candidate_dir, ignore_errors=True)

    # Final evaluation summary for the target. Always on the FULL eval set:
    # when per-step gating ran on split 'P'/'M', the headline number must not
    # be computed on the items that drove the accept/reject decisions.
    if eval_enabled:
        if eval_split is not None:
            print(f"\n[final] scoring on the FULL eval set (per-step gating used split '{eval_split}')")
        final_acc = evaluate_with_retry(
            f"{target_label} final",
            model_path=str(working_dir),
            task_or_domain=target_domain_or_task,
            gpu_ids=gpu_ids,
            timeout_minutes=timeout_minutes,
            batch_size=batch_size,
            temperature=temperature,
            output_dir=os.path.join(output_dir, target_label, "final"),
            eval_split=None,
            # Headline full-set numbers stay on the subprocess path (same
            # rationale as the persistent-eval teardown above): every full-set
            # number in the project is measured the same way.
            allow_inprocess=False,
            # CONSISTENCY INVARIANT: the final full-set eval MUST use the same
            # per-model chat template as the baseline and per-step gating evals,
            # or the headline number would be prompted differently from the
            # scores that drove every accept/reject. Same target_label -> same
            # apply_chat_template value.
            registry_label=target_label,
        )
        if final_acc is None:
            print(f"\n[final] {target_label}: evaluation failed after retries; no score.")
        else:
            print(f"\n[final] {target_label} score: {final_acc:.4f}")
            # Replay runs: persist the full-set score with the recovered variant so the
            # number travels with the model instead of living only in a log.
            if replay_mode and save_variant_dir:
                meta_path = os.path.join(save_variant_dir, "micr_replay.json")
                try:
                    meta = json.load(open(meta_path))
                except Exception:
                    meta = {}
                meta["final_full_score"] = float(final_acc)
                meta["final_eval_split"] = "full"
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
                print(f"[replay] final full-set score recorded in {meta_path}")
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
    parser.add_argument(
        "--batch_size",
        default="auto",
        help="Evaluation batch size. Use an integer or 'auto'.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--drop_tolerance", type=float, default=2.0, help="Allowed absolute drop before rejecting a step")
    parser.add_argument("--sorted_ops_out", default="./sorted_ops.csv", help="Where to export sorted ops CSV")
    parser.add_argument("--no_eval", action="store_true", help="Disable evaluation (accept all steps without scoring)")
    parser.add_argument(
        "--sort_mode",
        choices=["normal", "separate", "together"],
        default="normal",
        help="Order of operations. 'normal': preserve CSV order exactly (default). "
             "'separate': sort each step by layer. "
             "'together': combine steps, de-duplicate favoring step2, then sort all by layer."
    )
    parser.add_argument(
        "--bundle_eval_mode",
        choices=["group", "auto", "layer_full"],
        default="auto",
        help=(
            "How to bundle contiguous CSV rows into one evaluation step. "
            "'group': evaluate per contiguous attn/mlp block. "
            "'auto': detect full 7-component layer bundles and evaluate them once, "
            "otherwise use group mode. "
            "'layer_full': always evaluate contiguous same-layer rows as one step."
        ),
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
        "--enable-scaling",
        dest="enable_scaling",
        action="store_true",
        # NOTE: even without this flag, scaling AUTO-ENABLES per-op for
        # cross-family merges (see op_needs_scaling). This flag forces it
        # for every op.
        help=(
            "Rescale merged subcomponents to match the original per-layer std devs. "
            "Uses row-wise affine moment matching against the target weight."
        ),
    )
    parser.add_argument(
        "--scaling",
        choices=["auto", "on", "off"],
        default="auto",
        help="Per-run scaling policy, decided per model SET. 'auto' (default): "
             "exact historical behavior (--enable-scaling plus per-op "
             "cross-family/AUTO_SCALE_FAMILIES auto-trigger; replay of OLD "
             "recorded runs via --enable-scaling stays byte-compatible). "
             "'on': scale every merge op (deepseek / qwen2.5-7b pairs). "
             "'off': never scale -- overrides the auto-trigger; chosen for "
             "same-base finetune sets (llama5, qwen3-32b trio).",
    )
    parser.add_argument(
        "--eval_split",
        choices=["P", "M", "full"],
        default="M",
        help="Score the baseline and every per-step gating eval on a materialized half of the "
             "task's eval set: 'M' is the MICR half, 'P' the profiler half (materialize them "
             "first with merge_tools/micr/eval_splits.py). The final summary eval always runs "
             "on the full set. Default 'full' scores everything on the full set and leaves all "
             "eval commands byte-identical to before this flag existed.",
    )
    parser.add_argument(
        "--merge_device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device for the merge arithmetic. auto (default) = cpu unless --enable-scaling. "
             "Without scaling, cpu and cuda are bitwise identical, cpu is faster, and it keeps "
             "this process off the GPU (a cuda merge holds a ~610MB context the evaluator cannot use). "
             "With scaling the result is device-dependent, so auto keeps cuda.",
    )

    replay = parser.add_argument_group(
        "replay",
        "Rebuild a variant from a finished run's steps.csv instead of evaluating each step. "
        "Only 'accepted' blocks are applied, in step_idx order; rejected blocks were reverted "
        "by the original run and leave no trace, so replaying the accepted ones reproduces its "
        "weights. Per-step evaluation is skipped entirely; the final eval still runs unless "
        "--no_eval is given.",
    )
    replay.add_argument(
        "--replay_steps_csv",
        default=None,
        help="A steps.csv produced by a previous run of this script.",
    )
    replay.add_argument(
        "--replay_cutoff",
        type=int,
        default=None,
        help="Inclusive. Apply only accepted steps at or before this cutoff. "
             "Omit to apply every accepted step.",
    )
    replay.add_argument(
        "--replay_cutoff_mode",
        choices=["step_idx", "line"],
        default="step_idx",
        help="How to read --replay_cutoff. 'step_idx' (default) matches "
             "visualization/plot_steps.py, whose apply_step_cutoff filters df.step_idx <= cutoff. "
             "'line' matches scripts/formatter.py, whose cutoffs are 1-based CSV line numbers; "
             "since the first data row is the baseline (step_idx=-1), line == step_idx + 3.",
    )
    replay.add_argument(
        "--save_variant_dir",
        default=None,
        help="Copy the final replayed model here (replay mode only). Unless --no_eval is "
             "given, the recovered variant is then scored once on the FULL eval set and the "
             "score is recorded into <dir>/micr_replay.json (final_full_score). Pass "
             "--no_eval for a rebuild-only run with no evaluation at all.",
    )
    args = parser.parse_args()
    if args.replay_cutoff is not None and args.replay_steps_csv is None:
        parser.error("--replay_cutoff requires --replay_steps_csv")
    if args.save_variant_dir and args.replay_steps_csv is None:
        parser.error("--save_variant_dir requires --replay_steps_csv")
    if args.scaling == "off" and args.enable_scaling:
        parser.error("--scaling off contradicts --enable-scaling")

    eval_enabled = not args.no_eval
    # --scaling on is equivalent to --enable-scaling; setting the flag keeps
    # every downstream consumer of enable_scaling (merge-device resolution,
    # the [scale] banner, micr_replay.json) consistent with the forced mode.
    enable_scaling = args.enable_scaling or args.scaling == "on"

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
        bundle_eval_mode=args.bundle_eval_mode,
        ignore_other_families=args.ignore_other_families,
        initial_baseline=args.initial_baseline,
        force_calc_baseline=args.force_calc_baseline,
        enable_scaling=enable_scaling,
        merge_device=args.merge_device,
        replay_steps_csv=args.replay_steps_csv,
        replay_cutoff=args.replay_cutoff,
        replay_cutoff_mode=args.replay_cutoff_mode,
        save_variant_dir=args.save_variant_dir,
        eval_split=args.eval_split,
        scaling_mode=args.scaling,
    )


if __name__ == "__main__":
    main()
