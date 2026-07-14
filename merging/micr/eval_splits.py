#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eval-split machinery: deterministic P/M partitions of each task's eval set.

Why
---
The MICR workflow reuses one eval set three times: the gaussian profiler ranks
layers with it, MICR's per-step gating accepts/rejects merges with it, and the
final headline number is computed on it. That lets the gating overfit the very
items the profile came from, and the headline number is then measured on items
that were used to make decisions. This module splits every task's eval set into
two seeded, disjoint, exhaustive halves:

    P  ("profiler")  -- gaussian_profiler.py --eval_split P
    M  ("micr")      -- micr/run_eval.py     --eval_split M   (per-step gating)
    full             -- default everywhere; the final eval always uses it

Nothing here changes any score arithmetic: a split run simply evaluates the
same task on a subset of its documents. When no split is requested, every
command the evaluators build is byte-identical to before this module existed.

How each harness consumes a split
---------------------------------
* math-evaluation-harness (gsm8k-cot / gsm8k-pal): `materialize()` writes
  ``data/gsm8k/testP.jsonl`` / ``testM.jsonl`` next to the vendored harness's
  ``test.jsonl`` (subset rows, original order). The evaluator then passes
  ``--split testP``; the harness's data_loader reads
  ``{data_dir}/gsm8k/{split}.jsonl`` directly.

* lm_eval tasks (mmlu_*, sst2, humaneval, ...): `materialize()` writes, into a
  configs directory (default: ``merge_tools/micr/eval_split_configs``),
    - a task-variant YAML ``<task>_<SPLIT>.yaml`` that ``include:``s the
      installed task's own YAML and overrides ``process_docs``, and
    - a generated hook module ``_micr_split_<task>_<SPLIT>.py`` whose
      ``process_docs(dataset)`` selects the partition's indices.
  The evaluator then runs ``lm_eval --tasks <task>_<SPLIT> --include_path
  <configs dir>``. This is the stock lm_eval custom-task mechanism: the
  ``!function module.fn`` YAML tag loads ``module.py`` from the directory the
  YAML lives in, and ``process_docs`` is applied to each raw dataset split
  before docs are built. lm_eval calls ``process_docs`` on *every* split it
  touches (fewshot/dev splits included), so the generated hook checks the
  dataset's split name and only partitions the split that is actually scored
  (``test_split``, else ``validation_split``); other splits -- e.g. MMLU's
  5-doc ``dev`` fewshot pool -- pass through untouched. If the parent task
  already defines ``process_docs``, the hook applies it first and partitions
  the processed docs, so indices always refer to the final doc order.

Determinism / audit
-------------------
`make_partition(task, n_items, seed, fractions)` shuffles ``range(n_items)``
with ``random.Random(f"micr-eval-split|{task}|{seed}")`` (string seeding is
stable across runs and CPython versions) and cuts it once; P and M are
disjoint, exhaustive, and returned sorted. `split_spec()` serializes a
run_config-friendly dict (task, seed, fractions, sizes, sha256 index hashes).
`materialize()` writes the spec (plus the exact indices) to a ``*.spec.json``
next to whatever it generated; runners print it via `load_recorded_spec()`.

CLI
---
    python merge_tools/micr/eval_splits.py --task gsm8k-cot          # both halves
    python merge_tools/micr/eval_splits.py --task mmlu_econometrics --splits P,M
    python merge_tools/micr/eval_splits.py --task humaneval --seed 0
"""
import argparse
import hashlib
import inspect
import json
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

# Allow running this file directly (python merge_tools/micr/eval_splits.py) as
# well as importing it as merge_tools.micr.eval_splits. Same bootstrap as
# run_eval.py.
import types as _types
_MT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "merge_tools" not in sys.modules:
    _mt_pkg = _types.ModuleType("merge_tools")
    _mt_pkg.__path__ = [_MT_ROOT]
    sys.modules["merge_tools"] = _mt_pkg

SPLIT_NAMES = ("P", "M")

# gsm8k-cot and gsm8k-pal share one data file (data/gsm8k/test.jsonl), so they
# share one partition, keyed canonically.
_MATH_TASKS = ("gsm8k-cot", "gsm8k-pal", "gsm8k")
_MATH_PARTITION_KEY = "gsm8k"


# --------------------------------------------------------------------- basics

def normalize_eval_split(value) -> Optional[str]:
    """Map user input to 'P', 'M', or None (= full set). Raises on junk."""
    if value is None:
        return None
    v = str(value).strip()
    if v.lower() in ("", "none", "full"):
        return None
    if v.upper() in SPLIT_NAMES:
        return v.upper()
    raise ValueError(
        f"eval_split must be one of {SPLIT_NAMES + ('full',)} or None; got {value!r}"
    )


def make_partition(task, n_items, seed, fractions=(0.5, 0.5)):
    """
    Deterministic disjoint exhaustive partition of range(n_items) into
    {'P': [...], 'M': [...]} (each sorted ascending).

    NOTE: this function's source is embedded verbatim (via inspect.getsource)
    into generated lm_eval hook modules, so it must stay self-contained: only
    the `random` module, no other names from this file.
    """
    n = int(n_items)
    if n < 2:
        raise ValueError(f"need at least 2 items to split, got {n_items!r}")
    f = tuple(float(x) for x in fractions)
    if len(f) != 2 or min(f) <= 0.0 or abs(sum(f) - 1.0) > 1e-9:
        raise ValueError(f"fractions must be two positive numbers summing to 1, got {fractions!r}")
    rng = random.Random(f"micr-eval-split|{task}|{int(seed)}")
    idx = list(range(n))
    rng.shuffle(idx)
    n_p = int(round(n * f[0]))
    n_p = min(max(n_p, 1), n - 1)  # both halves non-empty
    return {"P": sorted(idx[:n_p]), "M": sorted(idx[n_p:])}


def _index_hash(indices: Sequence[int]) -> str:
    return hashlib.sha256(",".join(map(str, indices)).encode("utf-8")).hexdigest()[:16]


def split_spec(task, seed, fractions=(0.5, 0.5), n_items=None, partition=None) -> Dict:
    """Run_config-friendly description of a partition (hashes, not indices)."""
    f = tuple(float(x) for x in fractions)
    spec: Dict = {
        "task": task,
        "seed": int(seed),
        "fractions": list(f),
        "algorithm": (
            "random.Random(f'micr-eval-split|{task}|{seed}') shuffle of range(n); "
            "P = first round(n*fractions[0]) (clamped to [1, n-1]), M = the rest; "
            "both sorted ascending"
        ),
        "index_hash_algorithm": "sha256(','.join(indices)).hexdigest()[:16]",
    }
    if partition is None and n_items is not None:
        partition = make_partition(task, n_items, seed, f)
    if partition is not None:
        spec["n_items"] = sum(len(v) for v in partition.values())
        spec["split_sizes"] = {k: len(v) for k, v in partition.items()}
        spec["index_hashes"] = {k: _index_hash(v) for k, v in partition.items()}
    else:
        spec["n_items"] = None
        spec["split_sizes"] = None
        spec["index_hashes"] = None
        spec["note"] = (
            "n_items unknown at materialize time; the generated process_docs hook "
            "recomputes the identical partition from the dataset length at eval time"
        )
    return spec


def default_lm_eval_config_dir() -> str:
    """Where generated lm_eval task-variant YAMLs live (--include_path target)."""
    env = os.environ.get("MICR_EVAL_SPLIT_CONFIG_DIR")
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_split_configs")


def _eval_harness_module():
    try:
        from merge_tools.micr import eval_harness as eh  # type: ignore
        return eh
    except ImportError:
        import importlib.util
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_harness.py")
        spec = importlib.util.spec_from_file_location("_micr_eval_harness_for_splits", path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def math_split_file(split_name: str, math_dir: Optional[str] = None) -> str:
    """Path the math harness will read for --split test<SPLIT>."""
    if math_dir is None:
        math_dir = _eval_harness_module().math_harness_dir()
    return os.path.join(math_dir, "data", "gsm8k", f"test{split_name}.jsonl")


# ------------------------------------------------------------- materialize: math

def _materialize_math(split_name: str, seed: int, fractions, out_dir: Optional[str]) -> Dict:
    math_dir = _eval_harness_module().math_harness_dir()
    src = os.path.join(math_dir, "data", "gsm8k", "test.jsonl")
    if not os.path.exists(src):
        raise FileNotFoundError(f"vendored gsm8k test set not found: {src}")
    with open(src, "r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    n = len(lines)
    partition = make_partition(_MATH_PARTITION_KEY, n, seed, fractions)

    # Default destination is the harness's own data dir: math_eval.py reads
    # {data_dir}/gsm8k/{split}.jsonl with data_dir=./data relative to the
    # harness checkout, and the evaluator only passes --split. An out_dir
    # override is for testing; the harness will not see it without --data_dir.
    data_dir = out_dir if out_dir else os.path.join(math_dir, "data")
    dest_dir = os.path.join(data_dir, "gsm8k")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f"test{split_name}.jsonl")
    indices = partition[split_name]
    with open(dest, "w", encoding="utf-8") as f:
        for i in indices:
            ln = lines[i]
            f.write(ln if ln.endswith("\n") else ln + "\n")

    spec = split_spec(_MATH_PARTITION_KEY, seed, fractions, partition=partition)
    spec.update({
        "kind": "math-harness-jsonl",
        "split": split_name,
        "source": src,
        "materialized": dest,
        "harness_split_arg": f"test{split_name}",
    })
    spec_path = os.path.join(dest_dir, f"test{split_name}.spec.json")
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump({**spec, "indices": indices}, f, indent=2)
    spec["spec_file"] = spec_path
    return spec


# ---------------------------------------------------------- materialize: lm_eval

_TASK_MANAGER = None


def _task_manager():
    """One TaskManager per process; its init walks every installed task YAML."""
    global _TASK_MANAGER
    if _TASK_MANAGER is None:
        try:
            from lm_eval.tasks import TaskManager  # type: ignore
        except ImportError as e:
            raise RuntimeError(f"lm_eval is not importable; cannot materialize lm_eval splits: {e}")
        _TASK_MANAGER = TaskManager()
    return _TASK_MANAGER


def _find_task_yaml(task: str) -> str:
    """Locate the installed lm_eval YAML that defines `task`."""
    tm = _task_manager()
    entry = tm.task_index.get(task)
    if entry is None:
        raise ValueError(f"Task '{task}' not found in the installed lm_eval task index.")
    if entry.get("type") != "task":
        raise ValueError(
            f"'{task}' is registered as a '{entry.get('type')}', not an individual task; "
            "eval splits need a single task config."
        )
    yaml_path = entry.get("yaml_path")
    if not yaml_path or yaml_path == -1:
        raise ValueError(f"Task '{task}' has no YAML config on disk; cannot derive a split variant.")
    return os.path.abspath(yaml_path)


def _merged_task_config(yaml_path: str) -> Dict:
    from lm_eval import utils as lm_utils  # type: ignore
    return lm_utils.load_yaml_config(yaml_path, mode="simple")


def _load_raw_yaml(yaml_path: str) -> Dict:
    """Parse one YAML file, keeping !function references as plain strings."""
    import yaml

    class _Loader(yaml.SafeLoader):
        pass

    _Loader.add_constructor("!function", lambda loader, node: loader.construct_scalar(node))
    with open(yaml_path, "rb") as f:
        return yaml.load(f, Loader=_Loader) or {}


def _find_parent_process_docs(yaml_path: str) -> Optional[Tuple[str, str]]:
    """
    Walk a task YAML's include chain and return (dir_of_declaring_yaml,
    "module.function") for its process_docs, or None. Mirrors
    lm_eval.utils.load_yaml_config merge order: the child wins, then earlier
    entries in an include list win over later ones.
    """
    cfg = _load_raw_yaml(yaml_path)
    if "process_docs" in cfg:
        ref = cfg["process_docs"]
        if not isinstance(ref, str) or "." not in ref:
            raise RuntimeError(
                f"Unsupported process_docs value {ref!r} in {yaml_path}; expected '!function module.fn'."
            )
        return (os.path.dirname(os.path.abspath(yaml_path)), ref)
    inc = cfg.get("include")
    if inc:
        paths = [inc] if isinstance(inc, str) else list(inc)
        for p in paths:
            full = p if os.path.isfile(p) else os.path.join(os.path.dirname(yaml_path), p)
            found = _find_parent_process_docs(full)
            if found is not None:
                return found
    return None


_HOOK_MODULE_TEMPLATE = '''\
"""Auto-generated by merge_tools/micr/eval_splits.py -- do not edit by hand.

lm_eval `process_docs` hook selecting eval-split {split_name!r} of task {task!r}.
Referenced from {variant}.yaml as `!function {module_name}.process_docs`.
"""
import importlib.util
import os
import random
import sys

TASK = {task!r}
SPLIT_NAME = {split_name!r}
EVAL_SPLIT = {eval_split!r}          # name of the dataset split that gets scored
SEED = {seed!r}
FRACTIONS = {fractions!r}
EXPECTED_N = {n_items!r}             # None => size is read from the dataset at eval time
FIXED_INDICES = {fixed_indices!r}    # None => recomputed from (TASK, n, SEED, FRACTIONS)
PARENT_PROCESS_DOCS = {parent_pd!r}  # (module_dir, "module.function") or None


# ---- partition algorithm: embedded verbatim from eval_splits.make_partition ----
{make_partition_src}

def _parent_fn():
    module_dir, ref = PARENT_PROCESS_DOCS
    module_name, _, fn_name = ref.rpartition(".")
    path = os.path.join(module_dir, *module_name.split(".")) + ".py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load parent process_docs module {{path}}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, fn_name)


def process_docs(dataset):
    # lm_eval applies process_docs to every raw split it touches (fewshot/dev
    # pools included). Partition only the split that is actually scored; any
    # parent preprocessing still applies to all splits, exactly as it did for
    # the unsplit task.
    raw_split = getattr(dataset, "split", None)
    if PARENT_PROCESS_DOCS is not None:
        dataset = _parent_fn()(dataset)
    if raw_split is None:
        print(
            f"[eval-split] WARNING: dataset for task {{TASK}} carries no split name; "
            f"leaving it unpartitioned (full set).",
            file=sys.stderr,
        )
        return dataset
    if str(raw_split) != EVAL_SPLIT:
        return dataset
    n = len(dataset)
    if EXPECTED_N is not None and n != EXPECTED_N:
        raise ValueError(
            f"eval-split {{SPLIT_NAME}} of {{TASK}} was materialized for {{EXPECTED_N}} docs "
            f"but the dataset has {{n}}; re-run merge_tools/micr/eval_splits.py."
        )
    if FIXED_INDICES is not None:
        indices = FIXED_INDICES
    else:
        indices = make_partition(TASK, n, SEED, FRACTIONS)[SPLIT_NAME]
    return dataset.select(indices)
'''

_VARIANT_YAML_TEMPLATE = """\
# Auto-generated by merge_tools/micr/eval_splits.py -- do not edit by hand.
# Eval-split variant of task {task!r}: split {split_name!r}, seed={seed}, fractions={fractions}.
# Consume with: lm_eval --tasks {variant} --include_path {cfg_dir}
include: {parent_yaml_json}
task: {variant}
tag: micr_eval_split
process_docs: !function {module_name}.process_docs
{extra}"""


# ---- task-specific extra YAML appended to generated variants ----------------
# ifeval / Light-IF-32B (Qwen3-based), decided 2026-07: score with
# --apply_chat_template (raw prompts make the model CONTINUE the prompt
# instead of answering; measured ~21% prompt_level_strict = noise). Under the
# chat template every response opens with a <think>...</think> block whose
# prose violates ifeval's format checks (measured 18%), so the variants add a
# filter that strips a LEADING think block from each response BEFORE
# process_results. lm_eval 0.4.9.2 mechanics (verified): a task's filter_list
# REPLACES the default 'none' filter; FilterEnsemble output lands in
# instance.filtered_resps[<name>], which the evaluator feeds to
# process_results, and metric keys become "<metric>,strip_think" (the suffix
# is the FILTER name -- keep micr/persistent_eval's metric priority in sync).
# RegexFilter extracts group group_select from findall and .strip()s it; this
# pattern matches EVERY response (the think part is optional), so no-think
# responses pass through whole (modulo the .strip()) and the fallback
# "[invalid]" is unreachable.
#
# max_gen_toks is raised from the stock 1280: the think block alone exhausts
# that budget (observed on saved Light-IF samples: 4/4 responses truncated
# mid-<think>, no closing tag -- unrecoverable by stripping). YAML
# include-merge replaces generation_kwargs WHOLESALE, so the stock keys are
# restated verbatim alongside the raised cap.
THINK_STRIP_REGEX = r"(?s)^(?:\s*<think>.*?</think>\s*)?(.*)$"
IFEVAL_MAX_GEN_TOKS = 8192
_IFEVAL_EXTRA_YAML = """\
filter_list:
  - name: strip_think
    filter:
      - function: regex
        regex_pattern: '(?s)^(?:\\s*<think>.*?</think>\\s*)?(.*)$'
        group_select: 0
      - function: take_first
generation_kwargs:
  until: []
  do_sample: false
  temperature: 0.0
  max_gen_toks: 8192
"""
# tinyMMLU / Light-IF-32B, decided 2026-07: score the P/M split variants with
# PLAIN accuracy, dropping tinyBenchmarks' IRT aggregation. tinyMMLU is 100
# IRT-calibrated questions and its stock metric is `acc_norm` aggregated by
# `!function agg_functions.agg_gpirt_mmlu` (tinyBenchmarks' g-PIRT estimator).
# That estimator hardcodes IRT parameters tied to ALL 100 questions in their
# original order, so it only accepts a 100-length item vector: on our 50-item
# P/M split it raises `IndexError: index 50 is out of bounds for axis 0 with
# size 50` inside tb.evaluate (agg_functions.agg_gpirt_mmlu). A 50-question
# subset therefore CANNOT use the IRT metric. The split variants override
# metric_list with plain `acc` (mean aggregation) -- the standard lm_eval
# multiple_choice metric (see tasks/mmlu default_template acc). lm_eval
# include-merge (utils.load_yaml_config: final.update(child)) REPLACES the
# parent's metric_list WHOLESALE, so the IRT acc_norm row is gone and no IRT
# code runs. output_type stays `multiple_choice` (inherited via include), so
# `acc` is real MC accuracy and the reported key is `acc,none` (default filter).
# NOTE: only the SPLIT variants get this override; the unsplit `tinyMMLU` task
# is used directly with its IRT metric everywhere else (tinyMMLU is not in
# FULLSET_VARIANTS), so the non-split path is unchanged. MICR/profiler already
# consume `acc,none`: eval_harness.parse_score's tiny* branch tries acc_norm
# first then falls through to the generic acc patterns, and
# inprocess_eval._extract_score's tiny* order is ("acc_norm,none","acc,none").
_TINYMMLU_EXTRA_YAML = """\
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
"""
# medqa scores at chance (~0.30) at 0-shot; 1-shot ~= 5-shot accuracy at far
# lower cost (verified: 62.45% vs 62.3%). The installed medqa.yaml parent carries
# no num_fewshot, so the split/regenerated variant must set it explicitly -- else
# regeneration silently reverts medqa to 0-shot chance.
_MEDQA_EXTRA_YAML = "num_fewshot: 1\n"
_TASK_EXTRA_YAML = {"ifeval": _IFEVAL_EXTRA_YAML, "tinyMMLU": _TINYMMLU_EXTRA_YAML,
                    "medqa_4options": _MEDQA_EXTRA_YAML}

# Full-set (no split) variant names for tasks that need the extra YAML even
# without a P/M split -- consumed by micr/eval_harness.py's command builders
# (LM_EVAL_FULLSET_TASK_OVERRIDES; keep the two maps in sync).
FULLSET_VARIANTS = {"ifeval": "ifeval_nothink", "medqa_4options": "medqa_4options_fs"}

_FULLSET_VARIANT_YAML_TEMPLATE = """\
# Auto-generated by merge_tools/micr/eval_splits.py -- do not edit by hand.
# Full-set (no split) variant of task {task!r} carrying the same task-specific
# overrides as the generated split variants (see _TASK_EXTRA_YAML).
# Consume with: lm_eval --tasks {variant} --include_path {cfg_dir}
include: {parent_yaml_json}
task: {variant}
tag: micr_eval_split
{extra}"""


def materialize_fullset_variant(task: str, out_dir: Optional[str] = None) -> Optional[str]:
    """Write the full-set variant YAML for tasks in FULLSET_VARIANTS.

    Called automatically whenever a split of the task is materialized (so a
    fresh artifact clone regenerates it alongside <task>_P/<task>_M), and
    callable directly. Returns the YAML path, or None for tasks without a
    full-set variant.
    """
    variant = FULLSET_VARIANTS.get(task)
    if variant is None:
        return None
    cfg_dir = os.path.abspath(out_dir if out_dir else default_lm_eval_config_dir())
    os.makedirs(cfg_dir, exist_ok=True)
    parent_yaml = _find_task_yaml(task)
    yaml_path = os.path.join(cfg_dir, f"{variant}.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(_FULLSET_VARIANT_YAML_TEMPLATE.format(
            task=task,
            variant=variant,
            cfg_dir=cfg_dir,
            parent_yaml_json=json.dumps(parent_yaml),
            extra=_TASK_EXTRA_YAML[task],
        ))
    return yaml_path


def _materialize_lm_eval(task: str, split_name: str, seed: int, fractions,
                         out_dir: Optional[str], n_items: Optional[int]) -> Dict:
    cfg_dir = os.path.abspath(out_dir if out_dir else default_lm_eval_config_dir())
    os.makedirs(cfg_dir, exist_ok=True)

    parent_yaml = _find_task_yaml(task)
    merged = _merged_task_config(parent_yaml)
    eval_split = merged.get("test_split") or merged.get("validation_split")
    if not eval_split:
        raise RuntimeError(
            f"Task '{task}' ({parent_yaml}) declares neither test_split nor validation_split; "
            "cannot tell which dataset split gets scored."
        )
    parent_pd = _find_parent_process_docs(parent_yaml)

    fixed_indices = None
    partition = None
    if n_items is not None:
        partition = make_partition(task, n_items, seed, fractions)
        fixed_indices = partition[split_name]

    variant = f"{task}_{split_name}"
    module_name = f"_micr_split_{task}_{split_name}"
    module_path = os.path.join(cfg_dir, f"{module_name}.py")
    yaml_path = os.path.join(cfg_dir, f"{variant}.yaml")

    with open(module_path, "w", encoding="utf-8") as f:
        f.write(_HOOK_MODULE_TEMPLATE.format(
            task=task,
            split_name=split_name,
            variant=variant,
            module_name=module_name,
            eval_split=str(eval_split),
            seed=int(seed),
            fractions=tuple(float(x) for x in fractions),
            n_items=n_items,
            fixed_indices=fixed_indices,
            parent_pd=parent_pd,
            make_partition_src=inspect.getsource(make_partition),
        ))
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(_VARIANT_YAML_TEMPLATE.format(
            task=task,
            split_name=split_name,
            seed=int(seed),
            fractions=tuple(float(x) for x in fractions),
            variant=variant,
            cfg_dir=cfg_dir,
            parent_yaml_json=json.dumps(parent_yaml),
            module_name=module_name,
            # empty for tasks without an entry -> byte-identical output
            extra=_TASK_EXTRA_YAML.get(task, ""),
        ))
    # Side effect for tasks with a full-set variant (ifeval): regenerate it in
    # lock-step with the split variants, so a fresh clone gets all three.
    materialize_fullset_variant(task, out_dir=cfg_dir)

    spec = split_spec(task, seed, fractions, n_items=n_items, partition=partition)
    spec.update({
        "kind": "lm_eval-config",
        "split": split_name,
        "variant_task": variant,
        "eval_split_name": str(eval_split),
        "parent_yaml": parent_yaml,
        "config_yaml": yaml_path,
        "hook_module": module_path,
        "include_path": cfg_dir,
        "parent_process_docs": list(parent_pd) if parent_pd else None,
    })
    spec_path = os.path.join(cfg_dir, f"{variant}.spec.json")
    payload = dict(spec)
    if fixed_indices is not None:
        payload["indices"] = fixed_indices
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    spec["spec_file"] = spec_path
    return spec


# --------------------------------------------------------------------- public API

def materialize(task: str, split_name: str, out_dir: Optional[str] = None, *,
                seed: int = 0, fractions=(0.5, 0.5), n_items: Optional[int] = None) -> Dict:
    """
    Write the artifacts that make `--eval_split <split_name>` work for `task`,
    and return the split spec (also persisted as a *.spec.json).

    gsm8k-cot / gsm8k-pal / gsm8k -> subset jsonl in the vendored math
    harness's data dir (out_dir overrides the harness data dir; testing only).
    Anything else -> lm_eval task-variant YAML + process_docs hook module in
    the configs dir (out_dir overrides default_lm_eval_config_dir()).

    n_items (lm_eval tasks only): pin the expected eval-set size; bakes the
    exact indices into the hook, which then hard-fails on any size mismatch.
    """
    split = normalize_eval_split(split_name)
    if split is None:
        raise ValueError(f"materialize needs split 'P' or 'M', got {split_name!r}")
    if task in _MATH_TASKS:
        return _materialize_math(split, seed, fractions, out_dir)
    return _materialize_lm_eval(task, split, seed, fractions, out_dir, n_items)


def load_recorded_spec(task: str, split_name: str,
                       config_dir: Optional[str] = None) -> Optional[Dict]:
    """Read back the spec written by materialize(), or None if absent."""
    split = normalize_eval_split(split_name)
    if split is None:
        return None
    if task in _MATH_TASKS:
        math_dir = _eval_harness_module().math_harness_dir()
        path = os.path.join(math_dir, "data", "gsm8k", f"test{split}.spec.json")
    else:
        path = os.path.join(config_dir or default_lm_eval_config_dir(),
                            f"{task}_{split}.spec.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            spec = json.load(f)
    except Exception:
        return None
    spec.pop("indices", None)  # keep it run_config-sized; hashes identify the sets
    return spec


# --------------------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Materialize deterministic P/M eval splits for MICR / gaussian profiler.",
    )
    ap.add_argument("--task", required=True,
                    help="Registry task name (e.g. gsm8k-cot, mmlu_econometrics, humaneval).")
    ap.add_argument("--splits", default="P,M", help="Comma-separated split names. Default: P,M")
    ap.add_argument("--seed", type=int, default=0, help="Partition seed. Default: 0")
    ap.add_argument("--fractions", default="0.5,0.5",
                    help="P,M fractions summing to 1. Default: 0.5,0.5")
    ap.add_argument("--out_dir", default=None,
                    help="Override output dir (math: harness data dir; lm_eval: configs dir).")
    ap.add_argument("--n_items", type=int, default=None,
                    help="lm_eval tasks only: pin the eval-set size and bake fixed indices.")
    args = ap.parse_args()

    fractions = tuple(float(x) for x in args.fractions.split(","))
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        spec = materialize(args.task, split, args.out_dir,
                           seed=args.seed, fractions=fractions, n_items=args.n_items)
        print(json.dumps(spec, indent=2))


if __name__ == "__main__":
    main()
