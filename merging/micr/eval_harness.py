"""
MICR's evaluation layer, self-contained. Replaces the `unified-llm-eval` checkout.

What `unified-llm-eval` actually was, for MICR's purposes:
  * a task registry mapping task -> evaluator class
  * two evaluator classes that build a subprocess argv and regex a score from stdout
  * an EnvironmentManager that ran that argv via `conda run -n <name>`

That is about 200 lines, and it is ported here **verbatim** -- same argv, same
`parse_score` regexes -- so the numbers do not move. Two things change:

  1. No conda environment name. The eval runs with ``sys.executable``, i.e. the
     interpreter MICR is already running in. The `harness_env` / `languages_env`
     names never existed on this machine anyway; pointing them at the current env
     reproduced gsm8k 81.3 and humaneval 68.9, both in-band.
  2. No `UNIFIED_LLM_EVAL_ROOT` path dependency.

What is NOT ported, because it cannot be: the vendored **math-evaluation-harness**
(41 MB) defines the math metric. `lm_eval`'s own `gsm8k_cot` scores the same model
**63.61** instead of 81.2, because its regex cannot read ``$\\boxed{460}$`` and extracts
the literal string ``'$.'``. Point `MICR_MATH_HARNESS_DIR` at that checkout.

`vendor/HumanEval` is not needed at all: `humaneval` runs through `lm_eval`.

Environment:
    MICR_MATH_HARNESS_DIR   path to math-evaluation-harness (required for --domain math)

Eval splits (opt-in):
    eval_settings["eval_split"] in {"P", "M"} evaluates on a materialized half
    of the task's eval set (see micr/eval_splits.py). math path: adds
    ``--split test<SPLIT>``; lm_eval path: swaps the task for its generated
    ``<task>_<SPLIT>`` variant and adds ``--include_path <configs dir>``
    (override the dir with eval_settings["eval_split_config_dir"]).
    Absent / None / "full" leaves every command byte-identical to before this
    feature existed.
"""
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from typing import Optional

# The math harness is vendored at merge_tools/vendor/math-evaluation-harness. It runs as a
# subprocess with cwd set to this directory; math_eval.py writes generations under its
# ./output/ (fixed filename, always --overwrite). Because --overwrite empties
# `processed_samples`, the score is computed from this run's generations in memory -- two
# concurrent MICR runs race on that artifact file, never on the number.
DEFAULT_MATH_HARNESS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "math-evaluation-harness"
)


def math_harness_dir(env_config: Optional[dict] = None) -> str:
    if env_config and env_config.get("math_harness_dir"):
        return env_config["math_harness_dir"]
    return os.environ.get("MICR_MATH_HARNESS_DIR", DEFAULT_MATH_HARNESS_DIR)


# ------------------------------------------------------------- registry caps
# Optional per-task example cap for the lm_eval (SimpleEvaluator) tasks,
# declared in the model registry (models_download/hf_repos.json) as an entry
# field "eval_limit". Consumed by both runners (gaussian_profiler.evaluate_model
# and run_eval.evaluate_model_for_task), which feed it into
# eval_settings["limit"] -> lm_eval's --limit. NO default cap exists anywhere:
# a registry without eval_limit fields leaves every command byte-identical.

_MODEL_REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models_download", "hf_repos.json",
)
_eval_limit_cache: Optional[tuple] = None  # (registry path, {task: limit})


def registry_eval_limit(registry_task_name: str, registry_path: Optional[str] = None):
    """
    The registry's optional example cap for ``registry_task_name``, or None.

    Reads models_download/hf_repos.json (override the file with the
    MICR_MODEL_REGISTRY env var) and returns the "eval_limit" of the entry
    whose "task" equals ``registry_task_name``. If several entries declare a
    cap for the same task, the last one in file order wins.

    Semantics of the value = lm_eval --limit, i.e. a DETERMINISTIC PREFIX of
    the task's docs: an int >= 1 caps to the first N examples, a float in
    (0, 1) to that fraction. Because the P/M split variants each materialize
    disjoint doc lists, prefix caps on P and M stay disjoint automatically.
    Only SimpleEvaluator (lm_eval) commands consume eval_settings["limit"];
    the math harness and humaneval ignore it.

    Any problem (missing/unreadable registry, non-numeric value) returns None
    so callers change nothing.
    """
    global _eval_limit_cache
    path = registry_path or os.environ.get("MICR_MODEL_REGISTRY") or _MODEL_REGISTRY_PATH
    if _eval_limit_cache is None or _eval_limit_cache[0] != path:
        table = {}
        try:
            with open(path) as fh:
                registry = json.load(fh)
            for _label, entry in registry.items():
                if not isinstance(entry, dict):
                    continue
                task, lim = entry.get("task"), entry.get("eval_limit")
                if task and lim is not None:
                    table[str(task)] = lim
        except Exception:
            table = {}
        _eval_limit_cache = (path, table)
    value = _eval_limit_cache[1].get(registry_task_name)
    if value is None:
        return None
    try:
        return int(value) if float(value) >= 1 and float(value) == int(value) else float(value)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------- per-model chat template flag
# Optional PER-MODEL control, declared in the same registry
# (models_download/hf_repos.json) as a boolean entry field
# "apply_chat_template". Some instruct models score at chance on MC/loglikelihood
# tasks unless their prompts go through the chat template (measured:
# Llama-3.1-8B-UltraMedical on medqa_4options 27.83 raw vs ~70%+ templated),
# while other models on the SAME task handle raw prompts fine (MedGo on
# medqa_4options 76.26 raw) -- so the lever must be per MODEL, not per task.
# Read by both runners (gaussian_profiler.evaluate_model and
# run_eval.evaluate_model_for_task) into eval_settings["apply_chat_template"];
# consumed by BaseEvaluator._wants_chat_template (UNION with the per-task
# LM_EVAL_CHAT_TEMPLATE_TASKS). NO default exists: a registry without the field
# leaves every command byte-identical.

_apply_chat_template_cache: Optional[tuple] = None  # (path, by_label, by_task)


def _coerce_registry_bool(value):
    """Registry field -> True/False/None. Accepts JSON bool, 0/1, and the usual
    truthy/falsy strings; anything else (or absent) -> None so callers change
    nothing."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "yes", "on"):
            return True
        if v in ("0", "false", "no", "off"):
            return False
    return None


def _load_apply_chat_template_tables(registry_path: Optional[str] = None):
    """(by_label, by_task) tables of the optional "apply_chat_template" field,
    cached per registry file (same file / MICR_MODEL_REGISTRY override as
    registry_eval_limit). by_label maps label -> (value, entry_task); by_task
    keeps the last non-None value in file order, mirroring registry_eval_limit's
    last-wins semantics. Any problem -> empty tables."""
    global _apply_chat_template_cache
    path = registry_path or os.environ.get("MICR_MODEL_REGISTRY") or _MODEL_REGISTRY_PATH
    if _apply_chat_template_cache is None or _apply_chat_template_cache[0] != path:
        by_label, by_task = {}, {}
        try:
            with open(path) as fh:
                registry = json.load(fh)
            for label, entry in registry.items():
                if not isinstance(entry, dict):
                    continue
                # by_label holds EVERY model entry, its value None when the field
                # is absent/uninterpretable. That is deliberate: a KNOWN model
                # without the field must resolve to None (raw) and must NOT
                # inherit a task-sharing sibling's flag (MedGo, no field, must
                # stay raw even though UltraMedical templates the same task).
                val = (_coerce_registry_bool(entry.get("apply_chat_template"))
                       if "apply_chat_template" in entry else None)
                by_label[str(label)] = (val, entry.get("task"))
                task = entry.get("task")
                if val is not None and task:  # last non-None wins (task fallback)
                    by_task[str(task)] = val
        except Exception:
            by_label, by_task = {}, {}
        _apply_chat_template_cache = (path, by_label, by_task)
    return _apply_chat_template_cache[1], _apply_chat_template_cache[2]


def registry_apply_chat_template(registry_task_name: Optional[str] = None,
                                 registry_label: Optional[str] = None,
                                 registry_path: Optional[str] = None):
    """
    The registry's optional per-model chat-template flag, or None.

    Mirrors registry_eval_limit's resolution -- same file, same
    MICR_MODEL_REGISTRY override, same last-wins-by-task fallback -- but reads
    the boolean entry field "apply_chat_template" instead of "eval_limit".

    Resolution order (per-MODEL first, so two models that share a task -- e.g.
    Llama-3.1-8B-UltraMedical and MedGo, both on medqa_4options -- can disagree):
      1. registry_label: if it is a KNOWN registry entry AND (registry_task_name
         is None or equals that entry's own "task"), return that entry's value
         -- INCLUDING None when the entry has no field. A known model without
         the field is raw and must NOT inherit a task-sharing sibling's flag
         (this is exactly the MedGo-vs-UltraMedical case). A label whose own
         task does not match the task being evaluated does not apply and falls
         through to task resolution.
      2. registry_task_name: the entry whose "task" == this name that sets the
         field wins (last in file order), exactly like registry_eval_limit. Used
         when no label is known (ad-hoc target, path not in the registry).

    Returns True / False / None. Any problem (missing/unreadable registry,
    non-boolean value, no relevant entry) returns None so callers change nothing
    -- the gate-off default that keeps every command byte-identical.
    """
    by_label, by_task = _load_apply_chat_template_tables(registry_path)
    if registry_label is not None and str(registry_label) in by_label:
        val, label_task = by_label[str(registry_label)]
        if (registry_task_name is None or label_task is None
                or str(label_task) == str(registry_task_name)):
            return val
    if registry_task_name is not None:
        return by_task.get(str(registry_task_name))
    return None


def registry_label_for_path(model_path, registry_path: Optional[str] = None):
    """The registry label whose "local_path" resolves to ``model_path``, or None.

    Lets a runner that only holds a model *path* (the profiler's --model)
    recover the per-model registry key, so registry_apply_chat_template can
    disambiguate models that share a task. Matches on normalized absolute paths;
    any problem returns None (falls back to task resolution)."""
    path = registry_path or os.environ.get("MICR_MODEL_REGISTRY") or _MODEL_REGISTRY_PATH
    try:
        target = os.path.realpath(str(model_path))
        with open(path) as fh:
            registry = json.load(fh)
        for label, entry in registry.items():
            if not isinstance(entry, dict):
                continue
            lp = entry.get("local_path")
            if lp and os.path.realpath(str(lp)) == target:
                return str(label)
    except Exception:
        return None
    return None


# --------------------------------------------------------------------- score parsing
# Ported verbatim from unified-llm-eval/utils/result_parser.py. Do not "clean up":
# these regexes define how a subprocess's stdout becomes a number.

def parse_score(stdout: str, task_name: str) -> Optional[float]:
    if task_name == "humaneval":
        match = re.search(r"\|\s*humaneval\s*\|.*?\|\s*pass@1\s*\|.*?(\d+\.?\d*)", stdout)
        if match:
            return float(match.group(1)) * 100
        match = re.search(r"'pass@1':\s*(\d+\.?\d+)", stdout)
        if match:
            return float(match.group(1)) * 100

    elif task_name in ["gsm8k-cot", "gsm8k-pal"]:
        match = re.search(r"'acc':\s*([\d\.]+)", stdout)
        if match:
            return float(match.group(1))          # already a percentage
        for line in reversed(stdout.strip().split("\n")):
            if re.match(r"^\d+\.\d+(\s+\d+\.\d+)?$", line.strip()):
                return float(line.strip().split()[0])

    elif task_name.startswith("ifeval"):
        # New branch (NOT part of the verbatim port; added with the persistent-
        # eval wiring): ifeval's lm_eval table has no plain 'acc' metric row --
        # its metrics are {prompt,inst}_level_{strict,loose}_acc -- so every
        # generic pattern below misses and the task always ended in
        # PARSE_ERROR. Score = prompt_level_strict_acc, the first metric in
        # ifeval's metric_list. Keep in sync with
        # micr/persistent_eval._run_lm_eval, which extracts the same key so
        # the persistent and subprocess paths report the same number.
        # Variant-name/filter safe: the pattern keys on the METRIC cell only,
        # so rows for ifeval_nothink / ifeval_P / ifeval_M (think-strip
        # variants; Filter column 'strip_think' instead of 'none') match too.
        m = re.search(r"\|\s*prompt_level_strict_acc\s*\|[^|]*\|\s*([0-9.]+)\s*\|", stdout)
        if m:
            return float(m.group(1)) * 100
        m = re.search(r"['\"]prompt_level_strict_acc[^'\"]*['\"]:\s*([0-9.]+)", stdout)
        if m:
            return float(m.group(1)) * 100

    if task_name.startswith("tiny"):
        for pat, scale in ((r'\|\s*' + re.escape(task_name) + r'\s*\|.*?\|\s*acc_norm\s*\|.*?\|([0-9.]+)\|', 100),
                           (r"'acc_norm':\s*([0-9.]+)", 100),
                           (r'"acc_norm[^"]*":\s*([0-9.]+)', 100)):
            m = re.search(pat, stdout)
            if m:
                return float(m.group(1)) * scale

    for pat in (r'\|\s*[\w_]+\s*\|.*?\|\s*acc\s*\|.*?\|([0-9.]+)\|',
                r"'acc':\s*([0-9.]+)",
                r'"acc[^"]*":\s*([0-9.]+)',
                r'\|\s*acc\s*\|[^|]*\|([0-9.]+)\s*\|'):
        m = re.search(pat, stdout)
        if m:
            return float(m.group(1)) * 100

    m = re.search(r"Score is\s*(\d+\.?\d+)", stdout)
    if m:
        return float(m.group(1)) * 100
    return None


# --------------------------------------------------------------------- evaluators

# ---- ifeval / Light-IF-32B eval config (decided 2026-07) --------------------
# Raw prompts make the model CONTINUE the prompt instead of answering
# (~21% prompt_level_strict = noise); under --apply_chat_template it answers,
# but every response opens with a <think>...</think> block whose prose
# violates ifeval's format checks (18%). So ifeval is ALWAYS evaluated through
# LOCAL task variants (micr/eval_split_configs, generated by
# micr/eval_splits.py) that strip a leading think block before scoring, with
# the chat template applied:
#   full set -> ifeval_nothink ; split P/M -> ifeval_P / ifeval_M (the
#   generated split variants carry the same filter).
# Keep these two maps in sync with eval_splits.FULLSET_VARIANTS/_TASK_EXTRA_YAML.
LM_EVAL_FULLSET_TASK_OVERRIDES = {"ifeval": "ifeval_nothink",
                                  "medqa_4options": "medqa_4options_fs"}
LM_EVAL_CHAT_TEMPLATE_TASKS = {"ifeval"}


class BaseEvaluator:
    """Same interface as unified-llm-eval's BaseEvaluator: _construct_command + evaluate."""

    def __init__(self, env_manager=None, env_config=None, eval_settings=None):
        self.env_manager = env_manager          # unused; kept for signature compatibility
        self.env_config = env_config or {}
        self.eval_settings = eval_settings or {}

    def _construct_command(self, model_path, task_name):
        raise NotImplementedError

    # ------------------------------------------------- chat template decision
    def _wants_chat_template(self, task_name) -> bool:
        """Should this eval pass lm_eval's --apply_chat_template?

        UNION of two independent signals, so a per-model flag ADDS to (never
        removes) the per-task forcing:
          * per-TASK  : task_name in LM_EVAL_CHAT_TEMPLATE_TASKS -- ifeval is
                        always templated, for EVERY model;
          * per-MODEL : eval_settings["apply_chat_template"] truthy -- set by the
                        runners from the registry's per-entry flag
                        (eval_harness.registry_apply_chat_template).
        With neither signal this is False and the command is byte-identical to
        before this feature existed. An explicit per-model False cannot turn
        ifeval off (the task force wins), matching the union contract."""
        if task_name in LM_EVAL_CHAT_TEMPLATE_TASKS:
            return True
        return bool(self.eval_settings.get("apply_chat_template"))

    # ---------------------------------------------------------- eval splits
    # Opt-in via eval_settings["eval_split"] in {"P","M"}; see micr/eval_splits.py.
    # With the key absent/None/"full" these helpers change nothing and every
    # command stays byte-identical to the pre-split code.

    def _eval_split(self):
        raw = self.eval_settings.get("eval_split")
        if raw is None:
            return None
        v = str(raw).strip()
        if v.lower() in ("", "none", "full"):
            return None
        if v.upper() in ("P", "M"):
            return v.upper()
        raise ValueError(f"eval_settings['eval_split'] must be P, M, full or None; got {raw!r}")

    def _split_include_dir(self):
        d = self.eval_settings.get("eval_split_config_dir")
        if d:
            return str(d)
        # Keep in sync with eval_splits.default_lm_eval_config_dir(); inlined so
        # this module keeps working when loaded outside the merge_tools package.
        d = os.environ.get("MICR_EVAL_SPLIT_CONFIG_DIR")
        if d:
            return d
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_split_configs")

    def _parse_score_split_fallback(self, output, task_name):
        """
        Split runs rename lm_eval tasks (task -> task_P / task_M), so the
        stdout table row carries the variant name. Only reached when the
        verbatim parse above returned None AND a variant name was recorded;
        never on a default (full-set) run.
        """
        effective = getattr(self, "_parse_task_name", None)
        if not effective or effective == task_name:
            return None
        parsed = parse_score(output, effective)
        if parsed is not None:
            return parsed
        # humaneval's pass@1 table pattern hardcodes the task name; retry it
        # with the variant name.
        m = re.search(r"\|\s*" + re.escape(effective) + r"\s*\|.*?\|\s*pass@1\s*\|.*?(\d+\.?\d*)", output)
        if m:
            return float(m.group(1)) * 100
        return None

    def evaluate(self, model_config, task_name, run_id=1) -> dict:
        start = time.time()
        command, cwd, env_name = self._construct_command(model_config["path"], task_name)
        # Run in *this* interpreter's environment. No conda, no env name.
        argv = [sys.executable if c == "python" else c for c in command]
        timeout = self.eval_settings.get("timeout_minutes", 30) * 60

        try:
            result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            result = subprocess.CompletedProcess(argv, -1, "", f"Command timed out after {timeout} seconds")
            timed_out = True
        except Exception as e:
            result = subprocess.CompletedProcess(argv, -1, "", f"Error running command: {e}")
            timed_out = False

        status, score, error_log = "FAILED", "N/A", ""
        if timed_out:
            status, error_log = "TIMEOUT", f"Task timed out after {timeout / 60} minutes."
        elif result.returncode == 0:
            combined_output = result.stdout + "\n" + result.stderr
            parsed = parse_score(combined_output, task_name)
            if parsed is None:
                parsed = self._parse_score_split_fallback(combined_output, task_name)
            if parsed is not None:
                status, score = "SUCCESS", f"{parsed:.2f}%"
            else:
                status, error_log = "PARSE_ERROR", "Could not parse score from output."
        else:
            error_log = result.stderr or result.stdout

        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_config["model_name"],
            "task": task_name,
            "run_id": run_id,
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "current"),
            "score": score,
            "status": status,
            "duration": f"{(time.time() - start) / 60:.2f}min",
            "error_log": str(error_log).strip()[-2000:],
        }


class SimpleEvaluator(BaseEvaluator):
    """lm_eval tasks: mmlu_*, sst2, truthfulqa_mc2, ifeval, medqa, ..."""

    def _construct_command(self, model_path, task_name):
        gpu_ids = self.eval_settings.get("gpu_ids", "0")
        tp_size = len(gpu_ids.split(","))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

        batch_size = self.eval_settings.get("batch_size", "auto")
        temperature = self.eval_settings.get("temperature", 0.0)
        use_vllm = self.eval_settings.get("use_vllm", False)

        if use_vllm:
            backend = "vllm"
            # 0.8, not 0.9: eval engines must coexist with the caller's
            # GPU-resident model (profiler/MICR keep the working model loaded;
            # a 7-8B parent leaves ~124 GiB free on an H200, below 0.9's
            # 125.8 GiB demand -- observed refusing to boot). 0.8 = 111.9 GiB
            # fits with headroom; KV capacity stays ample for these tasks.
            # Applies to persistent sessions too (they parse this same string).
            # Overridable per task via eval_settings["vllm_gpu_memory_utilization"]
            # (Light-IF-32B single-GPU co-residency needs 0.55).
            gmu = self.eval_settings.get("vllm_gpu_memory_utilization", 0.8)
            model_args = (f"pretrained={model_path},dtype=bfloat16,"
                          f"tensor_parallel_size={tp_size},gpu_memory_utilization={gmu}")
            device_arg = []
        else:
            backend = "hf"
            model_args = f"pretrained={model_path},dtype=bfloat16"
            if tp_size > 1:
                model_args += ",parallelize=True"
                device_arg = []
            else:
                device_arg = ["--device", "cuda"]

        # Eval split (opt-in): run the generated <task>_<SPLIT> variant from the
        # split configs dir. Default: task_arg == task_name, no extra args --
        # EXCEPT tasks with a full-set override (ifeval -> ifeval_nothink),
        # whose local variant also needs the --include_path.
        split = self._eval_split()
        if split is None:
            task_arg = LM_EVAL_FULLSET_TASK_OVERRIDES.get(task_name, task_name)
            needs_include_path = task_arg != task_name
        else:
            task_arg = f"{task_name}_{split}"
            needs_include_path = True
        self._parse_task_name = task_arg

        gen_kwargs_str = json.dumps({"temperature": temperature,
                                     "do_sample": False if temperature == 0.0 else True})
        command = [
            "python", "-m", "lm_eval",
            "--model", backend,
            "--tasks", task_arg,
            "--model_args", model_args,
            "--gen_kwargs", gen_kwargs_str,
            "--batch_size", str(batch_size),
            "--output_path", f"./evaluation_results/{task_arg}_output.json",
        ]
        limit = self.eval_settings.get("limit")
        if limit is not None:
            command.extend(["--limit", str(limit)])
        if needs_include_path:
            command.extend(["--include_path", self._split_include_dir()])
        if self._wants_chat_template(task_name):
            # Presence-only flag (CLI nargs='?' const=True); placed before
            # device_arg so any following token starts with '-' and cannot be
            # swallowed as the optional template-name argument. Fires for
            # LM_EVAL_CHAT_TEMPLATE_TASKS (ifeval, every model) OR a per-model
            # eval_settings["apply_chat_template"] (registry-driven). Both the hf
            # and vllm backends build through this one method, so both get the
            # flag; the persistent (vllm) and in-process (hf) fast paths forward
            # it because it is present in this argv.
            command.append("--apply_chat_template")
        command.extend(device_arg)
        return command, None, None


class HarnessEvaluator(BaseEvaluator):
    """humaneval (lm_eval) and gsm8k-cot/pal (vendored math-evaluation-harness)."""

    def _construct_command(self, model_path, task_name):
        gpu_ids = self.eval_settings.get("gpu_ids", "0")
        tp_size = len(gpu_ids.split(","))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

        if task_name == "humaneval":
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"
            # Eval split (opt-in): humaneval runs through lm_eval, so it uses
            # the same generated-variant mechanism as SimpleEvaluator.
            split = self._eval_split()
            task_arg = "humaneval" if split is None else f"humaneval_{split}"
            self._parse_task_name = task_arg
            batch_size = self.eval_settings.get("batch_size", "auto")
            temperature = self.eval_settings.get("temperature", 0.0)
            # gpu_memory_utilization=0.8: same coexistence constraint as the
            # other vllm builder above -- the default 0.9 cannot boot next to
            # the caller's GPU-resident model. Same per-task override key as
            # SimpleEvaluator (vllm_gpu_memory_utilization, default 0.8).
            gmu = self.eval_settings.get("vllm_gpu_memory_utilization", 0.8)
            model_args = (f"pretrained={model_path},tensor_parallel_size={tp_size},"
                          f"dtype=bfloat16,gpu_memory_utilization={gmu}")
            gen_kwargs_str = json.dumps({"temperature": temperature,
                                         "do_sample": False if temperature == 0.0 else True})
            command = [
                "python", "-m", "lm_eval",
                "--model", "vllm",
                "--tasks", task_arg,
                "--model_args", model_args,
                "--gen_kwargs", gen_kwargs_str,
                "--device", "cuda",
                "--batch_size", str(batch_size),
                "--output_path", f"./evaluation_results/{task_arg}_output.json",
                "--confirm_run_unsafe_code",
            ]
            if self._wants_chat_template("humaneval"):
                # Presence-only flag; humaneval is not in
                # LM_EVAL_CHAT_TEMPLATE_TASKS, so this only fires for a per-model
                # eval_settings["apply_chat_template"] (e.g. an instruct coder
                # that needs its chat template). Any following token
                # (--include_path) starts with '-'. Gate-off: no flag -> not
                # appended -> byte-identical.
                command.append("--apply_chat_template")
            if split is not None:
                command.extend(["--include_path", self._split_include_dir()])
            return command, None, None

        if task_name in ("gsm8k-cot", "gsm8k-pal"):
            self._parse_task_name = task_name  # math splits keep the task name
            math_dir = math_harness_dir(self.env_config)
            prompt_type = "cot" if task_name == "gsm8k-cot" else "pal"
            temperature = self.eval_settings.get("temperature", 0.0)
            batch_size = self.eval_settings.get("batch_size", 32)
            script = os.path.abspath(os.path.join(math_dir, "math_eval.py"))
            if not os.path.exists(script):
                raise FileNotFoundError(
                    f"math_eval.py not found at {script}. Set MICR_MATH_HARNESS_DIR."
                )
            command = [
                "python", script,
                "--model_name_or_path", model_path,
                "--data_names", "gsm8k",
                "--prompt_type", prompt_type,
                "--use_vllm",
                "--temperature", str(temperature),
                "--save_outputs",
                "--overwrite",
                "--batch_size", str(batch_size),
                "--use_safetensors",
            ]
            nts = self.eval_settings.get("num_test_sample")
            if nts:
                command += ["--num_test_sample", str(nts),
                            "--seed", str(self.eval_settings.get("seed", 0))]
            # Eval split (opt-in): the harness reads data/gsm8k/<split>.jsonl,
            # materialized by micr/eval_splits.py. Fail fast if it is missing.
            split = self._eval_split()
            if split is not None:
                split_file = os.path.join(math_dir, "data", "gsm8k", f"test{split}.jsonl")
                if not os.path.exists(split_file):
                    raise FileNotFoundError(
                        f"eval split file not found: {split_file}. Materialize it first: "
                        f"python merge_tools/micr/eval_splits.py --task gsm8k-cot --splits {split}"
                    )
                command += ["--split", f"test{split}"]
            return command, math_dir, None

        raise ValueError(f"HarnessEvaluator does not support task: {task_name}")


# --------------------------------------------------------------------- registry

TASK_REGISTRY = {
    # HarnessEvaluator
    "gsm8k-cot": {"evaluator": HarnessEvaluator, "env": None},
    "gsm8k-pal": {"evaluator": HarnessEvaluator, "env": None},
    "humaneval": {"evaluator": HarnessEvaluator, "env": None},
    # SimpleEvaluator -- the five MICR domains
    "mmlu_econometrics": {"evaluator": SimpleEvaluator, "env": None},
    "mmlu_professional_law": {"evaluator": SimpleEvaluator, "env": None},
    "mmlu_professional_medicine": {"evaluator": SimpleEvaluator, "env": None},
    "truthfulqa_mc2": {"evaluator": SimpleEvaluator, "env": None},
    "sst2": {"evaluator": SimpleEvaluator, "env": None},
    # SimpleEvaluator -- the 32B / extra domains added in the working copy
    "ifeval": {"evaluator": SimpleEvaluator, "env": None},
    "medqa": {"evaluator": SimpleEvaluator, "env": None},
    "medqa_4options": {"evaluator": SimpleEvaluator, "env": None},
    "m_mmlu_ru": {"evaluator": SimpleEvaluator, "env": None},
    "mmlu_ru": {"evaluator": SimpleEvaluator, "env": None},
    "mmlu_international_law": {"evaluator": SimpleEvaluator, "env": None},
    "careqa": {"evaluator": SimpleEvaluator, "env": None},
    "sciq": {"evaluator": SimpleEvaluator, "env": None},
    "pubmedqa": {"evaluator": SimpleEvaluator, "env": None},
    "financial_tweets": {"evaluator": SimpleEvaluator, "env": None},
    "tinyMMLU": {"evaluator": SimpleEvaluator, "env": None},
}


class EnvironmentManager:
    """No-op stand-in. Kept so the runners' call sites do not change."""

    def __init__(self, harness_env=None, languages_env=None, *_a, **_k):
        self.harness_env = harness_env or os.environ.get("CONDA_DEFAULT_ENV", "current")
        self.languages_env = languages_env or self.harness_env
