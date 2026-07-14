"""
In-process HF evaluator for MC/loglikelihood tasks.

Every profiler cell (and every MICR step on the hf backend) spawns a fresh
``lm_eval --model hf`` subprocess that re-loads the candidate checkpoint from
disk -- ~60-80 s for a 62 GB 32B model -- and then scores it with whatever
batch fits BESIDE the caller's GPU-resident copy (~15 GiB headroom on an H200
when the parent holds a 32B). For multiple-choice / loglikelihood tasks none
of that is necessary: the caller already holds the exact candidate weights in
memory, and lm_eval's Python API can score an ALREADY-LOADED model object:

    lm = lm_eval.models.huggingface.HFLM(pretrained=<PreTrainedModel>,
                                         tokenizer=<dir or None>, ...)
    lm_eval.simple_evaluate(model=lm, tasks=[task], task_manager=...)

(verified against the installed lm_eval 0.4.9.2: ``HFLM.__init__`` accepts a
``transformers.PreTrainedModel`` for ``pretrained`` and takes device/config
from the object; ``simple_evaluate`` accepts any ``lm_eval.api.model.LM``
instance plus a ``TaskManager(include_path=...)`` for the generated
``<task>_P``/``<task>_M`` split variants -- the same pattern
``persistent_eval._run_lm_eval`` already uses for the vLLM backend.)

Metric parity: the argv is taken from the evaluator's own
``_construct_command`` output -- task variant, --include_path, --limit,
--batch_size, --gen_kwargs included -- so *what* is evaluated is never
reimplemented here; only *where the model lives* changes. Scores are rounded
to the same 2-decimal grid the subprocess parser reports.

CONSISTENCY CAVEAT (documented, by design): in-process scores are NOT
guaranteed bit-identical to subprocess scores. The subprocess auto-detects its
batch size under different free GPU memory (it must fit a SECOND copy of the
model next to the caller's), and batch composition/padding can flip
near-tied loglikelihood comparisons. Within a run the fast path is
self-consistent: the gates depend only on (env flag, task, backend), so the
baseline and every candidate take the same path, and any (baseline - score)
drop is computed within one engine. Callers enforce the baseline side of that
(see gaussian_profiler's baseline block and run_eval's evaluate_model_for_task).

Any unsupported case, any exception, any doubt -> return None, and the caller
runs the legacy subprocess path unchanged.

Environment:
    MICR_INPROCESS_HF_EVAL     enable/disable the in-process fast path.
                               DEFAULT ON ("1") -- fastest-eval-path-by-default
                               policy. Set MICR_INPROCESS_HF_EVAL=0 to restore
                               the legacy behavior where every eval subprocesses
                               exactly as before this module existed (the
                               strict-bit-reproduction escape hatch). The score
                               is bit-identical either way (deterministic
                               per-example loglikelihood); the flag only changes
                               WHERE the model is scored, never WHAT.
"""
import gc
import json
import os
from typing import Optional

# Registry tasks that are MC/loglikelihood end-to-end on the lm_eval harness
# (SimpleEvaluator tasks minus the generative ifeval). Only these are served;
# everything else falls back to the subprocess path untouched. Generative
# tasks would *run* correctly through simple_evaluate too, but they are where
# vLLM (persistent_eval) wins -- this module is deliberately scoped to the
# loglikelihood tasks whose subprocesses spend their time re-loading weights.
MC_HF_TASKS = frozenset({
    "mmlu_econometrics",
    "mmlu_professional_law",
    "truthfulqa_mc2",
    "sst2",
    "medqa",
    "medqa_4options",
    "m_mmlu_ru",
    "mmlu_ru",
    "mmlu_international_law",
    "careqa",
    "sciq",
    "pubmedqa",
    "financial_tweets",
    "tinyMMLU",
})

_disabled_reason: Optional[str] = None

# One TaskManager per --include_path per process (same caching rationale as
# persistent_eval._task_managers: its __init__ walks every installed task YAML,
# ~2s -- rebuilding it per cell would eat a chunk of the win).
_task_managers: dict = {}


def enabled() -> bool:
    """In-process fast path toggle. DEFAULT ON ("1"): scored bit-identically to
    the subprocess path (deterministic per-example loglikelihood), so it is a
    pure-win default. Set MICR_INPROCESS_HF_EVAL=0 to force the legacy
    subprocess path byte-for-byte (strict-reproduction escape hatch)."""
    if _disabled_reason:
        return False
    return os.environ.get("MICR_INPROCESS_HF_EVAL", "1").strip().lower() in (
        "1", "true", "yes", "on",
    )


def disable(reason: str) -> None:
    """Permanently disable the fast path for this process; every subsequent
    eval takes the subprocess path. Used by callers to enforce baseline/cell
    path-consistency, and internally after an unexpected failure."""
    global _disabled_reason
    _disabled_reason = reason
    print(f"  [inprocess-eval] disabled for this run: {reason}")


def eligible(task_key: str, eval_settings: Optional[dict]) -> bool:
    """Cheap pre-gate (no command construction): flag ON, MC/loglikelihood
    registry task, hf backend (SimpleEvaluator builds ``--model hf`` iff
    ``use_vllm`` is falsy). Callers use this to decide whether a candidate
    save can be skipped; :func:`evaluate` re-checks the real argv."""
    if not enabled():
        return False
    if task_key not in MC_HF_TASKS:
        return False
    if isinstance(eval_settings, dict) and eval_settings.get("use_vllm", False):
        return False
    return True


def _flag(command, name, default=None):
    if name in command:
        return command[command.index(name) + 1]
    return default


def _parse_gen_kwargs(value: str):
    """Mirror lm_eval's CLI --gen_kwargs parsing (type=try_parse_json); see
    persistent_eval._parse_gen_kwargs for the failure mode this avoids."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        if "{" in value:
            raise ValueError(f"--gen_kwargs is malformed JSON: {value!r}")
        return value


def _extract_score(results: dict, task: str, task_key: str) -> Optional[float]:
    """results[task] -> percentage on the subprocess parser's 2-decimal grid.

    Key order mirrors eval_harness.parse_score: tiny* tasks report acc_norm
    (tinyBenchmarks' IRT-weighted metric), everything else the plain acc row
    the generic table regex would have matched first."""
    metrics = results.get("results", {}).get(task)
    if not metrics:
        return None
    order = ("acc_norm,none", "acc,none") if task_key.startswith("tiny") \
        else ("acc,none", "acc_norm,none")
    for key in order:
        if key in metrics:
            return round(float(metrics[key]) * 100.0, 2)
    for k, v in metrics.items():  # single-metric tasks
        if not k.endswith("_stderr") and isinstance(v, (int, float)):
            return round(float(v) * 100.0, 2)
    return None


def evaluate(evaluator_cls, env_manager, env_config, eval_settings,
             model, model_path: str, task_key: str) -> Optional[float]:
    """
    Score ``model`` on ``task_key`` in this process via lm_eval's Python API.

    model:      an ALREADY-LOADED ``transformers.PreTrainedModel`` (profiler:
                the resident, possibly perturbed model -- scored as-is, never
                mutated, never freed), or a checkpoint dir path ``str`` (MICR:
                the saved candidate; loaded here with the CLI's own
                ``create_from_arg_string`` and freed afterwards).
    model_path: checkpoint dir used for ``_construct_command`` and, for the
                resident-model case, the tokenizer (the same assets the
                subprocess path would have copied beside the candidate).

    Returns None whenever the caller should fall back to the subprocess path.
    NOTE for resident-model callers: on None the candidate was NEVER saved to
    disk -- the fallback must save it before subprocess-evaluating (falling
    through to a subprocess on ``model_path`` would score the SOURCE weights).
    """
    if not enabled():
        return None
    if task_key not in MC_HF_TASKS:
        return None

    try:
        command, _cwd, _env = evaluator_cls(
            env_manager, env_config, dict(eval_settings)
        )._construct_command(str(model_path), task_key)
    except Exception as e:
        print(f"  [inprocess-eval] cannot construct command "
              f"({type(e).__name__}); using subprocess")
        return None
    if list(command[:3]) != ["python", "-m", "lm_eval"]:
        return None
    if _flag(command, "--model") != "hf":
        return None                       # vLLM-backed: not this module's job

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    task = _flag(command, "--tasks")           # split variant already resolved
    batch_size = _flag(command, "--batch_size", "auto")
    own_model = isinstance(model, str)

    kwargs = {}
    gen_kwargs = _flag(command, "--gen_kwargs")
    if gen_kwargs:
        kwargs["gen_kwargs"] = _parse_gen_kwargs(gen_kwargs)
    if "--apply_chat_template" in command:
        # THE GAP this fix closes: the evaluator's argv now carries a
        # presence-only --apply_chat_template whenever the task
        # (LM_EVAL_CHAT_TEMPLATE_TASKS) or the per-model registry flag
        # (eval_settings["apply_chat_template"]) asks for it. Without this line
        # the in-process fast path -- the DEFAULT for MC/loglikelihood hf tasks
        # -- would silently drop the template and feed a chat-tuned model raw
        # prompts (the Llama-3.1-8B-UltraMedical/medqa_4options 27.83 bug).
        # simple_evaluate takes the same-named kwarg (verified on the installed
        # lm_eval 0.4.9.2: simple_evaluate(..., apply_chat_template=...) forwards
        # to evaluate() and lm.apply_chat_template); True == the CLI's
        # presence-only const, so in-process and subprocess prompt identically.
        kwargs["apply_chat_template"] = True
    limit = _flag(command, "--limit")
    if limit is not None:
        kwargs["limit"] = float(limit)         # CLI parses --limit with type=float
    include_path = _flag(command, "--include_path")
    tm_key = include_path or ""
    if tm_key not in _task_managers:
        from lm_eval.tasks import TaskManager
        _task_managers[tm_key] = TaskManager(include_path=include_path)
    kwargs["task_manager"] = _task_managers[tm_key]

    lm = None
    try:
        try:
            if own_model:
                # The CLI's own construction, byte-for-byte (lm_eval/evaluator.py
                # simple_evaluate's str-model branch): model_args string in,
                # batch_size/max_batch_size/device as additional_config.
                from lm_eval.api.registry import get_model
                lm = get_model("hf").create_from_arg_string(
                    _flag(command, "--model_args", ""),
                    {
                        "batch_size": batch_size,
                        "max_batch_size": None,
                        "device": _flag(command, "--device"),
                    },
                )
            else:
                # Wrap the caller's resident model object. HFLM takes device
                # and config from the object; the tokenizer comes from
                # model_path -- the same files the subprocess candidate dirs
                # carry (copied from this very directory). model_args from the
                # argv (pretrained=..., dtype, parallelize) are irrelevant
                # here: the weights are already loaded and placed.
                tok = str(model_path) if os.path.isdir(str(model_path)) else None
                lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch_size)
            res = lm_eval.simple_evaluate(
                model=lm, tasks=[task], batch_size=batch_size, **kwargs
            )
        finally:
            # Free what WE created; never touch a caller's resident model.
            if lm is not None:
                if own_model:
                    try:
                        lm._model = None
                    except Exception:
                        pass
                del lm
            # Release cached allocator blocks UNCONDITIONALLY (resident case
            # included). Batched scoring leaves tens of GiB of activation
            # blocks cached in the parent's CUDA allocator (measured: a 61.4
            # GiB resident 32B grew to 101 GiB after one in-process eval),
            # and any later subprocess eval -- a fallback cell, a vLLM task's
            # engine -- must fit a second model copy beside this process.
            # empty_cache() frees only unused cached blocks, never the
            # resident weights.
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    except Exception as e:
        # Failures here are systematic (task YAML, dataset cache, OOM), and a
        # silent per-cell subprocess fallback would drift away from an
        # in-process baseline. Disable the fast path loudly; the caller's
        # fallback still scores THIS candidate via the subprocess path.
        disable(f"{type(e).__name__}: {e}")
        return None

    score = _extract_score(res, task, task_key)
    if score is None:
        print(f"  [inprocess-eval] no recognizable metric for '{task}'; "
              f"using subprocess")
    return score
