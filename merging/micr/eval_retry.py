"""
Shared evaluation-retry policy for the MICR runners.

Each runner's ``evaluate_model_for_task`` returns ``None`` when no score was
produced (harness unavailable, unknown task, evaluator crash or timeout) and a
float otherwise, so a returned ``0.0`` always means the model genuinely scored
zero.

Keeping those two apart is not cosmetic. A failed evaluation reported as 0.0
becomes a fabricated catastrophic score: as a step it forces a spurious
``rejected``, and as a baseline it pins ``last_score`` at 0.0, which turns the
reject test into ``score < -drop_tolerance`` and accepts every subsequent step
unconditionally.

Failures are usually transient on a shared GPU (vLLM racing for free memory at
engine start, harness timeouts), so retry with backoff before giving up.

Environment:
    MICR_EVAL_RETRIES        extra attempts for a step/final eval  (default 2)
    MICR_EVAL_RETRY_BACKOFF  first backoff in seconds, doubling    (default 30)
    MICR_BASELINE_MAX_WAIT   how long a baseline may keep retrying (default 3600)
"""
import os
import time
from typing import Callable, Optional

_MAX_BACKOFF_S = 300.0


def eval_retries() -> int:
    return int(os.environ.get("MICR_EVAL_RETRIES", "2"))


def eval_retry_backoff_s() -> float:
    return float(os.environ.get("MICR_EVAL_RETRY_BACKOFF", "30"))


def baseline_max_wait_s() -> float:
    return float(os.environ.get("MICR_BASELINE_MAX_WAIT", "3600"))


def fallback_is_redundant(evaluator_cls, env_manager, env_config, eval_settings,
                          model_path: str, task_key: str) -> bool:
    """
    True when the ``use_vllm=False`` retry would run a BYTE-IDENTICAL command.

    HarnessEvaluator hardcodes ``--use_vllm`` (gsm8k) and ``--model vllm`` (humaneval)
    in its argv and never reads ``eval_settings["use_vllm"]``, so its "fallback" re-runs
    exactly the same subprocess. SimpleEvaluator does read it and genuinely switches to
    the HF backend, so its fallback is real.

    If the commands cannot be constructed we return False -- i.e. keep today's behavior.
    """
    try:
        a = evaluator_cls(env_manager, env_config, dict(eval_settings))._construct_command(model_path, task_key)
        retry = dict(eval_settings)
        retry["use_vllm"] = False
        b = evaluator_cls(env_manager, env_config, retry)._construct_command(model_path, task_key)
        return a == b
    except Exception:
        return False


def evaluate_with_retry(
    eval_fn: Callable[..., Optional[float]], label: str, *args, **kwargs
) -> Optional[float]:
    """
    Call ``eval_fn(*args, **kwargs)``, retrying while it produces no score.

    Returns the score, or None when every attempt failed. The caller must not
    turn that None into a number: the step was never judged.
    """
    retries = eval_retries()
    backoff = eval_retry_backoff_s()
    for attempt in range(retries + 1):
        # Retrying vLLM is cheap; re-running the HF fallback every attempt is not.
        # Let the last attempt (only) fall back to the other backend.
        score = eval_fn(*args, allow_fallback=(attempt == retries), **kwargs)
        if score is not None:
            return score
        if attempt < retries:
            wait = backoff * (2 ** attempt)
            print(
                f"  [eval-retry] {label}: attempt {attempt + 1}/{retries + 1} "
                f"produced no score; retrying in {wait:.0f}s"
            )
            time.sleep(wait)
    return None


def evaluate_baseline_blocking(
    eval_fn: Callable[..., Optional[float]], label: str, *args, **kwargs
) -> float:
    """
    Call ``eval_fn(*args, **kwargs)`` until it produces a score, or raise.

    The baseline must be a real measurement, so this never returns None and never
    falls back to 0.0. It backs off exponentially (capped) and gives up only once
    MICR_BASELINE_MAX_WAIT has elapsed.
    """
    max_wait = baseline_max_wait_s()
    backoff = eval_retry_backoff_s()
    started = time.monotonic()
    attempt = 0
    while True:
        # The baseline blocks until it gets a score, so always allow the real fallback.
        score = eval_fn(*args, allow_fallback=True, **kwargs)
        if score is not None:
            return score
        elapsed = time.monotonic() - started
        if elapsed >= max_wait:
            raise RuntimeError(
                f"Baseline evaluation for {label} produced no score after "
                f"{elapsed / 60:.1f} min of retries. Refusing to continue: a 0.0 "
                f"baseline would make every subsequent step accept unconditionally."
            )
        wait = min(backoff * (2 ** attempt), _MAX_BACKOFF_S)
        wait = min(wait, max(1.0, max_wait - elapsed))
        print(
            f"  [baseline-retry] {label}: no score; retrying in {wait:.0f}s "
            f"(elapsed {elapsed / 60:.1f}m of {max_wait / 60:.0f}m budget)"
        )
        time.sleep(wait)
        attempt += 1
