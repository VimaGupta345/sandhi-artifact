"""
The use_vllm=False fallback must run at most once, and only when it does something.

Two defects this pins down:

  1. HarnessEvaluator hardcodes "--use_vllm" (gsm8k) / "--model vllm" (humaneval) in its
     argv and never reads eval_settings["use_vllm"]. Its "fallback" therefore re-runs a
     BYTE-IDENTICAL subprocess. That predates the retry wrapper.
  2. evaluate_with_retry makes up to 3 attempts. If each attempt also ran the fallback,
     one failed step would cost 6 full evaluations (and 6x timeout_minutes).

Contract: skip the fallback when the constructed command is identical, and let only the
final retry attempt use the (genuine) HF fallback. The baseline blocks until it gets a
score, so it always allows the fallback.

Run: conda run -n mergeenv python micr/tests/test_fallback_policy.py
"""
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["MICR_EVAL_RETRIES"] = "2"
os.environ["MICR_EVAL_RETRY_BACKOFF"] = "0"

REPO = Path(__file__).resolve().parents[2]
_pkg = types.ModuleType("merge_tools")
_pkg.__path__ = [str(REPO)]
sys.modules.setdefault("merge_tools", _pkg)
sys.path.insert(0, str(REPO))

from merge_tools.micr import eval_retry, run_eval  # noqa: E402


class _HarnessLike:
    """Ignores use_vllm -- its argv is hardcoded. Like gsm8k / humaneval."""
    calls = 0
    results = []

    def __init__(self, env_manager, env_config, eval_settings):
        self.eval_settings = eval_settings

    def _construct_command(self, model_path, task_name):
        return (["python", "math_eval.py", "--use_vllm", model_path], "/cwd", "env")

    def evaluate(self, model_config, registry_key, run_id=1):
        type(self).calls += 1
        return type(self).results.pop(0)


class _SimpleLike:
    """Reads use_vllm and genuinely switches backend. Like mmlu / sst2."""
    calls = 0
    results = []

    def __init__(self, env_manager, env_config, eval_settings):
        self.eval_settings = eval_settings

    def _construct_command(self, model_path, task_name):
        backend = "vllm" if self.eval_settings.get("use_vllm") else "hf"
        return (["python", "-m", "lm_eval", "--model", backend, model_path], None, None)

    def evaluate(self, model_config, registry_key, run_id=1):
        type(self).calls += 1
        return type(self).results.pop(0)


def call(cls, results, **kw):
    cls.results = list(results)
    cls.calls = 0
    ctx = (object(), {}, {"use_vllm": True, "batch_size": 32})
    with mock.patch.object(run_eval, "TASK_REGISTRY", {"t": {"evaluator": cls}}), \
         mock.patch.object(run_eval, "create_unified_eval_context", return_value=ctx):
        out = run_eval.evaluate_model_for_task(model_path="/nonexistent", task_or_domain="t", **kw)
    return out, cls.calls


FAIL = {"status": "FAILED", "error_log": "vllm launch race"}
OK = {"status": "SUCCESS", "score": "81.20%"}


class TestFallbackIsRedundantDetection(unittest.TestCase):
    def test_harness_like_commands_are_identical(self):
        self.assertTrue(eval_retry.fallback_is_redundant(
            _HarnessLike, object(), {}, {"use_vllm": True}, "/m", "gsm8k-cot"))

    def test_simple_like_commands_differ(self):
        self.assertFalse(eval_retry.fallback_is_redundant(
            _SimpleLike, object(), {}, {"use_vllm": True}, "/m", "sst2"))

    def test_unconstructable_command_is_conservative(self):
        class Broken(_SimpleLike):
            def _construct_command(self, *a):
                raise FileNotFoundError("math_eval.py missing")
        self.assertFalse(eval_retry.fallback_is_redundant(
            Broken, object(), {}, {"use_vllm": True}, "/m", "t"),
            "if we cannot tell, keep today's behavior and run the fallback")


class TestSingleEvalPerAttempt(unittest.TestCase):
    def test_harness_failure_does_not_rerun_identical_command(self):
        out, calls = call(_HarnessLike, [FAIL])
        self.assertIsNone(out)
        self.assertEqual(calls, 1, "identical fallback must be skipped")

    def test_simple_failure_still_falls_back(self):
        out, calls = call(_SimpleLike, [FAIL, OK])
        self.assertEqual(out, 81.2)
        self.assertEqual(calls, 2, "HF fallback is a genuinely different command")

    def test_allow_fallback_false_suppresses_even_a_real_fallback(self):
        out, calls = call(_SimpleLike, [FAIL], allow_fallback=False)
        self.assertIsNone(out)
        self.assertEqual(calls, 1)

    def test_measured_zero_still_preserved(self):
        out, calls = call(_HarnessLike, [{"status": "SUCCESS", "score": "0.0"}])
        self.assertEqual(out, 0.0, "a genuinely measured 0.0 is not a failure")


class TestRetryUsesFallbackOnlyOnLastAttempt(unittest.TestCase):
    def test_only_final_attempt_allows_fallback(self):
        seen = []

        def fake(**kw):
            seen.append(kw["allow_fallback"])
            return None

        with mock.patch.object(eval_retry.time, "sleep"):
            out = eval_retry.evaluate_with_retry(fake, "lbl", model_path="/x", task_or_domain="t")
        self.assertIsNone(out)
        self.assertEqual(seen, [False, False, True], "1 initial + 2 retries; fallback only last")

    def test_worst_case_evaluator_calls(self):
        """3 vLLM attempts + 1 HF fallback = 4, not 6."""
        _SimpleLike.results = [FAIL, FAIL, FAIL, FAIL]
        _SimpleLike.calls = 0
        ctx = (object(), {}, {"use_vllm": True, "batch_size": 32})
        with mock.patch.object(run_eval, "TASK_REGISTRY", {"t": {"evaluator": _SimpleLike}}), \
             mock.patch.object(run_eval, "create_unified_eval_context", return_value=ctx), \
             mock.patch.object(eval_retry.time, "sleep"):
            out = eval_retry.evaluate_with_retry(
                run_eval.evaluate_model_for_task, "lbl",
                model_path="/nonexistent", task_or_domain="t")
        self.assertIsNone(out)
        self.assertEqual(_SimpleLike.calls, 4)

    def test_harness_worst_case_is_three_not_six(self):
        _HarnessLike.results = [FAIL, FAIL, FAIL]
        _HarnessLike.calls = 0
        ctx = (object(), {}, {"use_vllm": True, "batch_size": 32})
        with mock.patch.object(run_eval, "TASK_REGISTRY", {"t": {"evaluator": _HarnessLike}}), \
             mock.patch.object(run_eval, "create_unified_eval_context", return_value=ctx), \
             mock.patch.object(eval_retry.time, "sleep"):
            eval_retry.evaluate_with_retry(
                run_eval.evaluate_model_for_task, "lbl",
                model_path="/nonexistent", task_or_domain="t")
        self.assertEqual(_HarnessLike.calls, 3, "was 6 before: 3 attempts x (vllm + identical retry)")


class TestBaselineAlwaysAllowsFallback(unittest.TestCase):
    def test_baseline_passes_allow_fallback_true(self):
        seen = []

        def fake(**kw):
            seen.append(kw["allow_fallback"])
            return 81.2 if len(seen) == 3 else None

        with mock.patch.object(eval_retry.time, "sleep"):
            out = eval_retry.evaluate_baseline_blocking(fake, "lbl", model_path="/x", task_or_domain="t")
        self.assertEqual(out, 81.2)
        self.assertEqual(seen, [True, True, True], "baseline must always allow the real fallback")


class TestAllRunnersExposeTheFlag(unittest.TestCase):
    def test_signature(self):
        import inspect
        from merge_tools.micr import run_eval_32b, run_eval_lora, run_eval_quantized
        for name, mod in (("run_eval", run_eval), ("run_eval_32b", run_eval_32b),
                          ("run_eval_quantized", run_eval_quantized), ("run_eval_lora", run_eval_lora)):
            with self.subTest(runner=name):
                sig = inspect.signature(mod.evaluate_model_for_task)
                self.assertIn("allow_fallback", sig.parameters)
                self.assertIs(sig.parameters["allow_fallback"].default, True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
