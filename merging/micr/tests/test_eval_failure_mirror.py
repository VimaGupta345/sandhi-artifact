"""
CPU-only tests: the eval-failure sentinel holds in ALL four MICR runners.

run_eval.py is covered in depth by test_eval_failure_sentinel.py. This suite checks
that run_eval_32b.py, run_eval_quantized.py and run_eval_lora.py agree on the
contract:

    evaluate_model_for_task -> float   the model was scored (0.0 means it scored 0.0)
                            -> None    no score was produced; the step was never judged

and that each runner routes its evals through the shared retry policy.

Run: conda run -n mergeenv python micr/tests/test_eval_failure_mirror.py
"""
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["MICR_EVAL_RETRIES"] = "2"
os.environ["MICR_EVAL_RETRY_BACKOFF"] = "0"

_pkg = types.ModuleType("merge_tools")
_pkg.__path__ = [str(REPO)]
sys.modules.setdefault("merge_tools", _pkg)
sys.path.insert(0, str(REPO))

from merge_tools.micr import eval_retry  # noqa: E402
from merge_tools.micr import run_eval, run_eval_32b, run_eval_lora, run_eval_quantized  # noqa: E402

RUNNERS = {
    "run_eval": run_eval,
    "run_eval_32b": run_eval_32b,
    "run_eval_quantized": run_eval_quantized,
    "run_eval_lora": run_eval_lora,
}
# lora's evaluate_model_for_task returns a parsed SUCCESS score immediately, so it
# has no `measured_zero` guard to exercise; the others do.
KEYWORD_RUNNERS = {k: v for k, v in RUNNERS.items() if k != "run_eval_lora"}


class _Evaluator:
    results = []

    def __init__(self, env_manager, env_config, eval_settings):
        self.eval_settings = eval_settings

    def evaluate(self, model_config, registry_key, run_id=1):
        return type(self).results.pop(0)


def _call(mod, results):
    _Evaluator.results = list(results)
    ctx = (object(), {}, {"use_vllm": True, "batch_size": 32})
    with mock.patch.object(mod, "TASK_REGISTRY", {"gsm8k-cot": {"evaluator": _Evaluator}}), \
         mock.patch.object(mod, "create_unified_eval_context", return_value=ctx):
        return mod.evaluate_model_for_task("/nonexistent/model", "gsm8k-cot")


class TestContractAcrossRunners(unittest.TestCase):
    def test_all_runners_annotate_optional_return(self):
        for name, mod in KEYWORD_RUNNERS.items():
            with self.subTest(runner=name):
                ann = mod.evaluate_model_for_task.__annotations__.get("return")
                self.assertEqual(str(ann), "typing.Optional[float]", f"{name} must return Optional[float]")

    def test_all_runners_expose_shared_retry_wrappers(self):
        for name, mod in RUNNERS.items():
            with self.subTest(runner=name):
                self.assertTrue(callable(getattr(mod, "evaluate_with_retry", None)))
                self.assertTrue(callable(getattr(mod, "evaluate_baseline_blocking", None)))

    def test_failure_returns_none_in_every_runner(self):
        for name, mod in RUNNERS.items():
            with self.subTest(runner=name):
                out = _call(mod, [
                    {"status": "FAILED", "error_log": "vllm launch race"},
                    {"status": "FAILED", "error_log": "vllm launch race"},
                ])
                self.assertIsNone(out, f"{name}: a failed eval must not be reported as 0.0")

    def test_success_returns_score_in_every_runner(self):
        for name, mod in RUNNERS.items():
            with self.subTest(runner=name):
                self.assertEqual(_call(mod, [{"status": "SUCCESS", "score": "81.20%"}]), 81.2)

    def test_genuine_zero_preserved_in_keyword_runners(self):
        for name, mod in KEYWORD_RUNNERS.items():
            with self.subTest(runner=name):
                out = _call(mod, [
                    {"status": "SUCCESS", "score": "0.0"},   # real measurement of zero
                    {"status": "FAILED", "error_log": "n/a"},  # fallback produces nothing
                ])
                self.assertEqual(out, 0.0, f"{name}: a measured 0.0 must stay 0.0, not become None")

    def test_lora_returns_measured_zero_directly(self):
        # lora returns any parsed SUCCESS score without consulting the fallback
        self.assertEqual(_call(run_eval_lora, [{"status": "SUCCESS", "score": "0.0"}]), 0.0)

    def test_missing_task_returns_none_in_every_runner(self):
        for name, mod in RUNNERS.items():
            with self.subTest(runner=name):
                with mock.patch.object(mod, "TASK_REGISTRY", {}):
                    self.assertIsNone(mod.evaluate_model_for_task("/x", "nope"))

    def test_no_registry_returns_none_in_every_runner(self):
        for name, mod in RUNNERS.items():
            with self.subTest(runner=name):
                with mock.patch.object(mod, "TASK_REGISTRY", None):
                    self.assertIsNone(mod.evaluate_model_for_task("/x", "gsm8k-cot"))


class TestWrappersDelegateToSharedPolicy(unittest.TestCase):
    def test_retry_budget_honored_per_runner(self):
        for name, mod in RUNNERS.items():
            with self.subTest(runner=name):
                with mock.patch.object(mod, "evaluate_model_for_task", side_effect=[None, None, None]) as m, \
                     mock.patch.object(eval_retry.time, "sleep"):
                    self.assertIsNone(mod.evaluate_with_retry("lbl", model_path="/x", task_or_domain="g"))
                self.assertEqual(m.call_count, 3, f"{name}: 1 initial + 2 retries")

    def test_baseline_blocks_then_succeeds_per_runner(self):
        for name, mod in RUNNERS.items():
            with self.subTest(runner=name):
                with mock.patch.object(mod, "evaluate_model_for_task", side_effect=[None, None, 77.5]), \
                     mock.patch.object(eval_retry.time, "sleep"):
                    self.assertEqual(
                        mod.evaluate_baseline_blocking("lbl", model_path="/x", task_or_domain="g"), 77.5
                    )

    def test_baseline_never_returns_zero_on_failure(self):
        ticks = [0.0]

        def fake_monotonic():
            ticks.append(ticks[-1] + eval_retry.baseline_max_wait_s() + 1.0)
            return ticks[-2]

        for name, mod in RUNNERS.items():
            with self.subTest(runner=name):
                with mock.patch.object(mod, "evaluate_model_for_task", return_value=None), \
                     mock.patch.object(eval_retry.time, "sleep"), \
                     mock.patch.object(eval_retry.time, "monotonic", side_effect=fake_monotonic):
                    with self.assertRaises(RuntimeError):
                        mod.evaluate_baseline_blocking("lbl", model_path="/x", task_or_domain="g")
                ticks[:] = [0.0]

    def test_lora_wrappers_accept_positional_args(self):
        """lora calls evaluate_model_for_task(tmp_dir, domain, gpu_ids=...) positionally.

        evaluate_with_retry also threads allow_fallback: only the final attempt may run
        the (genuinely different) HF fallback. With MICR_EVAL_RETRIES=2 the first attempt
        gets allow_fallback=False.
        """
        with mock.patch.object(run_eval_lora, "evaluate_model_for_task", return_value=64.0) as m, \
             mock.patch.object(eval_retry.time, "sleep"):
            out = run_eval_lora.evaluate_with_retry("lbl", "/tmp/x", "finance", gpu_ids="0")
        self.assertEqual(out, 64.0)
        m.assert_called_once_with("/tmp/x", "finance", allow_fallback=False, gpu_ids="0")


class TestNoRunnerStillReturnsZeroOnFailure(unittest.TestCase):
    """Guard against a future edit reintroducing the sentinel."""

    def test_source_has_no_bare_zero_return_in_evaluate(self):
        import inspect
        for name, mod in RUNNERS.items():
            with self.subTest(runner=name):
                src = inspect.getsource(mod.evaluate_model_for_task)
                # the only legal `return 0.0` is the measured-zero branch
                zero_returns = [ln.strip() for ln in src.splitlines() if ln.strip() == "return 0.0"]
                if name == "run_eval_lora":
                    self.assertEqual(len(zero_returns), 0, "lora returns parsed scores directly")
                else:
                    self.assertEqual(len(zero_returns), 1, f"{name}: only the measured_zero branch may return 0.0")
                    self.assertIn("measured_zero", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
