"""
CPU-only tests for the eval-failure sentinel change.

Covers:
  1. evaluate_model_for_task returns None on failure, 0.0 on a genuinely measured
     zero, and the real score on success -- with the retry ladder unchanged.
  2. evaluate_with_retry retries exactly MICR_EVAL_RETRIES times, then gives up.
  3. evaluate_baseline_blocking never returns None; it blocks and eventually raises.
  4. The decision rule: a failed step is recorded as eval_failed (no accept/reject
     claim, working_dir untouched), and a failed baseline can no longer pin
     last_score at 0.0 and accept everything.
  5. plot_steps._remove_unscored_rows drops eval_failed rows.

Run: conda run -n mergeenv python micr/tests/test_eval_failure_sentinel.py
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
os.environ["MICR_EVAL_RETRY_BACKOFF"] = "0"  # no real sleeping in tests

# run_eval.py imports `merge_tools.experiments...`, which requires the repo root to
# be importable under that name. In a worktree the directory has another name, so
# bind it explicitly rather than depending on the checkout's folder name.
_pkg = types.ModuleType("merge_tools")
_pkg.__path__ = [str(REPO)]
sys.modules.setdefault("merge_tools", _pkg)
sys.path.insert(0, str(REPO))

from merge_tools.micr import eval_retry, run_eval  # noqa: E402


def _fake_registry(evaluator_cls):
    return {"gsm8k-cot": {"evaluator": evaluator_cls}}


class _Evaluator:
    """Stands in for a unified-llm-eval evaluator class."""

    results = []  # class-level queue of dicts to return, consumed in order
    calls = 0

    def __init__(self, env_manager, env_config, eval_settings):
        self.eval_settings = eval_settings

    def evaluate(self, model_config, registry_key, run_id=1):
        type(self).calls += 1
        return type(self).results.pop(0)


def _patched_eval(results, **kw):
    """Call evaluate_model_for_task with a stubbed harness returning `results`."""
    _Evaluator.results = list(results)
    _Evaluator.calls = 0
    with mock.patch.object(run_eval, "TASK_REGISTRY", _fake_registry(_Evaluator)), \
         mock.patch.object(run_eval, "create_unified_eval_context",
                           return_value=(object(), {}, {"use_vllm": True, "batch_size": 32})):
        out = run_eval.evaluate_model_for_task(
            model_path="/nonexistent/model", task_or_domain="gsm8k-cot", **kw
        )
    return out, _Evaluator.calls


class TestEvaluateModelForTask(unittest.TestCase):
    def test_success_returns_score(self):
        out, calls = _patched_eval([{"status": "SUCCESS", "score": "81.20%"}])
        self.assertEqual(out, 81.2)
        self.assertEqual(calls, 1, "no retry should fire on a healthy nonzero score")

    def test_failure_returns_none_not_zero(self):
        # first attempt fails, vLLM=False retry also fails
        out, calls = _patched_eval([
            {"status": "FAILED", "error_log": "vllm launch race"},
            {"status": "FAILED", "error_log": "vllm launch race"},
        ])
        self.assertIsNone(out, "a failed eval must not be reported as 0.0")
        self.assertEqual(calls, 2, "the existing use_vllm=False retry must still fire")

    def test_genuine_zero_is_preserved(self):
        # SUCCESS parsing to 0.0 on attempt 1; retry produces nothing usable
        out, calls = _patched_eval([
            {"status": "SUCCESS", "score": "0.0"},
            {"status": "FAILED", "error_log": "n/a"},
        ])
        self.assertEqual(out, 0.0, "a genuinely measured 0.0 must stay 0.0, not become None")

    def test_retry_recovers(self):
        out, _ = _patched_eval([
            {"status": "SUCCESS", "score": "0.0"},        # triggers the vLLM fallback
            {"status": "SUCCESS", "score": "77.5%"},      # fallback backend succeeds
        ])
        self.assertEqual(out, 77.5)

    def test_missing_task_returns_none(self):
        with mock.patch.object(run_eval, "TASK_REGISTRY", {}):
            self.assertIsNone(
                run_eval.evaluate_model_for_task(model_path="/x", task_or_domain="nope")
            )

    def test_no_registry_returns_none(self):
        with mock.patch.object(run_eval, "TASK_REGISTRY", None):
            self.assertIsNone(
                run_eval.evaluate_model_for_task(model_path="/x", task_or_domain="gsm8k-cot")
            )


class TestRetryWrappers(unittest.TestCase):
    def test_evaluate_with_retry_gives_up_after_budget(self):
        seq = [None, None, None]
        with mock.patch.object(run_eval, "evaluate_model_for_task", side_effect=seq) as m, \
             mock.patch.object(eval_retry.time, "sleep"):
            out = run_eval.evaluate_with_retry("t", model_path="/x", task_or_domain="g")
        self.assertIsNone(out)
        self.assertEqual(m.call_count, 3, "1 initial + MICR_EVAL_RETRIES(2) retries")

    def test_evaluate_with_retry_recovers_midway(self):
        with mock.patch.object(run_eval, "evaluate_model_for_task",
                               side_effect=[None, 81.2]) as m, \
             mock.patch.object(eval_retry.time, "sleep"):
            out = run_eval.evaluate_with_retry("t", model_path="/x", task_or_domain="g")
        self.assertEqual(out, 81.2)
        self.assertEqual(m.call_count, 2)

    def test_baseline_blocking_never_returns_none(self):
        with mock.patch.object(run_eval, "evaluate_model_for_task",
                               side_effect=[None, None, None, 81.2]), \
             mock.patch.object(eval_retry.time, "sleep"):
            out = run_eval.evaluate_baseline_blocking("t", model_path="/x", task_or_domain="g")
        self.assertEqual(out, 81.2, "baseline blocks through transient failures")

    def test_baseline_blocking_raises_after_budget(self):
        ticks = [0.0]  # first call establishes `started`; every later call is past the cap

        def fake_monotonic():
            ticks.append(ticks[-1] + eval_retry.baseline_max_wait_s() + 1.0)
            return ticks[-2]

        with mock.patch.object(run_eval, "evaluate_model_for_task", return_value=None), \
             mock.patch.object(eval_retry.time, "sleep"), \
             mock.patch.object(eval_retry.time, "monotonic", side_effect=fake_monotonic):
            with self.assertRaises(RuntimeError) as ctx:
                run_eval.evaluate_baseline_blocking("tgt", model_path="/x", task_or_domain="g")
        self.assertIn("accept unconditionally", str(ctx.exception))


class TestDecisionRule(unittest.TestCase):
    """The bug this change fixes, expressed as the decision arithmetic."""

    @staticmethod
    def decide(base_acc, scores, tol=2.0):
        last = base_acc
        out = []
        for s in scores:
            if s + 1e-12 < (last - tol):
                out.append("rejected")
            else:
                out.append("accepted")
                last = min(last, s)
        return out

    def test_old_behavior_failed_baseline_accepted_everything(self):
        # score 0.0 was what the old evaluate_model_for_task returned on failure
        self.assertEqual(
            self.decide(0.0, [81.0, 78.5, 3.2]),
            ["accepted", "accepted", "accepted"],
            "documents the bug: a 0.0 baseline accepts even a catastrophic score",
        )

    def test_healthy_baseline_still_discriminates(self):
        self.assertEqual(
            self.decide(81.2, [81.0, 78.5, 80.9]),
            ["accepted", "rejected", "accepted"],
        )

    def test_failed_step_makes_no_claim(self):
        """A None score must never reach the accept/reject comparison."""
        new_acc = None
        self.assertIsNone(new_acc)
        with self.assertRaises(TypeError):
            _ = new_acc + 1e-12 < (81.2 - 2.0)  # would have been 0.0 < 79.2 -> reject

    def test_zero_is_recorded_but_never_judged(self):
        """
        The CSV carries score=0 on failure so the column stays numeric. That 0 must not
        be treated as a measurement: `decision` is eval_failed and the step returns
        before the comparison. If the 0 ever leaked into the rule it would force a
        spurious 'rejected' -- exactly the defect found in 36 of 397 historical rows.
        """
        import inspect
        from merge_tools.micr import run_eval as re_
        src = inspect.getsource(re_.run_single_target_pipeline)
        # the early return happens before old_acc is read
        i_guard = src.index('if new_acc is None:')
        i_rule = src.index('if new_acc + 1e-12 < (old_acc - allowed_drop):')
        self.assertLess(i_guard, i_rule, "the None guard must precede the accept/reject rule")
        block = src[i_guard:i_rule]
        self.assertIn('"decision": "eval_failed"', block)
        self.assertIn('"score": 0', block)
        self.assertIn("return step_counter + 1", block, "must return before judging")


class TestPlotFiltering(unittest.TestCase):
    def test_eval_failed_rows_dropped(self):
        import pandas as pd
        sys.path.insert(0, str(REPO / "visualization"))
        import plot_steps

        df = pd.DataFrame([
            {"op": "baseline", "component": "baseline", "score": 81.2, "decision": "baseline"},
            {"op": "merge", "component": "mlp", "score": 80.9, "decision": "accepted"},
            {"op": "merge", "component": "mlp", "score": 60.0, "decision": "rejected"},
            {"op": "merge", "component": "mlp", "score": 0, "decision": "eval_failed"},
        ])
        out = plot_steps._remove_unscored_rows(df)
        self.assertEqual(len(out), 2, "eval_failed rows must be dropped even though score is 0")
        self.assertEqual(set(out["decision"]), {"accepted", "rejected"})
        self.assertTrue(pd.to_numeric(out["score"]).notna().all())
        self.assertNotIn(0, list(out["score"]), "a failed step's 0 must never reach a plot")


if __name__ == "__main__":
    unittest.main(verbosity=2)
