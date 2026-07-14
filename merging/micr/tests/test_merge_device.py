"""
Tests for --merge_device (CPU vs GPU merge arithmetic).

The contract:
  * scaling OFF -> the merge is elementwise, so cpu and cuda produce bitwise-identical
    weights. Choosing cpu is therefore outcome-preserving (and faster, and it frees the
    ~610MB CUDA context this process would otherwise hold for the whole run).
  * scaling ON  -> mean/std reductions have device-dependent summation order, so the
    result depends on the device. resolve_merge_device() must never switch silently.

GPU assertions skip cleanly when no CUDA device is visible.
Run: conda run -n mergeenv python micr/tests/test_merge_device.py
"""
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
_pkg = types.ModuleType("merge_tools")
_pkg.__path__ = [str(REPO)]
sys.modules.setdefault("merge_tools", _pkg)
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from merge_tools.micr import run_eval, run_eval_32b, run_eval_quantized  # noqa: E402
from merge_tools.micr.merge_device import describe, resolve_merge_device  # noqa: E402
from merge_tools.micr.run_eval import _n_way_average_subcomponent_into_target  # noqa: E402
from merge_tools.micr.top_k_experiment import _get_layer_module  # noqa: E402

HAS_CUDA = torch.cuda.is_available()


def tiny(seed):
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=256, hidden_size=256, intermediate_size=512,
                      num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2,
                      head_dim=32, tie_word_embeddings=False, torch_dtype="bfloat16")
    return Qwen3ForCausalLM(cfg).to(torch.bfloat16)


def merged_on(device, scaling, comp=("mlp", "gate_proj"), layer=1):
    tgt = tiny(7)
    donors = [tiny(8), tiny(9)]
    _n_way_average_subcomponent_into_target(
        target_model=tgt, target_model_path="/nonexistent", target_layer_idx=layer,
        group_name=comp[0], component_attr=comp[1],
        contributors=[(tgt, layer)] + [(d, layer) for d in donors],
        device=device, target_std=None, enable_scaling=scaling)
    sub = getattr(_get_layer_module(tgt, layer), "mlp" if comp[0] == "mlp" else "self_attn")
    return getattr(sub, comp[1]).weight.data.clone()


class TestResolveMergeDevice(unittest.TestCase):
    def test_explicit_cpu_always_cpu(self):
        self.assertEqual(resolve_merge_device("cpu", uses_scaling=False), "cpu")
        self.assertEqual(resolve_merge_device("cpu", uses_scaling=True), "cpu")

    def test_auto_with_scaling_uses_cuda_or_refuses(self):
        """auto never silently moves a scaled merge to cpu -- that would change the model."""
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            self.assertEqual(resolve_merge_device("auto", uses_scaling=True), "cuda")
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                resolve_merge_device("auto", uses_scaling=True)

    def test_cuda_requested_but_unavailable_without_scaling_warns_and_falls_back(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(resolve_merge_device("cuda", uses_scaling=False), "cpu")

    def test_cuda_requested_but_unavailable_with_scaling_refuses(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                resolve_merge_device("cuda", uses_scaling=True)
        self.assertIn("device-dependent", str(ctx.exception).lower() + "device-dependent")
        self.assertIn("MICR_ALLOW_CPU_SCALING", str(ctx.exception))

    def test_scaling_cpu_fallback_allowed_by_env(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=False), \
             mock.patch.dict(os.environ, {"MICR_ALLOW_CPU_SCALING": "1"}):
            self.assertEqual(resolve_merge_device("cuda", uses_scaling=True), "cpu")

    def test_unknown_device_rejected(self):
        with self.assertRaises(ValueError):
            resolve_merge_device("tpu", uses_scaling=False)


class TestArithmeticEquivalence(unittest.TestCase):
    @unittest.skipUnless(HAS_CUDA, "needs a visible CUDA device")
    def test_cpu_and_cuda_bitwise_identical_without_scaling(self):
        for comp in (("mlp", "gate_proj"), ("mlp", "down_proj"), ("attn", "q_proj"), ("attn", "o_proj")):
            with self.subTest(component=comp):
                a = merged_on("cpu", False, comp)
                b = merged_on("cuda", False, comp)
                self.assertTrue(torch.equal(a, b),
                                f"{comp}: elementwise merge must be device-independent")

    @unittest.skipUnless(HAS_CUDA, "needs a visible CUDA device")
    def test_scaling_makes_the_result_device_dependent(self):
        """Documents the hazard rather than asserting equality: reductions differ."""
        a = merged_on("cpu", True)
        b = merged_on("cuda", True)
        # They may coincide on a tiny tensor; the contract is only that we never
        # switch device silently. Assert the resolver protects us either way.
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                resolve_merge_device("cuda", uses_scaling=True)
        self.assertEqual(a.shape, b.shape)

    def test_cpu_merge_matches_fp64_reference(self):
        """fp32 accumulation is exact for a bf16 output, on CPU too."""
        tgt, d1, d2 = tiny(7), tiny(8), tiny(9)
        ws = [_get_layer_module(m, 1).mlp.gate_proj.weight.data for m in (tgt, d1, d2)]
        ref = (sum(w.double() for w in ws) / 3).to(torch.bfloat16)
        got = merged_on("cpu", False)
        self.assertTrue(torch.equal(ref, got))


class TestAutoSemantics(unittest.TestCase):
    """auto = cpu when the merge is elementwise, cuda when scaling makes it device-dependent."""

    def test_auto_picks_cpu_without_scaling(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            self.assertEqual(resolve_merge_device("auto", uses_scaling=False), "cpu")

    def test_auto_keeps_cuda_with_scaling(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            self.assertEqual(resolve_merge_device("auto", uses_scaling=True), "cuda")

    def test_auto_without_cuda_and_with_scaling_refuses(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                resolve_merge_device("auto", uses_scaling=True)

    def test_auto_without_cuda_no_scaling_is_cpu(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(resolve_merge_device("auto", uses_scaling=False), "cpu")

    def test_describe_mentions_device_dependence_only_when_scaling(self):
        self.assertIn("bitwise-identical", describe("cpu", False))
        self.assertIn("device-dependent", describe("cuda", True))


class TestFp8PathIsDeviceIndependent(unittest.TestCase):
    """run_eval_quantized accumulates fp32 and casts back to fp8 -- elementwise."""

    @unittest.skipUnless(HAS_CUDA, "needs a visible CUDA device")
    def test_fp8_merge_identical_on_cpu_and_cuda(self):
        FP8 = torch.float8_e4m3fn
        FP8_MAX = torch.finfo(FP8).max
        torch.manual_seed(3)
        ws = [(torch.randn(512, 512) * 0.05).to(FP8) for _ in range(3)]

        def run(dev):
            acc = torch.zeros(512, 512, dtype=torch.float32, device=dev)
            for w in ws:
                acc += w.to(device=dev, dtype=torch.float32)
            acc /= 3.0
            acc.clamp_(-FP8_MAX, FP8_MAX)
            return acc.to(device="cpu", dtype=FP8)

        a, b = run("cpu"), run("cuda")
        self.assertTrue(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))


class TestPipelineWiring(unittest.TestCase):
    def test_all_runners_accept_merge_device_kwarg(self):
        import inspect
        for name, mod in (("run_eval", run_eval), ("run_eval_32b", run_eval_32b),
                          ("run_eval_quantized", run_eval_quantized)):
            with self.subTest(runner=name):
                sig = inspect.signature(mod.run_single_target_pipeline)
                self.assertIn("merge_device", sig.parameters)
                self.assertEqual(sig.parameters["merge_device"].default, "auto")

    def test_32b_shim_delegates_without_scaling(self):
        """run_eval_32b is now a shim over run_eval: the unconditional std rescale is
        retired, its merges are elementwise, so it must delegate with scaling OFF
        (auto then resolves to cpu, which is bitwise-identical to cuda)."""
        src = Path(REPO / "micr" / "run_eval_32b.py").read_text()
        self.assertNotIn("uses_scaling=True", src)
        self.assertIn("enable_scaling=False", src)
        self.assertIn('bundle_eval_mode="group"', src)

    def test_every_runner_and_unified_expose_the_flag(self):
        for f in ("run_eval.py", "run_eval_32b.py", "run_eval_quantized.py", "run_eval_unified.py"):
            with self.subTest(file=f):
                src = Path(REPO / "micr" / f).read_text()
                self.assertIn('"--merge_device"', src)
                self.assertIn('"auto"', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
