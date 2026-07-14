"""
End-to-end byte-identity tests for the delta-shard candidate save.

The load-bearing test is TestEndToEndMockMerge: perform a REAL merge into a tiny model,
then save it twice -- once with the delta-shard path, once with the legacy full save --
and assert every emitted file is byte-for-byte identical.

Also covers: hardlink survival across the accept path (rmtree + move), boundary-straddling
layers, the tied-embeddings key set, .bin fallback, verify-mode catching an undeclared
mutation, and the MICR_INCREMENTAL_SAVE=0 escape hatch.

Run: conda run -n mergeenv python micr/tests/test_delta_shard_save.py
"""
import filecmp
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MICR_VERIFY_SAVE", "0")  # opt in per-test
os.environ.setdefault("MICR_SAVE_SHARD_SIZE", "64KB")  # force many shards on tiny models

REPO = Path(__file__).resolve().parents[2]
_pkg = types.ModuleType("merge_tools")
_pkg.__path__ = [str(REPO)]
sys.modules.setdefault("merge_tools", _pkg)
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from merge_tools.micr import save_utils  # noqa: E402
from merge_tools.micr.run_eval import (  # noqa: E402
    _n_way_average_subcomponent_into_target, _save_model_to_temp_dir_for_eval)
from merge_tools.micr.top_k_experiment import _get_layer_module, _load_model  # noqa: E402

ATTRS = {"attn_q": "q_proj", "attn_k": "k_proj", "attn_v": "v_proj", "attn_o": "o_proj",
         "mlp_gate": "gate_proj", "mlp_up": "up_proj", "mlp_down": "down_proj"}


def tiny(seed, layers=4, tie=False):
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=512, hidden_size=128, intermediate_size=256,
                      num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=32, tie_word_embeddings=tie, torch_dtype="bfloat16")
    return Qwen3ForCausalLM(cfg).to(torch.bfloat16)


def reset_state():
    save_utils._fast_path_disabled = False
    save_utils._verify_budget = None
    for k in save_utils._stats:
        save_utils._stats[k] = 0


def dirs_byte_identical(a, b, testcase):
    fa = sorted(p.name for p in Path(a).iterdir() if p.is_file())
    fb = sorted(p.name for p in Path(b).iterdir() if p.is_file())
    testcase.assertEqual(fa, fb, "different file sets")
    for n in fa:
        testcase.assertTrue(filecmp.cmp(str(Path(a) / n), str(Path(b) / n), shallow=False),
                            f"{n} differs byte-for-byte")


class _Base(unittest.TestCase):
    def setUp(self):
        reset_state()
        self.root = tempfile.mkdtemp(prefix="micr_delta_")
        self.tmp_root = os.path.join(self.root, "tmp_eval_single")
        os.makedirs(self.tmp_root, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def make_working_dir(self, seed=1, layers=4, tie=False, shard=None, safe=True):
        """working_dir as the pipeline produces it after its first accept: our own full_save."""
        wd = os.path.join(self.root, "working")
        m = tiny(seed, layers, tie)
        if not safe:                                   # .bin source: no index, no marker
            m.save_pretrained(wd, safe_serialization=False)
            return wd
        if shard:                                      # override for the straddle fixture
            old = os.environ.get("MICR_SAVE_SHARD_SIZE")
            os.environ["MICR_SAVE_SHARD_SIZE"] = shard
            try: save_utils.full_save(m, wd)
            finally:
                if old is None: os.environ.pop("MICR_SAVE_SHARD_SIZE", None)
                else: os.environ["MICR_SAVE_SHARD_SIZE"] = old
        else:
            save_utils.full_save(m, wd)
        save_utils._stats["full"] = 0                  # fixture setup is not part of the assertions
        return wd

    def merge_one_layer(self, wd, layer, comps):
        """A real merge: average target with two donors into `comps` of `layer`."""
        tgt = _load_model(wd, device_hint="cpu", dtype_hint=torch.bfloat16)
        donors = [tiny(7), tiny(8)]
        applied = {}
        for comp in comps:
            attr = ATTRS[comp]
            grp = "attn" if comp.startswith("attn") else "mlp"
            _n_way_average_subcomponent_into_target(
                target_model=tgt, target_model_path=wd, target_layer_idx=layer,
                group_name=grp, component_attr=attr,
                contributors=[(tgt, layer)] + [(d, layer) for d in donors],
                device="cpu", target_std=None, enable_scaling=False)
            applied[comp] = attr
        return tgt, applied


class TestEndToEndMockMerge(_Base):
    """Real merge -> delta save vs full save -> every byte must match."""

    def _roundtrip(self, layer, comps, layers=4, tie=False, shard=None):
        wd = self.make_working_dir(layers=layers, tie=tie, shard=shard)
        tgt, applied = self.merge_one_layer(wd, layer, comps)
        keys = save_utils.changed_keys_for(layer, applied)

        delta = tempfile.mkdtemp(dir=self.tmp_root)
        save_utils.save_candidate(tgt, wd, delta, keys)

        full = tempfile.mkdtemp(dir=self.tmp_root)
        save_utils.full_save(tgt, full)

        dirs_byte_identical(delta, full, self)
        return wd, delta, full

    def test_mlp_bundle_byte_identical(self):
        self._roundtrip(2, ["mlp_gate", "mlp_up", "mlp_down"])
        self.assertEqual(save_utils.stats()["delta"], 1)
        self.assertGreater(save_utils.stats()["hardlinked_shards"], 0)

    def test_attn_bundle_byte_identical(self):
        self._roundtrip(1, ["attn_q", "attn_k", "attn_v", "attn_o"])

    def test_full_layer_bundle_byte_identical(self):
        self._roundtrip(3, list(ATTRS))

    def test_first_and_last_layer(self):
        self._roundtrip(0, list(ATTRS))
        reset_state()
        self._roundtrip(3, list(ATTRS))

    def test_boundary_straddling_layer(self):
        """Tiny shards force a layer's tensors across a shard boundary."""
        wd, delta, _ = self._roundtrip(2, list(ATTRS))
        wm = json.load(open(Path(wd) / save_utils.INDEX_NAME))["weight_map"]
        keys = save_utils.changed_keys_for(2, ATTRS)
        self.assertGreater(len({wm[k] for k in keys}), 1, "fixture must straddle shards")

    def test_tied_embeddings(self):
        self._roundtrip(1, ["mlp_gate"], tie=True)

    def test_reloaded_tensors_are_equal(self):
        wd, delta, full = self._roundtrip(2, list(ATTRS))
        a = _load_model(delta, device_hint="cpu", dtype_hint=torch.bfloat16).state_dict()
        b = _load_model(full, device_hint="cpu", dtype_hint=torch.bfloat16).state_dict()
        self.assertEqual(set(a), set(b))
        for k in a:
            self.assertTrue(torch.equal(a[k], b[k]), k)


class TestAcceptPathHardlinkSurvival(_Base):
    """rmtree(working_dir) + move(tmp) must not corrupt hardlinked shards."""

    def test_accept_then_reload(self):
        wd = self.make_working_dir()
        tgt, applied = self.merge_one_layer(wd, 1, list(ATTRS))
        keys = save_utils.changed_keys_for(1, applied)
        tmp = tempfile.mkdtemp(dir=self.tmp_root)
        save_utils.save_candidate(tgt, wd, tmp, keys)

        before = {k: v.clone() for k, v in tgt.state_dict().items()}
        shutil.rmtree(wd)              # exactly what the accept path does
        shutil.move(tmp, wd)

        after = _load_model(wd, device_hint="cpu", dtype_hint=torch.bfloat16).state_dict()
        for k in before:
            self.assertTrue(torch.equal(before[k], after[k]), f"{k} corrupted by accept")

    def test_second_delta_save_off_the_accepted_dir(self):
        """After an accept, working_dir is itself a delta output; a further delta must work."""
        wd = self.make_working_dir()
        for layer in (1, 2):
            tgt, applied = self.merge_one_layer(wd, layer, ["mlp_gate", "mlp_up", "mlp_down"])
            tmp = tempfile.mkdtemp(dir=self.tmp_root)
            save_utils.save_candidate(tgt, wd, tmp, save_utils.changed_keys_for(layer, applied))
            shutil.rmtree(wd); shutil.move(tmp, wd)
        self.assertEqual(save_utils.stats()["delta"], 2)
        m = _load_model(wd, device_hint="cpu", dtype_hint=torch.bfloat16)
        self.assertTrue(torch.isfinite(_get_layer_module(m, 2).mlp.gate_proj.weight).all())


class TestFallbacks(_Base):
    def test_bin_source_falls_back_to_full_save(self):
        wd = self.make_working_dir(safe=False)          # .bin, no safetensors index
        tgt, applied = self.merge_one_layer(wd, 1, ["mlp_gate"])
        tmp = tempfile.mkdtemp(dir=self.tmp_root)
        save_utils.save_candidate(tgt, wd, tmp, save_utils.changed_keys_for(1, applied))
        self.assertEqual(save_utils.stats()["full"], 1)
        self.assertEqual(save_utils.stats()["delta"], 0)

    def test_unknown_changed_key_falls_back(self):
        wd = self.make_working_dir()
        tgt, _ = self.merge_one_layer(wd, 1, ["mlp_gate"])
        tmp = tempfile.mkdtemp(dir=self.tmp_root)
        save_utils.save_candidate(tgt, wd, tmp, {"model.layers.99.mlp.gate_proj.weight"})
        self.assertEqual(save_utils.stats()["full"], 1)

    def test_no_changed_keys_falls_back(self):
        wd = self.make_working_dir()
        tgt, _ = self.merge_one_layer(wd, 1, ["mlp_gate"])
        tmp = tempfile.mkdtemp(dir=self.tmp_root)
        save_utils.save_candidate(tgt, wd, tmp, set())
        self.assertEqual(save_utils.stats()["full"], 1)

    def test_escape_hatch_forces_full_save(self):
        os.environ["MICR_INCREMENTAL_SAVE"] = "0"
        try:
            wd = self.make_working_dir()
            tgt, applied = self.merge_one_layer(wd, 1, ["mlp_gate"])
            tmp = tempfile.mkdtemp(dir=self.tmp_root)
            save_utils.save_candidate(tgt, wd, tmp, save_utils.changed_keys_for(1, applied))
            self.assertEqual(save_utils.stats()["full"], 1)
            self.assertEqual(save_utils.stats()["delta"], 0)
        finally:
            os.environ["MICR_INCREMENTAL_SAVE"] = "1"


class TestVerifyMode(_Base):
    def test_verify_passes_on_honest_save(self):
        os.environ["MICR_VERIFY_SAVE"] = "1"
        try:
            reset_state()
            wd = self.make_working_dir()
            tgt, applied = self.merge_one_layer(wd, 1, list(ATTRS))
            tmp = tempfile.mkdtemp(dir=self.tmp_root)
            save_utils.save_candidate(tgt, wd, tmp, save_utils.changed_keys_for(1, applied))
            self.assertFalse(save_utils._fast_path_disabled)
            full = tempfile.mkdtemp(dir=self.tmp_root)
            save_utils.full_save(tgt, full)
            dirs_byte_identical(tmp, full, self)
        finally:
            os.environ["MICR_VERIFY_SAVE"] = "0"

    def test_verify_catches_undeclared_mutation(self):
        """Mutate a tensor the block did NOT declare: verify must adopt full-save bytes."""
        os.environ["MICR_VERIFY_SAVE"] = "1"
        try:
            reset_state()
            wd = self.make_working_dir()
            tgt, applied = self.merge_one_layer(wd, 1, ["mlp_gate"])
            with torch.no_grad():                       # undeclared: different layer
                _get_layer_module(tgt, 3).mlp.up_proj.weight.add_(1.0)
            tmp = tempfile.mkdtemp(dir=self.tmp_root)
            save_utils.save_candidate(tgt, wd, tmp, save_utils.changed_keys_for(1, applied))
            self.assertTrue(save_utils._fast_path_disabled, "verify must disable the fast path")
            full = tempfile.mkdtemp(dir=self.tmp_root)
            save_utils.full_save(tgt, full)
            dirs_byte_identical(tmp, full, self)        # full-save bytes were adopted
        finally:
            os.environ["MICR_VERIFY_SAVE"] = "0"


class TestPipelineWiring(_Base):
    def test_save_helper_accepts_changed_keys_and_delegates(self):
        wd = self.make_working_dir()
        tgt, applied = self.merge_one_layer(wd, 1, list(ATTRS))
        d = _save_model_to_temp_dir_for_eval(
            tgt, base_model_path=wd, tmp_root=self.tmp_root,
            changed_keys=save_utils.changed_keys_for(1, applied))
        self.assertEqual(save_utils.stats()["delta"], 1)
        self.assertTrue((Path(d) / "config.json").exists())
        self.assertTrue(any(p.suffix == ".safetensors" for p in Path(d).iterdir()))

    def test_save_helper_without_changed_keys_is_legacy(self):
        wd = self.make_working_dir()
        tgt, _ = self.merge_one_layer(wd, 1, ["mlp_gate"])
        _save_model_to_temp_dir_for_eval(tgt, base_model_path=wd, tmp_root=self.tmp_root)
        self.assertEqual(save_utils.stats()["full"], 1)
        self.assertEqual(save_utils.stats()["delta"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
