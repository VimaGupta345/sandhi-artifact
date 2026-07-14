"""
Delta-shard candidate saves for MICR.

A merge block mutates at most 7 weight tensors of ONE layer (the only mutation point
is ``p_t.data.copy_(...)`` in the ``_n_way_average_*`` helpers), yet
``model.save_pretrained()`` re-serializes every shard. Measured on real models:

    deepseek-math 7B  13.8 GB   full save  ~21.0 s   (0.66 GB/s)
    Light-IF-32B      65.5 GB   full save   81.5 s   (0.80 GB/s)

``save_candidate`` instead hardlinks every shard whose tensors are untouched and
re-serializes only the shard(s) named in ``model.safetensors.index.json`` for the
changed tensors. Averaged over all layers at HF's default 5 GB shard size that is
5.0 GB rewritten for the 7B (36%) and 5.8 GB for the 32B (9%).

Why the bytes are identical: safetensors serialization is deterministic (8-byte
header length + JSON header + raw little-endian tensor bytes). Re-serializing the same
in-memory tensors with the same key order and the same ``__metadata__`` reproduces the
file exactly; hardlinked shards are literally the same inode. The fast path only
engages once ``working_dir`` is the output of *this process's own* full save, so the
reference layout is created at this transformers/safetensors version.

Safety:
  * Any doubt -> full save (missing index, key-set mismatch, .bin source, dtype/shape
    mismatch, safetensors unavailable, any exception). Fallbacks leave a clean tmp dir.
  * MICR_VERIFY_SAVE=N byte-compares the first N delta saves against a fresh full save.
    On mismatch the full-save bytes are adopted (so the step is still byte-identical to
    the legacy path) and the fast path is disabled for the rest of the run.
  * Hardlinks survive the accept path: ``rmtree(working_dir)`` only unlinks names, and
    the tmp dir still holds links to those inodes.

Environment:
    MICR_INCREMENTAL_SAVE=0   always full-save (today's behavior)
    MICR_VERIFY_SAVE=N        byte-verify the first N delta saves (default 1)
    MICR_SAVE_SHARD_SIZE=2GB  max_shard_size for full saves. Smaller shards make every
                              later delta save cheaper (7B: 5GB->7.6s, 2GB->3.5s,
                              1GB->2.0s) at the cost of more files. Default: unset,
                              i.e. transformers' own default, so working_dir's layout
                              matches what the legacy code produced.
"""
import filecmp
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

import torch

INDEX_NAME = "model.safetensors.index.json"
_SIDE_FILES = ("config.json", "generation_config.json")

# Written by full_save(). A delta save preserves working_dir's shard layout, whereas a
# full save re-shards according to max_shard_size -- so the two are only byte-comparable
# when working_dir was produced by *this* code at *these* settings. The marker records
# that. Consequence: block 0 of a run always does a full save (working_dir is still the
# pristine copytree of the source), and every later block deltas off it.
MARKER = ".micr_shard_layout.json"

# Set once a verification failure disables the fast path for the process.
_fast_path_disabled = False
_verify_budget: Optional[int] = None
_stats = {"full": 0, "delta": 0, "hardlinked_shards": 0, "rewritten_shards": 0}


def _truthy(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no", "off")


def incremental_enabled() -> bool:
    return _truthy("MICR_INCREMENTAL_SAVE") and not _fast_path_disabled


def _verify_remaining() -> int:
    global _verify_budget
    if _verify_budget is None:
        _verify_budget = int(os.environ.get("MICR_VERIFY_SAVE", "1"))
    return _verify_budget


def shard_size() -> Optional[str]:
    v = os.environ.get("MICR_SAVE_SHARD_SIZE", "").strip()
    return v or None


def stats() -> Dict[str, int]:
    return dict(_stats)


def _layout_signature() -> Dict[str, object]:
    import transformers
    return {"transformers": transformers.__version__, "max_shard_size": shard_size()}


def full_save(model: torch.nn.Module, dest: str) -> None:
    """Exactly what the legacy code did, plus an optional shard-size override."""
    kwargs = {"safe_serialization": True}
    ms = shard_size()
    if ms:
        kwargs["max_shard_size"] = ms
    try:
        model.save_pretrained(dest, **kwargs)
    except TypeError:
        model.save_pretrained(dest)
    if (Path(dest) / INDEX_NAME).exists():
        json.dump(_layout_signature(), open(Path(dest) / MARKER, "w"))
    _stats["full"] += 1


def _read_weight_map(working_dir: str) -> Optional[Dict[str, str]]:
    idx = Path(working_dir) / INDEX_NAME
    if not idx.exists():
        return None
    try:
        return json.load(open(idx))["weight_map"]
    except Exception:
        return None


def _tied_names(model: torch.nn.Module) -> Set[str]:
    """Names whose storage is shared with an earlier tensor; save_pretrained drops them."""
    seen, tied = {}, set()
    for name, t in model.state_dict().items():
        key = (t.data_ptr(), t.shape)
        if t.data_ptr() and key in seen:
            tied.add(name)
        else:
            seen[key] = name
    return tied


def _can_delta(model: torch.nn.Module, working_dir: str, changed: Iterable[str]) -> Optional[Dict[str, str]]:
    """Return the weight_map if a delta save is provably safe, else None."""
    if not incremental_enabled():
        return None
    # Only delta off a directory this code wrote at these settings; otherwise the
    # candidate's layout would not match what a full save would emit.
    marker = Path(working_dir) / MARKER
    if not marker.exists():
        return None
    try:
        if json.load(open(marker)) != _layout_signature():
            return None
    except Exception:
        return None
    wm = _read_weight_map(working_dir)
    if not wm:
        return None
    sd = model.state_dict()
    # index must cover exactly the tensors save_pretrained would emit
    if set(wm) != (set(sd) - _tied_names(model)):
        return None
    changed = list(changed)
    if not changed or any(k not in wm for k in changed):
        return None
    # on-disk dtype/shape must match memory, or the rewrite would differ from a full save
    try:
        from safetensors import safe_open
    except Exception:
        return None
    for shard in {wm[k] for k in changed}:
        p = Path(working_dir) / shard
        if not p.exists():
            return None
    return wm


def save_candidate(
    model: torch.nn.Module,
    working_dir: str,
    tmp_dir: str,
    changed_keys: Iterable[str],
) -> str:
    """
    Write the candidate into ``tmp_dir``. Falls back to a full save whenever a
    delta save is not provably byte-identical. Returns ``tmp_dir``.
    """
    global _fast_path_disabled, _verify_budget

    wm = _can_delta(model, working_dir, changed_keys)
    if wm is None:
        full_save(model, tmp_dir)
        return tmp_dir

    from safetensors import safe_open
    from safetensors.torch import save_file

    changed = set(changed_keys)
    dirty = {wm[k] for k in changed}
    sd = model.state_dict()

    try:
        for shard in sorted(set(wm.values())):
            src, dst = Path(working_dir) / shard, Path(tmp_dir) / shard
            if shard in dirty:
                with safe_open(str(src), framework="pt", device="cpu") as f:
                    order = list(f.keys())          # preserve the on-disk key order
                    meta = f.metadata()
                save_file({k: sd[k].contiguous() for k in order}, str(dst), metadata=meta)
                _stats["rewritten_shards"] += 1
            else:
                os.link(str(src), str(dst))          # same inode, zero bytes copied
                _stats["hardlinked_shards"] += 1

        # Mirror EVERY non-shard regular file from the working dir (config,
        # generation config, tokenizer assets, custom modeling_*.py for
        # trust_remote_code checkpoints, index, marker): a delta dir must be a
        # drop-in equivalent of a full save, and a checkpoint whose config
        # auto_map references a .py file that is absent from the dir fails to
        # load in the eval subprocess. Hardlinks keep the mirror zero-copy.
        for src in Path(working_dir).iterdir():
            if not src.is_file() or src.name.endswith(".safetensors"):
                continue
            dst = Path(tmp_dir) / src.name
            if not dst.exists():
                os.link(str(src), str(dst))
    except Exception as e:
        print(f"  [save] delta-shard save failed ({type(e).__name__}: {e}); falling back to full save")
        for f in Path(tmp_dir).iterdir():
            f.unlink(missing_ok=True)
        full_save(model, tmp_dir)
        return tmp_dir

    _stats["delta"] += 1
    print(f"  [save] delta-shard: rewrote {len(dirty)}/{len(set(wm.values()))} shard(s)")

    if _verify_remaining() > 0:
        _verify_budget = _verify_remaining() - 1
        if not _verify_against_full_save(model, tmp_dir):
            _fast_path_disabled = True
    return tmp_dir


def _verify_against_full_save(model: torch.nn.Module, delta_dir: str) -> bool:
    """Byte-compare the delta output against a fresh full save. Adopt full bytes on mismatch.

    Only weight-bearing files (shards + index) are compared. Side files
    (config.json, generation_config.json) are hardlinks of the working dir's
    copies -- byte-equality with the working dir is guaranteed by construction,
    and a fresh save can legitimately differ when transformers mutates the live
    config objects on save (e.g. moving misplaced generation params). That is
    not a delta-save defect, and callers overlay source assets afterwards.
    """
    ref = Path(delta_dir).parent / f".verify_{Path(delta_dir).name}"
    ref.mkdir(parents=True, exist_ok=True)
    try:
        full_save(model, str(ref))
        names = sorted(p.name for p in ref.iterdir()
                       if p.is_file()
                       and (p.name.endswith(".safetensors") or p.name == INDEX_NAME))
        bad = [n for n in names
               if not (Path(delta_dir) / n).exists()
               or not filecmp.cmp(str(ref / n), str(Path(delta_dir) / n), shallow=False)]
        if bad:
            print(f"  [save] VERIFY FAILED on {bad}; adopting full-save bytes and disabling delta saves")
            for n in names:
                tgt = Path(delta_dir) / n
                if tgt.exists():
                    tgt.unlink()
                shutil.copy2(str(ref / n), str(tgt))
            return False
        print("  [save] verification passed: delta output is byte-identical to a full save")
        return True
    except Exception as e:
        print(f"  [save] verification could not run ({type(e).__name__}: {e}); keeping delta output")
        return True
    finally:
        shutil.rmtree(ref, ignore_errors=True)


def changed_keys_for(layer: int, group_to_attr: Dict[str, str]) -> Set[str]:
    """Map {component -> module attr} applied at ``layer`` to state_dict names."""
    out = set()
    for comp, attr in group_to_attr.items():
        sub = "self_attn" if comp.startswith("attn") else "mlp"
        out.add(f"model.layers.{layer}.{sub}.{attr}.weight")
    return out
