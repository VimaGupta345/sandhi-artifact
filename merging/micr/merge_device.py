"""
Where the MICR merge arithmetic runs, and why the choice is not free.

The merge upcasts every bf16 (or fp8) contributor to fp32, sums them, divides by k,
optionally moment-matches, and casts once back to the storage dtype.

Two facts, both measured on real Qwen3-32B weights (layer 30, mlp.gate_proj,
131,072,000 elements) and on the FP8 path (16,777,216 bytes):

1. WITHOUT scaling the merge is purely elementwise, so every output element depends
   only on the same-index inputs. CPU and CUDA agree BITWISE: 0 elements differ.
   CPU is also faster -- 1.814s vs 2.303s for a full-layer block -- because the merge
   is memory-bound and the CUDA path pays host<->device transfers for arithmetic
   neither device finds hard.

2. WITH scaling the merge computes mean/std reductions. Floating-point addition is not
   associative, and CPU and CUDA sum in different orders, so the merged weights depend
   on the device: 1,880 of 131,072,000 elements differ. That changes the eval score and
   can change an accept/reject decision.

Hence `auto`: run on CPU whenever that is provably identical, and on CUDA when the run
uses scaling, so no existing configuration silently changes its numbers.

There is a second reason to prefer CPU. Allocating any CUDA tensor creates a CUDA
context in the MICR process -- measured at 612 MB, of which 610 MB survives
`torch.cuda.empty_cache()` because torch does not account for it. That memory is
unavailable to the vLLM eval subprocess, which sizes its KV cache from what is free.
A CPU merge keeps the parent at 0 MB on the GPU. (`import torch`,
`torch.cuda.is_available()` and `torch.cuda.empty_cache()` do NOT create the context.)

Environment:
    MICR_ALLOW_CPU_SCALING=1   permit a CUDA->CPU fallback even with scaling on
"""
import os

import torch

CHOICES = ("cuda", "cpu", "auto")


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def resolve_merge_device(requested: str, uses_scaling: bool) -> str:
    """
    Resolve the merge device once per run. Never switch silently.

    `uses_scaling` must be True whenever the run applies moment matching or a std
    rescale -- i.e. `--enable-scaling` for run_eval / run_eval_quantized.
    (run_eval_32b is now a shim that delegates to run_eval with scaling off; its
    former unconditional `target_std` rescale is retired.)
    """
    if requested not in CHOICES:
        raise ValueError(f"Unknown merge device {requested!r}; expected one of {CHOICES}")

    available = torch.cuda.is_available()

    if requested == "cpu":
        return "cpu"

    if requested == "auto":
        # CPU is bitwise-identical and faster when the merge is elementwise.
        # With scaling the result is device-dependent, so keep today's device.
        if not uses_scaling:
            return "cpu"
        return "cuda" if available else _cpu_with_scaling_guard()

    # requested == "cuda"
    if available:
        return "cuda"
    if uses_scaling:
        return _cpu_with_scaling_guard()
    print("  [merge] CUDA unavailable; falling back to CPU (elementwise merge, bitwise identical).")
    return "cpu"


def _cpu_with_scaling_guard() -> str:
    if not _truthy("MICR_ALLOW_CPU_SCALING"):
        raise RuntimeError(
            "The merge would run on CPU while scaling is enabled, but scaling uses "
            "mean/std reductions whose summation order differs between CPU and CUDA. "
            "The merged weights -- and therefore the eval score and the accept/reject "
            "decision -- would not match a CUDA run. Pass --merge_device cpu to accept "
            "CPU arithmetic explicitly, or set MICR_ALLOW_CPU_SCALING=1."
        )
    print("  [merge] WARNING: CPU arithmetic with scaling enabled; "
          "results will differ from a CUDA run.")
    return "cpu"


def describe(device: str, uses_scaling: bool) -> str:
    if not uses_scaling:
        return f"[merge] arithmetic device: {device} (elementwise; bitwise-identical on cpu and cuda)"
    return f"[merge] arithmetic device: {device} (scaling uses reductions; results are device-dependent)"
