"""
One vLLM engine per run instead of one per step.

Every MICR step spawns a fresh evaluation subprocess that boots a vLLM engine from
scratch: process spawn, imports, weight load, torch.compile, CUDA-graph capture. Measured
on this machine: **26-28 s per boot**, against 5-9 s of actual generation. An in-place
``collective_rpc("reload_weights")`` costs **2.16 s**.

Verified on real merged weights (deepseek-coder + deepseek-math averaged at layer 10,
bf16, 96 generated tokens across 4 prompts): after reload_weights + reset_prefix_cache the
engine produced **token ids and per-token logprobs bitwise identical** to a fresh boot on
the same files. The merge changed 3 of 4 generations, so the comparison is meaningful.

Why this is not obviously safe, and why we check:
    A fresh boot runs ``load_weights`` **then** ``process_weights_after_loading``
    (model_loader/base_loader.py:55-56). ``reload_weights`` runs only ``load_weights``
    (v1/worker/gpu_model_runner.py:3423). For bf16 the post-load step is inert -- proven
    above. For **quantized checkpoints it is load-bearing** (weight repacking), so this
    module refuses to serve them.

Metric parity: the arguments are taken from the evaluator's own ``_construct_command``
output and fed to the harness's own parser (``math_eval.parse_args`` /
``VLLM.create_from_arg_string``). Nothing about *what* is evaluated is reimplemented here;
only *where the engine lives* changes.

Any unsupported task, any exception, any doubt -> return None, and the caller runs the
legacy subprocess path unchanged.

Spawn-safety (documented constraint):
    Booting the engine in-process makes vLLM start its EngineCore via the
    ``multiprocessing`` *spawn* method whenever CUDA is already initialized in the caller
    (vllm system_utils warns and overrides to spawn). A spawn child **re-imports the
    caller's __main__ module**; therefore the driver script that calls
    :func:`evaluate` must keep its work under ``if __name__ == "__main__":`` (standard
    Python requirement, multiprocessing docs "Safe importing of main module"). If it does
    not, the child re-executes the whole driver during its bootstrap; when that replay
    reaches this module, starting a grandchild engine raises ``RuntimeError("An attempt
    has been made to start a new process before the current process has finished its
    bootstrapping phase")``. :func:`evaluate` now detects that state
    (``current_process()._inheriting`` -- the exact flag CPython's
    ``multiprocessing.spawn._check_not_importing_main`` checks) and returns None without
    booting and without poisoning ``_disabled_reason``, so the parent run is unaffected.

Environment:
    MICR_PERSISTENT_EVAL=0   disable; every eval spawns a subprocess as before
"""
import contextlib
import json
import multiprocessing
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

# The engine reloads weights from the path it booted with, so the candidate must live at
# a stable location for the whole run.
CANDIDATE_DIRNAME = "candidate_eval"

_session: Optional["_Session"] = None
_disabled_reason: Optional[str] = None


def enabled() -> bool:
    if _disabled_reason:
        return False
    return os.environ.get("MICR_PERSISTENT_EVAL", "1").strip().lower() not in ("0", "false", "no", "off")


def session_active() -> bool:
    """True while a booted engine session is resident in this process.

    A successfully engine-served eval always leaves the session resident (for
    the next reload), so ``not session_active()`` right after an eval proves
    the engine did NOT serve it. Callers use this to enforce engine-
    consistency (see mirror_into_candidate)."""
    return _session is not None


def disable(reason: str) -> None:
    """Permanently disable the persistent path for this process: ``enabled()``
    turns False and every subsequent eval takes the subprocess path."""
    _disable(reason)


def candidate_dir(tmp_root: str) -> str:
    return os.path.join(tmp_root, CANDIDATE_DIRNAME)


def mirror_into_candidate(src_dir: str, tmp_root: str) -> str:
    """Recreate ``src_dir`` at the stable candidate path and return that path.

    ENGINE-CONSISTENCY primitive. The resident engine can carry a small
    CONSTANT per-task config offset against subprocess cold boots (measured:
    humaneval 82.32/82.32 subprocess vs 81.71/81.71 persistent -- exactly one
    problem, bit-stable within each path). Any run that scores candidates with
    the engine must therefore score its BASELINE with the same engine, or the
    offset contaminates every (baseline - candidate) drop. Callers do that by
    mirroring the unmodified model here and evaluating this path: the basename
    gate in :func:`evaluate` then routes it through the engine exactly like a
    per-step candidate.

    The bytes are identical to the source, so files are hardlinked when the
    filesystem allows it (silent per-file copy fallback). The directory is
    recreated from scratch, mirroring how per-step candidate saves rebuild it.
    """
    dst = candidate_dir(tmp_root)
    if os.path.isdir(dst):
        shutil.rmtree(dst, ignore_errors=True)
    for root, _dirs, files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        target_root = dst if rel == "." else os.path.join(dst, rel)
        os.makedirs(target_root, exist_ok=True)
        for fname in files:
            s = os.path.join(root, fname)
            t = os.path.join(target_root, fname)
            # NEVER write through an existing path: it may be a hardlink into
            # another directory. (Observed data loss: two jobs raced a SHARED
            # candidate dir; the copy fallback caught the resulting
            # FileExistsError and wrote model A's shard THROUGH model B's
            # leftover link, corrupting B's SOURCE checkpoint.) Unlink drops
            # only the name; the source inode is untouched.
            if os.path.lexists(t):
                os.unlink(t)
            try:
                os.link(s, t)
            except OSError:
                # true cross-device fallback: write a temp name, then rename
                tmp = t + ".mirror_tmp"
                shutil.copy2(s, tmp)
                os.replace(tmp, t)
    return dst


def _disable(reason: str) -> None:
    global _disabled_reason, _session
    _disabled_reason = reason
    print(f"  [persistent-eval] disabled for this run: {reason}")
    _session = None


def _in_spawn_bootstrap() -> bool:
    """True while a multiprocessing *spawn* child is importing the parent's __main__.

    This is the exact condition under which ``Process.start()`` raises
    RuntimeError('... before the current process has finished its bootstrapping phase'):
    CPython sets ``_inheriting`` on the child's current_process() for the duration of
    ``multiprocessing.spawn.prepare()`` and ``_check_not_importing_main`` tests it.
    In that state we are a re-imported copy of the driver (e.g. vLLM's own EngineCore
    child), so evaluating here would both crash and duplicate the parent's work.
    """
    return bool(getattr(multiprocessing.current_process(), "_inheriting", False))


def _is_quantized(model_dir: str) -> bool:
    try:
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
    except Exception:
        return False
    return bool(cfg.get("quantization_config") or cfg.get("compression_config"))


@contextlib.contextmanager
def _chdir(path: Optional[str]):
    if not path:
        yield
        return
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


# The vendored math harness ships top-level modules whose names collide with packages
# already imported by MICR: `utils` (unified-llm-eval/utils/) and `evaluate` (HuggingFace).
# The subprocess path never noticed because it ran with cwd=math_dir, so the harness's
# own files won sys.path[0]. In-process we must swap them in and out around each call.
_HARNESS_MODULES = ("math_eval", "utils", "parser", "evaluate", "data_loader",
                    "trajectory", "model_utils", "python_executor", "grader")
_harness_cache: dict = {}


@contextlib.contextmanager
def _harness_imports(cwd: str):
    def _pop(names):
        out = {}
        for k in list(sys.modules):
            if k in names or k.split(".")[0] in names:
                out[k] = sys.modules.pop(k)
        return out

    outer = _pop(_HARNESS_MODULES)          # stash MICR's versions
    old_path = list(sys.path)
    sys.path.insert(0, cwd)
    if _harness_cache:
        sys.modules.update(_harness_cache)  # reuse the harness's versions
    try:
        yield
        for n in _HARNESS_MODULES:          # remember them for the next call
            if n in sys.modules:
                _harness_cache[n] = sys.modules[n]
    finally:
        sys.path[:] = old_path
        _pop(_HARNESS_MODULES)              # remove the harness's versions
        sys.modules.update(outer)           # restore MICR's


@contextlib.contextmanager
def _argv(argv):
    old = sys.argv
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = old


class _Session:
    """A booted engine bound to one task family and one candidate directory."""

    def __init__(self, kind: str, model_dir: str):
        self.kind = kind            # "math" | "lm_eval"
        self.model_dir = model_dir
        self.llm = None             # vllm.LLM
        self.lm = None              # lm_eval LM wrapper (lm_eval kind only)

    def reload(self) -> float:
        import time
        t0 = time.time()
        self.llm.collective_rpc("reload_weights")
        self.llm.reset_prefix_cache()
        return time.time() - t0


def _warm_up(s: "_Session") -> None:
    """One reload cycle immediately after boot, BEFORE the first served eval.

    Measured (identity_check --mode baseline-consistency, humaneval, identical
    bytes through one engine): the very FIRST eval after a cold boot read
    81.71, while every post-reload eval read 82.32 -- self-consistent AND equal
    to the subprocess score exactly. The boot-state artifact belongs to the
    first eval only; a warm-up reload puts ALL served evals -- the
    engine-consistent baseline included -- in the reload-stable regime, for
    every caller, with no caller changes. (~2s once per run.)

    Post-warmup the engine agrees with subprocess cold boots on reload-stable
    tasks (humaneval 82.32, ifeval 21.0x3); gsm8k keeps an irreducible ~+/-0.3
    generation nondeterminism that subprocess cold boots show too -- accepted
    noise against the 2.0 tolerance, no scheme removes it. The
    measure-your-own-baseline policy in run_eval stays: it still guards the
    noise-correlation case even where the constant offset is gone.
    """
    try:
        dt = s.reload()
        print(f"  [persistent-eval] post-boot warm-up reload ({dt:.2f}s); "
              f"first served eval runs in the reload-stable regime")
    except Exception as e:
        # Warm-up is an accuracy-consistency aid, not a correctness gate; a
        # failure here must not take down the session that just booted fine.
        print(f"  [persistent-eval] warm-up reload failed ({type(e).__name__}: {e}); "
              f"first eval will run in boot state")


def _tp_size() -> int:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return max(1, len([g for g in vis.split(",") if g.strip()]))


# ---------------------------------------------------------------- math_eval (gsm8k)

def _run_math(command, cwd, model_dir) -> Optional[float]:
    """command == ["python", "<math_dir>/math_eval.py", *flags] built by HarnessEvaluator."""
    global _session
    with _harness_imports(cwd):
        import math_eval  # resolved from the harness dir, not from unified-llm-eval

        with _argv(["math_eval.py"] + list(command[2:])):
            args = math_eval.parse_args()

        # math_eval.py's __main__ runs set_seed(args.seed) before setup(); mirror it.
        # (Inert for the default greedy full-set config, but keeps the in-process run
        # byte-for-byte on the harness's own code path.)
        math_eval.set_seed(args.seed)

        if _session is not None and (_session.kind != "math"
                                     or _session.model_dir != model_dir):
            # Different task family OR different candidate path: the engine can only
            # reload from the path it booted with, so free it and boot fresh.
            shutdown()
        if _session is None:
            from vllm import LLM
            # Mirror math_eval.setup() exactly: same constructor, same defaults.
            print("  [persistent-eval] booting engine (once per run)")
            s = _Session("math", model_dir)
            s.llm = LLM(model=model_dir, tensor_parallel_size=_tp_size(), trust_remote_code=True,
                        gpu_memory_utilization=0.8)  # match the subprocess evaluators: 0.9 cannot boot next to the caller's GPU-resident model
            _session = s
            _warm_up(s)
        else:
            dt = _session.reload()
            print(f"  [persistent-eval] reloaded weights in {dt:.2f}s (vs ~26s cold boot)")

        data_name = args.data_names.split(",")[0]
        with _chdir(cwd):
            result = math_eval.main(_session.llm, None, data_name, args)
    acc = result.get("acc") if isinstance(result, dict) else None
    return float(acc) if acc is not None else None


# ---------------------------------------------------------------- lm_eval tasks

def _flag(command, name, default=None):
    if name in command:
        return command[command.index(name) + 1]
    return default


# One TaskManager per --include_path per process. Its __init__ walks every
# installed task YAML (~2s on this machine, measured with the eval-split
# configs dir) -- the same order of cost as the 2.16s weight reload this module
# exists to get down to, so rebuilding it per call would halve the win.
# (Same caching rationale as eval_splits._TASK_MANAGER.)
_task_managers: dict = {}


def _parse_gen_kwargs(value: str):
    """Mirror lm_eval's CLI, which parses --gen_kwargs with type=try_parse_json
    (lm_eval/__main__.py). The evaluators build this flag as a JSON string
    ('{"temperature": 0.0, "do_sample": false}'). Passing that *string* to
    simple_evaluate() instead routes it through simple_parse_args_string(), whose
    comma/equals split mangles it into {'{"temperature": 0.0': '', ...} and every
    generate_until task then dies with
    TypeError: Unexpected keyword argument '{"temperature": 0.0' at SamplingParams().
    """
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        if "{" in value:
            # The CLI raises here too (ArgumentTypeError) -- the subprocess path would
            # fail on the identical input, so failing loudly keeps behaviour aligned.
            raise ValueError(f"--gen_kwargs is malformed JSON: {value!r}")
        return value                                 # plain k=v,k=v string: pass through


def _run_lm_eval(command, model_dir) -> Optional[float]:
    """command == ["python","-m","lm_eval","--model",backend,"--tasks",task, ...]."""
    global _session
    if _flag(command, "--model") != "vllm":
        return None                                  # HF backend: let the subprocess run it

    import lm_eval
    from lm_eval.models.vllm_causallms import VLLM

    task = _flag(command, "--tasks")
    model_args = _flag(command, "--model_args", "")
    gen_kwargs = _flag(command, "--gen_kwargs")
    batch_size = _flag(command, "--batch_size", "auto")

    if _session is not None and (_session.kind != "lm_eval"
                                 or _session.model_dir != model_dir):
        # Different task family OR different candidate path: the engine can only
        # reload from the path it booted with, so free it and boot fresh.
        shutdown()
    if _session is None:
        print("  [persistent-eval] booting engine (once per run)")
        s = _Session("lm_eval", model_dir)
        # Same additional_config the CLI hands to create_from_arg_string
        # (lm_eval/evaluator.py simple_evaluate: batch_size/max_batch_size/device;
        # None values are dropped by create_from_arg_string itself).
        s.lm = VLLM.create_from_arg_string(model_args, {
            "batch_size": batch_size,
            "max_batch_size": None,
            "device": _flag(command, "--device"),
        })
        s.llm = s.lm.model
        _session = s
        _warm_up(s)
    else:
        dt = _session.reload()
        print(f"  [persistent-eval] reloaded weights in {dt:.2f}s (vs ~26s cold boot)")

    kwargs = {}
    if gen_kwargs:
        kwargs["gen_kwargs"] = _parse_gen_kwargs(gen_kwargs)
    if "--confirm_run_unsafe_code" in command:
        kwargs["confirm_run_unsafe_code"] = True
    if "--apply_chat_template" in command:
        # The evaluator's argv carries this presence-only flag whenever the task
        # is chat-templated (eval_harness's LM_EVAL_CHAT_TEMPLATE_TASKS -- ifeval,
        # every model) OR the per-model registry flag is set
        # (eval_settings["apply_chat_template"], via
        # eval_harness.registry_apply_chat_template). Either way it appears in
        # `command`, so this branch already forwards both cases -- no per-model
        # special-casing needed here. The CLI flag is presence-only (nargs='?',
        # const=True -> True); simple_evaluate takes the same-named kwarg, so
        # persistent and subprocess prompts match.
        kwargs["apply_chat_template"] = True
    limit = _flag(command, "--limit")
    if limit is not None:
        kwargs["limit"] = float(limit)               # CLI parses --limit with type=float
    include_path = _flag(command, "--include_path")  # eval-split task variants
    if include_path is not None:
        # Split runs evaluate the generated <task>_P/<task>_M variants, which
        # live outside the lm_eval package (micr/eval_split_configs). The CLI
        # subprocess sees them via --include_path; in-process we hand
        # simple_evaluate a TaskManager built over the same dir (installed
        # defaults still included, include_defaults=True), so `task` -- already
        # the variant name taken from the evaluator's own --tasks flag --
        # resolves to the identical YAML the subprocess would load.
        if include_path not in _task_managers:
            from lm_eval.tasks import TaskManager
            _task_managers[include_path] = TaskManager(include_path=include_path)
        kwargs["task_manager"] = _task_managers[include_path]
    res = lm_eval.simple_evaluate(model=_session.lm, tasks=[task], batch_size=batch_size, **kwargs)
    if not res or task not in res.get("results", {}):
        return None
    metrics = res["results"][task]
    # prompt_level_strict_acc is ifeval's headline metric (first in its
    # metric_list); keep in sync with eval_harness.parse_score's ifeval branch.
    # The ",<suffix>" in lm_eval metric keys is the FILTER name: ",none" for
    # the stock task, ",strip_think" for the think-strip variants
    # (ifeval_nothink / ifeval_P / ifeval_M -- their filter_list replaces the
    # default filter; see micr/eval_splits._TASK_EXTRA_YAML).
    for key in ("exact_match,flexible-extract", "pass@1,create_test",
                "prompt_level_strict_acc,strip_think",
                "prompt_level_strict_acc,none", "acc,none", "acc_norm,none"):
        if key in metrics:
            return float(metrics[key]) * 100.0
    # single-metric tasks: take the first non-stderr float
    for k, v in metrics.items():
        if not k.endswith("_stderr") and isinstance(v, (int, float)):
            return float(v) * 100.0
    return None


# ---------------------------------------------------------------- entry point

def evaluate(evaluator_cls, env_manager, env_config, eval_settings,
             model_dir: str, task_key: str) -> Optional[float]:
    """
    Score ``model_dir`` on ``task_key`` using a run-persistent engine.

    Returns None whenever the caller should fall back to the legacy subprocess path.
    """
    if not enabled():
        return None
    if _in_spawn_bootstrap():
        # We are a spawn child re-importing the driver's __main__ (see module docstring).
        # Booting here would raise the mp bootstrap RuntimeError and would duplicate the
        # parent's work anyway. Fall back silently; the *parent* session is untouched.
        return None
    if Path(model_dir).name != CANDIDATE_DIRNAME:
        return None                                  # engine reloads from a fixed path only
    if _is_quantized(model_dir):
        _disable("quantized checkpoint: reload_weights skips process_weights_after_loading")
        return None

    try:
        command, cwd, _env = evaluator_cls(env_manager, env_config, dict(eval_settings)) \
            ._construct_command(model_dir, task_key)
    except Exception as e:
        print(f"  [persistent-eval] cannot construct command ({type(e).__name__}); using subprocess")
        return None

    try:
        if len(command) > 1 and str(command[1]).endswith("math_eval.py"):
            score = _run_math(command, cwd, model_dir)
        elif command[:3] == ["python", "-m", "lm_eval"]:
            score = _run_lm_eval(command, model_dir)
        else:
            return None
    except RuntimeError as e:
        if "bootstrapping phase" in str(e):
            # Defense in depth for the spawn-bootstrap state (should be caught above):
            # transient, process-shape-dependent, and the subprocess fallback is always
            # correct -- do not permanently disable the parent's session over it.
            print("  [persistent-eval] engine boot attempted during multiprocessing "
                  "bootstrap; using subprocess for this call")
            return None
        _disable(f"{type(e).__name__}: {e}")
        return None
    except Exception as e:
        _disable(f"{type(e).__name__}: {e}")
        return None

    if score is None:
        return None
    # The legacy path reports through eval_harness.evaluate()'s f"{score:.2f}%" and the
    # caller's float(...) -- i.e. scores are only ever visible at 2-decimal resolution.
    # Round to the same grid so persistent scores are drop-in identical (e.g. lm_eval's
    # pass@1 = 81/164 must come back as 49.39, not 49.390243902439025).
    return round(float(score), 2)


def shutdown() -> None:
    global _session
    if _session is None:
        return
    s, _session = _session, None
    # Ask vLLM to stop the EngineCore explicitly where the API exists, then drop refs.
    for path in ("llm_engine.engine_core.shutdown",):
        try:
            obj = s.llm
            for name in path.split("."):
                obj = getattr(obj, name)
            obj()
        except Exception:
            pass
    s.lm = None
    s.llm = None
    del s
    try:
        import gc
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
