"""Auto-imported by Python at interpreter startup when this directory is on
PYTHONPATH (the SANDHI pipeline puts it there for every subprocess).

Purpose: lm_eval 0.4.9's ``get_task_dict`` contains a nested ``pretty_print_task``
that unconditionally does ``yaml_path.relative_to(<lm_eval tasks dir>)`` and raises
``ValueError`` for task YAMLs loaded via ``--include_path`` from outside the package
(our ``micr/eval_split_configs/*``). That ValueError propagates out of
``get_task_dict`` and aborts it before any eval runs — so it must be fixed at the
source line, not merely caught around the call. The line is purely for a log
message, so wrapping it in try/except is safe and side-effect-free.

We recompile ``get_task_dict`` in-memory (no site-packages edit → NO root needed:
the reference Docker image ships lm_eval owned by root and evaluators run the
container as their own UID). Because it is nested, we recompile the whole enclosing
function. This shim reaches the ``python -m lm_eval`` eval subprocesses too: they
inherit PYTHONPATH and auto-import sitecustomize before importing lm_eval.
"""
import sys

# The exact source lines lm_eval 0.4.9 ships; identical replacement to the
# historical file-level patch, applied here in memory instead.
_OLD = ("        lm_eval_tasks_path = Path(__file__).parent\n"
        "        relative_yaml_path = yaml_path.relative_to(lm_eval_tasks_path)")
_NEW = ("        lm_eval_tasks_path = Path(__file__).parent\n"
        "        try:\n"
        "            relative_yaml_path = yaml_path.relative_to(lm_eval_tasks_path)\n"
        "        except ValueError:\n"
        "            relative_yaml_path = yaml_path  # external --include_path yaml; log-only")


def _patch(mod):
    import inspect
    old_fn = getattr(mod, "get_task_dict", None)
    if old_fn is None or getattr(old_fn, "_micr_patched", False):
        return
    try:
        src = inspect.getsource(old_fn)
    except (OSError, TypeError):
        return
    if _OLD not in src:
        return  # lm_eval changed shape; leave it alone rather than guess
    try:
        exec(compile(src.replace(_OLD, _NEW, 1), mod.__file__, "exec"), mod.__dict__)
    except Exception:
        return
    new_fn = mod.get_task_dict
    new_fn._micr_patched = True
    # Rebind any module that already did `from lm_eval.tasks import get_task_dict`.
    for m in list(sys.modules.values()):
        try:
            if getattr(m, "get_task_dict", None) is old_fn:
                m.get_task_dict = new_fn
        except Exception:
            pass


if "lm_eval.tasks" in sys.modules:
    _patch(sys.modules["lm_eval.tasks"])
else:
    import importlib.abc

    class _LmEvalTasksFinder(importlib.abc.MetaPathFinder):
        """Delegate to the real finders for lm_eval.tasks, then wrap its loader so
        _patch runs right after the module executes. No re-import (avoids the
        re-entrancy that a nested import_module inside find_spec would cause)."""

        def find_spec(self, name, path=None, target=None):
            if name != "lm_eval.tasks":
                return None
            after = sys.meta_path[sys.meta_path.index(self) + 1:]
            for finder in after:
                try:
                    spec = finder.find_spec(name, path, target)
                except Exception:
                    spec = None
                if spec is None or spec.loader is None:
                    continue
                real_exec = spec.loader.exec_module

                def exec_module(module, _real=real_exec):
                    _real(module)
                    _patch(module)
                try:
                    spec.loader.exec_module = exec_module
                except Exception:
                    pass
                return spec
            return None

    if not any(isinstance(f, _LmEvalTasksFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _LmEvalTasksFinder())
