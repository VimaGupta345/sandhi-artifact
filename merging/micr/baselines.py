#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Central lookup for profiler-measured baseline scores, keyed by evaluation split.

The gaussian profiler writes one CSV per model to
``<repo>/gaussian_profiler/output/gaussian_<label>.csv`` with columns:

    timestamp,model,task,layer,variant,perturbation,score

and (in newer files) an additional ``split`` column naming the evaluation split
the row was measured on. Older files without a ``split`` column are treated as
measured on the ``"full"`` split.

The unperturbed baseline for a model is the row where ``perturbation == "baseline"``
and ``layer == -1``; its ``score`` is the model's clean accuracy on that split.

Scores measured on different splits are NOT comparable (the profiler evaluates
on its own split, MICR gates on another, and final numbers are on the full
set), so every lookup is split-exact: ``get_baseline`` only returns a score
whose stored split matches the requested split, and ``all_baselines`` keys its
result by ``(label, split)``.

This module is the single source of truth that the merge runners and the
plotting code use to look these numbers up, replacing the per-file
``HARDCODED_BASELINE_SCORES`` dicts that used to drift out of sync.

Interface (other code depends on these names/signatures):
    get_baseline(label, task=None, split=None) -> Optional[float]
    all_baselines() -> dict[tuple[str, str], float]   # keyed (label, split)

Neither function raises: on any error (missing dir, unreadable CSV, malformed
rows) they return ``None`` / omit the entry so callers can fall back to a fresh
measurement. They never fabricate a value.
"""

import csv
import glob
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

# Label key for a file is its filename stem after the ``gaussian_`` prefix,
# which is equivalent to the basename of the CSV's ``model`` column.
_PREFIX = "gaussian_"

# Split recorded for rows that predate the ``split`` column (and the split a
# lookup defaults to when the caller does not specify one).
DEFAULT_SPLIT = "full"


def output_dir() -> str:
    """Directory holding gaussian_<label>.csv files.

    Defaults to ``<repo>/gaussian_profiler/output`` (repo = the merge_tools
    package root, i.e. the parent of this file's ``micr`` directory) and is
    overridable via the ``MICR_GAUSSIAN_OUTPUT_DIR`` environment variable.
    """
    env = os.environ.get("MICR_GAUSSIAN_OUTPUT_DIR")
    if env and env.strip():
        return env.strip()
    repo_root = Path(__file__).resolve().parent.parent  # merge_tools/
    return str(repo_root / "gaussian_profiler" / "output")


def _parse_score(raw: object) -> Optional[float]:
    if raw is None:
        return None
    try:
        return float(str(raw).replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def _normalize_split(raw: object) -> str:
    """Normalize a split value; absent/blank means the full evaluation set."""
    if raw is None:
        return DEFAULT_SPLIT
    text = str(raw).strip()
    return text if text else DEFAULT_SPLIT


def _baseline_rows_from_file(csv_path: str, task: Optional[str] = None) -> Dict[str, float]:
    """Return ``{split: score}`` for every baseline row in ``csv_path``
    (layer == -1, perturbation == "baseline"), optionally filtered by ``task``.

    Rows without a ``split`` column (older profiler output) are recorded under
    ``DEFAULT_SPLIT``. The first readable baseline row per split wins. Returns
    an empty dict on any error; never raises.
    """
    out: Dict[str, float] = {}
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pert = str(row.get("perturbation", "")).strip().lower()
                if pert != "baseline":
                    continue
                layer_raw = str(row.get("layer", "")).strip()
                try:
                    if int(float(layer_raw)) != -1:
                        continue
                except (TypeError, ValueError):
                    continue
                if task is not None and str(row.get("task", "")).strip() != str(task).strip():
                    continue
                score = _parse_score(row.get("score"))
                if score is None:
                    continue
                row_split = _normalize_split(row.get("split"))
                if row_split not in out:
                    out[row_split] = score
        return out
    except (OSError, csv.Error):
        return out


def get_baseline(
    label: str,
    task: Optional[str] = None,
    split: Optional[str] = None,
) -> Optional[float]:
    """Look up the profiler baseline score for ``label`` on ``split``.

    ``label`` keys on the filename stem after ``gaussian_`` (equivalently the
    basename of the CSV's ``model`` column). If ``task`` is given and a file has
    multiple tasks, only baseline rows for that task are considered.

    ``split`` names the evaluation split the caller needs the baseline for
    (``None`` means ``"full"``). Baselines measured on other splits are not
    comparable, so the lookup only returns a score whose stored split (absent
    column = ``"full"``) exactly matches the requested split; otherwise it
    returns ``None`` and the caller should measure a fresh baseline.

    Returns the score as a float, or None if no matching file/row exists.
    Never raises.
    """
    if not label:
        return None
    try:
        csv_path = os.path.join(output_dir(), f"{_PREFIX}{label}.csv")
        if not os.path.isfile(csv_path):
            return None
        wanted = _normalize_split(split)
        return _baseline_rows_from_file(csv_path, task=task).get(wanted)
    except Exception:
        # Defensive: this lookup must never break a caller's control flow.
        return None


def all_baselines() -> Dict[Tuple[str, str], float]:
    """Map ``(label, split)`` for every ``gaussian_<label>.csv`` in the output
    dir to its baseline score on that split. Files without a readable baseline
    row are omitted; rows without a ``split`` column appear under
    ``(label, "full")``. Never raises."""
    out: Dict[Tuple[str, str], float] = {}
    try:
        pattern = os.path.join(output_dir(), f"{_PREFIX}*.csv")
        for csv_path in sorted(glob.glob(pattern)):
            stem = Path(csv_path).stem  # e.g. gaussian_finance_merged
            if not stem.startswith(_PREFIX):
                continue
            label = stem[len(_PREFIX):]
            if not label:
                continue
            for row_split, score in _baseline_rows_from_file(csv_path).items():
                out[(label, row_split)] = score
    except Exception:
        return out
    return out
