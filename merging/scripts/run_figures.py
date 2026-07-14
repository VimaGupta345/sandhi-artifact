#!/usr/bin/env python
"""
End-to-end driver for the paper's Figure 5 AND Figure 6 model sets:
HF download -> eval-split materialization -> gaussian profiler (split P) ->
clustering (5% threshold) -> MICR (split M) -> memory/accuracy analysis
(corrected distinct-tensor / full-model-N) + Pareto plots + operating-point jsonl.

Figure 5 sets (memory/accuracy Pareto):
    fig5b  5 Llama-3.1-8B                 (intra-family)
    fig5a  3 Qwen3-32B                    (intra-family; weights must be downloaded)
    fig5c  fig5b + 2 DeepSeek-7B          (cross-family; DeepSeek reuses the
                                           existing 6_deepseek_standardized run)
    fig5d  fig5b + DeepSeek + 2 Qwen2.5-7B + fig5a   (12 models)

Figure 6 sets (serving-deployment pools): fig6a..fig6e. Each is a composition of
already-profiled run-sets (+ recorded reuse) analyzed as one pool -- NO new
profiling. Our output for them is the Pareto frontier + operating-point specs;
the merged-model/serving side is downstream/out of scope. `--list-sets` prints
each pool's members.

Composed sets (5c/5d, all fig6) run no new merges across families -- cross-family
tensors never share (verified: recipes can't match across architectures) -- so
they are assembled at the analysis stage from their member run-sets. Add several
to one --sets to analyze them all off a single profiling pass (e.g.
--sets fig5b,fig5c,fig6c,fig6d reuses the one llama5 run-set for all four).

Models resolve to ONE local cache (load_registry): $SANDHI_MODELS_DIR if set, else
$HF_HOME/models (default repo-local). No second model source, no /scratch mounts
required -- weights download there on first run.

Settings (locked 2026-07-11): perturbation=avg only; groups=attn,mlp; noise seed
1234; profiler on eval-split P; clustering baseline_drop_threshold=5.0; MICR on
eval-split M with drop_tolerance=2.0; final numbers on the full set; memory =
distinct-tensor freed / sum of full on-disk model sizes.

Usage:
    python scripts/run_figures.py --run-name jul11 --sets fig5b --stages all
    python scripts/run_figures.py --run-name jul11 --sets fig5b,fig5a --stages prereq --dry-run
    python scripts/run_figures.py --run-name jul11 --sets fig5c --stages analysis

Every stage is resumable and idempotent; each writes under runs/<run-name>/.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYBIN = sys.executable
REGISTRY_PATH = REPO / "models_download" / "hf_repos.json"

# Atomic RUN-SETS: each is profiled/clustered/MICR'd exactly once.
RUNSETS = {
    "llama5": ["Llama-3.1-8B-UltraMedical", "Llama-3.1-8B-Instruct-multi-truth-judge",
               "Llama-SafetyGuard-Content-Binary", "calme-2.3-legalkit-8b",
               "Llama-3.1-Hawkish-8B"],
    "qwen25_2": ["Qwen2.5-Coder-7B-Instruct", "Qwen2.5-Math-7B-Instruct"],
    "qwen32b3": ["Light-IF-32B", "T-pro-it-2.0", "MedGo"],
    # DeepSeek pair re-run under current defaults (P/M split + persistent),
    # replacing the recorded 6_deepseek_standardized reuse. Cross-base (coder
    # vs math), so scaling ON.
    "deepseek2": ["deepseek-math-7b-instruct", "deepseek-coder-7b-instruct-v1.5"],
}

# Per-runset scaling policy (user decision): row-affine scaling ON only where
# the members are different pretrains/bases (the qwen2.5-7b coder/math pair;
# the deepseek pair is reused from its recorded --enable-scaling run). OFF for
# same-base finetune sets -- llama5 AND the qwen3-32b trio; for the trio this
# deliberately OVERRIDES run_eval's AUTO_SCALE_FAMILIES "qwen" auto-trigger.
# Unlisted/ad-hoc runsets fall back to "auto" (exact historical behavior).
# Applied to MICR runs AND to finaleval replays, so the replayed arithmetic
# matches what the recorded run executed.
RUNSET_SCALING = {"llama5": "off", "qwen25_2": "on", "qwen32b3": "off",
                  "deepseek2": "on"}

# Joint cross-family run-set: 5 llama + 2 deepseek clustered TOGETHER (std-shift
# aligns the two families) and MICR'd with scaling ON, so shape-compatible
# cross-family tensors (attn_q/attn_o, 4096x4096) can merge and self-select via
# MICR. Profiles are per-model (auto-adopted from results/profiler/), no re-sweep.
RUNSETS["llama5_deepseek2"] = RUNSETS["llama5"] + RUNSETS["deepseek2"]
RUNSET_SCALING["llama5_deepseek2"] = "on"

# LEGACY: the recorded 6_deepseek_standardized reuse. Superseded by the fresh
# `deepseek2` run-set (below) in every deepseek set. Kept for reference / to
# reproduce the recorded-deepseek figures if ever needed.
DEEPSEEK_REUSE = {
    "deepseek-coder-7b-instruct-v1.5": (
        "micr/results/6_deepseek_standardized/deepseek_coder/steps.csv",
        "clustering/candidates/6_deepseek_standardized"),
    "deepseek-math-7b-instruct": (
        "micr/results/6_deepseek_standardized/deepseek_math/steps.csv",
        "clustering/candidates/6_deepseek_standardized"),
}

# Figure sets = compositions of run-sets analyzed as one pool. DeepSeek sets now
# use the FRESH `deepseek2` run-set as a co-located leaf (5 llama + 2 deepseek =
# the 7-model cross-family pool), not the recorded reuse.
SETS = {
    "fig5a": {"runsets": ["qwen32b3"], "reuse": {}, "paper_mem": 49.8},
    "fig5b": {"runsets": ["llama5"], "reuse": {}, "paper_mem": 35.2},
    "fig5c": {"runsets": ["llama5", "deepseek2"], "reuse": {}, "paper_mem": 26.7},
    "fig5d": {"runsets": ["llama5", "qwen25_2", "qwen32b3", "deepseek2"], "reuse": {},
              "paper_mem": 38.0},
    # Figure 6 (serving benchmark) deployment pools. Our pipeline's job ends at
    # the Pareto frontier + operating-point specs for each pool; cutoff points
    # are chosen downstream and the merged-model/serving side is out of scope.
    # No new profiling: every config composes already-profiled run-sets.
    "fig6a": {"runsets": ["deepseek2"], "reuse": {}, "paper_mem": None},
    "fig6b": {"runsets": ["qwen25_2"], "reuse": {}, "paper_mem": None},
    "fig6c": {"runsets": ["llama5", "deepseek2"], "reuse": {}, "paper_mem": None},
    "fig6d": {"runsets": ["llama5"], "reuse": {}, "paper_mem": None},
    "fig6e": {"runsets": ["llama5", "qwen25_2", "deepseek2"], "reuse": {},
              "paper_mem": None},
}

LOCKED = {
    "perturbation": "avg",
    "groups": "attn,mlp",
    "noise_seed": 1234,
    "split_seed": 42,
    "profiler_eval_split": "P",
    "micr_eval_split": "M",
    "final_eval_split": "full",
    "baseline_drop_threshold": 5.0,
    "drop_tolerance": 2.0,
    "batch_size": "auto",
    "temperature": 0.0,
}


def log(msg: str) -> None:
    print(f"[fig5 {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


import re as _re

class _Bar:
    """In-place progress bar fed by the tools' own '[progress] N/M' lines.
    TTY: redraws one line with \r. Non-TTY: prints a plain line at 10% steps."""
    WIDTH = 28

    def __init__(self, context: str):
        self.context = context
        self.tty = sys.stdout.isatty()
        self.last_decile = -1
        self.active = False

    def update(self, cur: int, total: int, tail: str = ""):
        pct = 0 if total == 0 else cur / total
        if self.tty:
            filled = int(self.WIDTH * pct)
            bar = "#" * filled + "-" * (self.WIDTH - filled)
            sys.stdout.write(f"\r  [{bar}] {cur}/{total} {tail[:60]:60s}")
            sys.stdout.flush()
            self.active = True
        else:
            dec = int(pct * 10)
            if dec > self.last_decile:
                self.last_decile = dec
                print(f"  progress {cur}/{total} ({int(pct*100)}%) {tail}", flush=True)

    def clear(self):
        if self.tty and self.active:
            sys.stdout.write("\r" + " " * (self.WIDTH + 76) + "\r")
            sys.stdout.flush()
            self.active = False

    def done(self):
        if self.tty and self.active:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self.active = False


_PROGRESS_RE = _re.compile(r"\[progress\][^0-9]*(\d+)\s*/\s*(\d+)(.*)")


def sh(cmd, dry, log_file=None, env=None, cwd=None, echo=None, echo_prefix=""):
    """Run a stage command. Everything goes to log_file; lines matching the
    `echo` regex are ALSO shown live on the console (prefixed), so the user
    sees baselines/progress/decisions without opening the logs."""
    printable = " ".join(str(c) for c in cmd)
    log(("DRY  " if dry else "RUN  ") + printable + (f"  > {log_file}" if log_file else ""))
    if dry:
        return 0
    full_env = dict(os.environ)
    if env:
        full_env.update({k: str(v) for k, v in env.items()})
    if log_file and echo:
        pat = _re.compile(echo)
        with open(log_file, "a") as f:
            f.write(f"\n===== {datetime.now().isoformat()} =====\n{printable}\n")
            f.flush()
            proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1,
                                    env=full_env, cwd=cwd or str(REPO))
            bar = _Bar(echo_prefix)
            for line in proc.stdout:
                f.write(line)
                m = _PROGRESS_RE.search(line)
                if m:
                    bar.update(int(m.group(1)), int(m.group(2)), m.group(3).strip())
                elif pat.search(line):
                    bar.clear()
                    print(f"    {echo_prefix}{line.rstrip()}", flush=True)
            bar.done()
            f.flush()
            rc = proc.wait()
    elif log_file:
        with open(log_file, "a") as f:
            f.write(f"\n===== {datetime.now().isoformat()} =====\n{printable}\n")
            f.flush()
            rc = subprocess.call([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT,
                                 env=full_env, cwd=cwd or str(REPO))
    else:
        rc = subprocess.call([str(c) for c in cmd], env=full_env, cwd=cwd or str(REPO))
    if rc != 0:
        # Fail fast: a silently-failed stage cascades into garbage downstream
        # (observed: profiler import failure -> clustering on stale profiles).
        raise SystemExit(f"[fig5] command failed (exit {rc}): {printable}"
                         + (f"  [log: {log_file}]" if log_file else ""))
    return rc


def _hw_free_gpus(candidates):
    """Subset of candidate GPU ids that are actually free on the machine
    (memory.used < 8 GiB), so we never dispatch onto another user's job.
    If nvidia-smi is unavailable, trust the candidate list."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        free = set()
        for line in out.strip().splitlines():
            idx, used = [x.strip() for x in line.split(",")]
            if int(used) < 8000:
                free.add(idx)
        return [g for g in candidates if g in free]
    except Exception:
        return list(candidates)


# Set from --offline-evals in main(); read by dispatch().
OFFLINE_EVALS = False

# Set once at the top of main() to this invocation's start time; suffixed onto
# every per-run log filename (read by the stage_* funcs) so a re-run never
# overwrites — nor is mistaken for — a prior run's logs.
RUNSTAMP = ""

# Set from --profiles in main(); read by stage_profiler. Governs what happens
# when a REUSABLE profile CSV already exists: "ask" prompts (interactive TTY
# only; non-interactive falls back to "reuse" so batch/Docker runs never hang),
# "reuse" always reuses, "redo" always re-sweeps. A poisoned/truncated profile
# is always re-swept regardless of this setting.
PROFILES_MODE = "ask"
# GPUs per 32B (runner=32b) job. Default 1 (a 32B eval fits one 140GB card:
# ~62GB resident + ~62GB eval copy + activations ~127GB). Scale to 2 for
# vLLM/generative 32B tasks that reserve memory at boot (e.g. ifeval).
N_GPUS_32B = 1


def dispatch(jobs, gpus, dry, poll=20):
    """Run per-model jobs in parallel across the GPU list; the stage returns
    only when EVERY job finished (barrier). Each job dict: {label, n_gpus,
    make_cmd(gpu_str) -> argv, log}. GPUs are taken from `gpus` only while
    both unclaimed by us AND actually free on the machine; jobs queue until
    enough GPUs free up. Fail-fast: first nonzero exit kills the rest.
    With one GPU or one job, falls back to the sequential sh() path (which
    keeps the live progress bar)."""
    gpus = [g.strip() for g in gpus.split(",") if g.strip()] if isinstance(gpus, str) else list(gpus)
    if dry or len(gpus) <= 1 or len(jobs) <= 1:
        # Sequential fallback must apply the same job env as the parallel
        # branch (a 1-GPU run or single-job resume is the common case, and
        # silently dropping HF_HUB_OFFLINE here re-exposes evals to the
        # shared-cache EACCES the flag exists to avoid).
        # Persistent eval (one resident vLLM engine per job, ~2s weight reload
        # instead of a ~26s cold boot per eval) is pinned ON explicitly for
        # pipeline runs -- both the profiler flag and the MICR flag -- so a
        # job's behavior never depends on whatever the caller's shell exports.
        seq_env = {"MICR_PERSISTENT_EVAL": "1",
                   "GAUSSIAN_PROFILER_PERSISTENT_EVAL": "1",
                   # In-process HF eval for MC/loglikelihood tasks pinned ON
                   # explicitly (belt-and-suspenders with the module default,
                   # which is also "1"): bit-identical to the subprocess path,
                   # so it never moves a number, only skips the checkpoint
                   # re-load. MICR_INPROCESS_HF_EVAL=0 restores the old path.
                   "MICR_INPROCESS_HF_EVAL": "1"}
        if OFFLINE_EVALS:
            seq_env["HF_HUB_OFFLINE"] = "1"
        for j in jobs:
            print(f"  {j['label']}", flush=True)
            sh(j["make_cmd"](",".join(gpus[: j.get("n_gpus", 1)] or gpus)), dry,
               log_file=j["log"], env=seq_env, echo=j.get("echo"))
        return
    import time as _time
    pending = list(jobs)
    running = {}          # Popen -> (job, [gpu ids], log fh)
    claimed = set()
    done = 0
    failed = None
    print(f"  parallel dispatch: {len(jobs)} jobs over GPUs {gpus}", flush=True)
    while pending or running:
        for proc in list(running):
            if proc.poll() is None:
                continue
            job, alloc, fh = running.pop(proc)
            fh.close()
            claimed.difference_update(alloc)
            if proc.returncode != 0:
                failed = (job, proc.returncode)
                pending.clear()
            else:
                done += 1
                print(f"  done ({done}/{len(jobs)}) {job['label']}", flush=True)
        if failed and not running:
            break
        avail = [g for g in _hw_free_gpus([g for g in gpus if g not in claimed])]
        while pending and len(avail) >= pending[0].get("n_gpus", 1):
            job = pending.pop(0)
            alloc = [avail.pop(0) for _ in range(job.get("n_gpus", 1))]
            claimed.update(alloc)
            gpu_str = ",".join(alloc)
            cmd = [str(c) for c in job["make_cmd"](gpu_str)]
            fh = open(job["log"], "a")
            fh.write(f"\n===== {datetime.now().isoformat()} (gpu {gpu_str}) =====\n"
                     + " ".join(cmd) + "\n")
            fh.flush()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_str)
            # --offline-evals (opt-in): on machines with a SHARED read-only
            # HF_HOME, huggingface_hub's online HEAD/404 path intermittently
            # rewrites .no_exist/ marker files owned by other users (EACCES ->
            # failed eval). Offline mode makes eval jobs read the warm cache
            # only. Default is OFF so a fresh clone (artifact evaluation) can
            # still lazily download anything an eval needs.
            if OFFLINE_EVALS:
                env["HF_HUB_OFFLINE"] = "1"
            # Persistent eval pinned ON per job (see the sequential branch's
            # comment); keep the two branches' env handling in lockstep.
            env["MICR_PERSISTENT_EVAL"] = "1"
            env["GAUSSIAN_PROFILER_PERSISTENT_EVAL"] = "1"
            # In-process HF eval pinned ON explicitly (belt-and-suspenders with
            # the module default "1"): bit-identical, pure speed win for
            # MC/loglikelihood tasks. MICR_INPROCESS_HF_EVAL=0 restores old path.
            env["MICR_INPROCESS_HF_EVAL"] = "1"
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                    env=env, cwd=str(REPO))
            running[proc] = (job, alloc, fh)
            print(f"  start {job['label']} @ gpu {gpu_str} "
                  f"({len(running)} running, {len(pending)} queued)", flush=True)
        if pending or running:
            _time.sleep(poll)
    if failed:
        job, rc = failed
        raise SystemExit(f"[fig5] parallel job failed (exit {rc}): {job['label']} "
                         f"[log: {job['log']}]")


def load_registry():
    """Load the model registry and resolve every model to ONE local cache.

    Registry `local_path` values are relative model dir-names. They resolve under
    a single root so there is no second model source and no machine-specific path:

      root = $SANDHI_MODELS_DIR   (set it to reuse an existing download cache)
             else  $HF_HOME/models   (default: repo-local <repo>/hf_cache/models)

    So by default the pipeline is self-contained -- weights download into the
    mounted repo (prereq) and nothing under /scratch needs mounting. Absolute
    `local_path` entries (if any) are honored as-is for backward compatibility.
    """
    reg = json.load(open(REGISTRY_PATH))
    hf_home = os.environ.get("HF_HOME") or str(REPO / "hf_cache")
    root = Path(os.environ.get("SANDHI_MODELS_DIR") or (Path(hf_home) / "models"))
    for _lbl, e in reg.items():
        if isinstance(e, dict) and "local_path" in e:
            p = Path(e["local_path"])
            if not p.is_absolute():
                e["local_path"] = str(root / p)
    return reg


# Short aliases: --sets 5a == fig5a, etc. Atomic run-set names (llama5, qwen25_2,
# qwen32b3) are also accepted directly and run/analyze that run-set alone.
SET_ALIASES = {"5a": "fig5a", "5b": "fig5b", "5c": "fig5c", "5d": "fig5d",
               "a": "fig5a", "b": "fig5b", "c": "fig5c", "d": "fig5d",
               "6a": "fig6a", "6b": "fig6b", "6c": "fig6c", "6d": "fig6d",
               "6e": "fig6e"}


def resolve_set(name: str) -> str:
    name = SET_ALIASES.get(name.strip(), name.strip())
    if name in SETS:
        return name
    if name in RUNSETS:  # synthesize a single-runset figure set on the fly
        SETS[name] = {"runsets": [name], "reuse": {}, "paper_mem": None}
        return name
    raise KeyError(name)


def list_sets() -> str:
    lines = ["figure sets:"]
    for n, cfg in SETS.items():
        if n in RUNSETS:
            continue
        fresh = ", ".join(cfg["runsets"])
        reuse = ", ".join(cfg["reuse"]) if cfg["reuse"] else "-"
        lines.append(f"  {n:7s} (alias {n[-2:]}) runsets=[{fresh}] reuse=[{reuse}] "
                     f"paper_mem={cfg['paper_mem']}%")
    lines.append("atomic run-sets (usable directly as --sets values):")
    for rs, labels in RUNSETS.items():
        lines.append(f"  {rs:9s} {len(labels)} models: {', '.join(labels)}")
    return "\n".join(lines)


def runsets_for(fig_sets):
    """Atomic run-sets needed by the requested figure sets (deduped, ordered)."""
    out = []
    for s in fig_sets:
        for rs in SETS[s]["runsets"]:
            if rs not in out:
                out.append(rs)
    return out


def labels_for(fig_sets):
    return {lbl for rs in runsets_for(fig_sets) for lbl in RUNSETS[rs]}


def has_weights(path: Path) -> bool:
    if not path.is_dir():
        return False
    total = (sum(f.stat().st_size for f in path.glob("*.safetensors"))
             + sum(f.stat().st_size for f in path.glob("*.bin")))
    return total > 1 << 30  # >1 GiB of shards (.safetensors or .bin) = real weights


def measure_family(model_dir: Path):
    """Full on-disk size + per-component MiB from the safetensors index (works on
    metadata-only stubs too, since the index records every tensor's byte range)."""
    import struct
    idx = json.load(open(model_dir / "model.safetensors.index.json"))
    meta_total = idx.get("metadata", {}).get("total_size")
    cfg = json.load(open(model_dir / "config.json"))
    n_layers = cfg["num_hidden_layers"]
    sizes = {}
    weight_map = idx["weight_map"]
    have_shards = all((model_dir / s).exists() for s in set(weight_map.values()))
    if have_shards:
        per_tensor = {}
        for shard in set(weight_map.values()):
            with open(model_dir / shard, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            for k, v in hdr.items():
                if k != "__metadata__":
                    s, e = v["data_offsets"]
                    per_tensor[k] = e - s
        full_mib = sum(per_tensor.values()) / (1 << 20)
        for c, pat in (("q_proj", "self_attn.q_proj"), ("k_proj", "self_attn.k_proj"),
                       ("v_proj", "self_attn.v_proj"), ("o_proj", "self_attn.o_proj"),
                       ("gate_proj", "mlp.gate_proj"), ("up_proj", "mlp.up_proj"),
                       ("down_proj", "mlp.down_proj")):
            key = next((k for k in per_tensor if f"layers.0.{pat}.weight" in k), None)
            sizes[c] = round(per_tensor[key] / (1 << 20), 2) if key else 0.0
    else:
        # stub: derive component sizes from config shapes (bf16 = 2 bytes).
        # head_dim can be explicit and != hidden/num_heads (Qwen3: 128 vs 80),
        # so q/o_proj are (num_heads*head_dim, hidden), not (hidden, hidden).
        h = cfg["hidden_size"]; nh = cfg["num_attention_heads"]
        kvh = cfg.get("num_key_value_heads", nh)
        head = cfg.get("head_dim") or h // nh
        inter = cfg["intermediate_size"]
        mib = lambda a, b: round(a * b * 2 / (1 << 20), 2)
        sizes = {"q_proj": mib(nh * head, h), "k_proj": mib(kvh * head, h),
                 "v_proj": mib(kvh * head, h), "o_proj": mib(h, nh * head),
                 "gate_proj": mib(inter, h), "up_proj": mib(inter, h),
                 "down_proj": mib(h, inter)}
        full_mib = (meta_total or 0) / (1 << 20)
    return {"num_layers": n_layers, "full_model_mib": round(full_mib, 1), "component_mib": sizes}


# ---------------------------------------------------------------- stages

def stage_prereq(run, sets, reg, dry):
    log("== stage: prereq (models, datasets, deps, splits) ==")
    # 1) model weights
    for lbl in sorted(labels_for(sets)):
        e = reg[lbl]
        p = Path(e["local_path"])
        if has_weights(p):
            log(f"  weights OK      {lbl}")
            continue
        if e["hf_repo"].startswith("VERIFY_ME"):
            log(f"  !! {lbl}: weights missing and hf_repo unverified ({e['hf_repo']}) -- "
                f"fill models_download/hf_repos.json before running this set")
            if not dry:
                raise SystemExit(2)
            continue
        sh(["hf", "download", e["hf_repo"], "--local-dir", str(p)], dry,
           log_file=run / "logs" / f"download_{lbl}_{RUNSTAMP}.log")
    # 2) eval datasets + deps (network required once)
    tasks = sorted({reg[m]["task"] for m in labels_for(sets)})
    sh([PYBIN, "-m", "pip", "install", "--quiet", "langdetect", "immutabledict"], dry,
       log_file=run / "logs" / f"deps_{RUNSTAMP}.log")
    probe = ("import os;os.environ['HF_ALLOW_CODE_EVAL']='1';"
             "from lm_eval import tasks as T;tm=T.TaskManager();"
             f"[print(t, len(list((lambda x: x.test_docs() if x.has_test_docs() else x.validation_docs())(tm.load_task_or_group(t)[t])))) for t in {tasks!r} if t!='gsm8k-cot']")
    sh([PYBIN, "-c", probe], dry, log_file=run / "logs" / f"datasets_{RUNSTAMP}.log",
       env={"HF_HOME": os.environ["HF_HOME"]})
    # 3) materialize P/M splits for every task
    for t in tasks:
        sh([PYBIN, str(REPO / "micr" / "eval_splits.py"), "--task", t,
            "--seed", LOCKED["split_seed"]], dry, log_file=run / "logs" / f"splits_{RUNSTAMP}.log")


def _profile_ok_for_reuse(csv_out, entry):
    """(ok, detail) for an existing profile CSV. ok=True -> safe to reuse.

    Two gates, never raises:
    1) Content sanity: a real profile has a positive baseline row and
       non-constant scores. All-0.0 files (fabricated by pre-sentinel failure
       recording) must be re-swept -- row count alone cannot catch them.
    2) Completion: prefer the profiler's own <csv>.complete sentinel (rows may
       legitimately be missing -- eval-failure skips are honest holes, treated
       as not-avgable downstream, so a finished sweep can be short of the full
       cell count). Fall back to the expected-row count for pre-sentinel CSVs.
    """
    import csv as _csv
    try:
        with csv_out.open() as fh:
            rows = list(_csv.DictReader(fh))
        scores = [r.get("score") for r in rows]
        base = next((r.get("score") for r in rows
                     if r.get("perturbation") == "baseline" or r.get("layer") == "-1"),
                    None)
        if base is None or float(base) <= 0:
            return False, "no positive baseline row (poisoned or truncated)"
        if len(set(scores)) < 2:
            return False, "all scores identical (poisoned)"
    except Exception as e:
        return False, f"unreadable CSV ({type(e).__name__}: {e})"
    if (csv_out.parent / (csv_out.name + ".complete")).exists():
        return True, f"completion sentinel, {len(rows)} rows, sane scores"
    try:
        n_layers = json.loads(
            (Path(entry["local_path"]) / "config.json").read_text()
        )["num_hidden_layers"]
    except Exception as e:
        return False, f"cannot verify completeness ({type(e).__name__}: {e})"
    expected = n_layers * len(LOCKED["groups"].split(",")) + 1
    if len(rows) >= expected:
        return True, f"{len(rows)}/{expected} rows, sane scores"
    return False, f"only {len(rows)}/{expected} rows"


def _profile_decision(mode, lbl, ok, detail):
    """Return 'reuse' or 'redo' for an EXISTING profile CSV.

    A non-reusable (poisoned/truncated) profile is always re-swept regardless
    of mode. For a reusable one: 'reuse'/'redo' are explicit; 'ask' prompts the
    user when stdin is a TTY, and falls back to 'reuse' otherwise so that
    batch / Docker / CI runs never block waiting on input.
    """
    if not ok:
        return "redo"
    if mode in ("reuse", "redo"):
        return mode
    if not sys.stdin.isatty():
        log(f"  [profiles=ask · non-interactive] reusing {lbl} ({detail}); "
            f"pass --profiles redo to force a re-sweep")
        return "reuse"
    while True:
        ans = input(f"[profiles] existing profile for '{lbl}' ({detail}) — "
                    f"[r]euse or re[d]o? [r/d] ").strip().lower()
        if ans in ("", "r", "reuse"):
            return "reuse"
        if ans in ("d", "redo"):
            return "redo"
        print("  answer 'r' to reuse or 'd' to redo")


def stage_profiler(run, sets, reg, dry, gpus):
    log("== stage: profiler (split P, avg, seed %s) ==" % LOCKED["noise_seed"])
    out = run / "profiler"; out.mkdir(parents=True, exist_ok=True)
    jobs = []
    labels = sorted(labels_for(sets))
    for i, lbl in enumerate(labels, 1):
        e = reg[lbl]
        csv_out = out / f"gaussian_{lbl}.csv"
        # Auto-detect: adopt a valid profile from the canonical shared store
        # (results/profiler/, populated by `collect`) into this run when it has
        # none -- so evaluators never hand-copy profiles. `--profiles redo` skips
        # this (forces a fresh sweep); reuse/ask then see the adopted CSV below.
        _shared = REPO / "results" / "profiler" / f"gaussian_{lbl}.csv"
        if not csv_out.exists() and _shared.exists() and not dry and PROFILES_MODE != "redo":
            _ok, _ = _profile_ok_for_reuse(_shared, reg[lbl])
            if _ok:
                shutil.copy2(_shared, csv_out)
                if Path(str(_shared) + ".complete").exists():
                    shutil.copy2(str(_shared) + ".complete", str(csv_out) + ".complete")
                log(f"  adopted shared profile   {csv_out.name}  (from results/profiler/)")
        if csv_out.exists() and not dry:
            ok, detail = _profile_ok_for_reuse(csv_out, reg[lbl])
            if _profile_decision(PROFILES_MODE, lbl, ok, detail) == "reuse":
                log(f"  skip (reusable: {detail})   {csv_out.name}")
                continue
            # Re-sweeping (poisoned/truncated, or a user/mode-requested redo):
            # archive the existing CSV first. A live writer means a profiler
            # from a previous launch still has it open -- renaming underneath it
            # would interleave two jobs' rows. Refuse loudly instead.
            probe = subprocess.run(["pgrep", "-f", str(csv_out)],
                                   capture_output=True, text=True)
            if probe.returncode == 0:
                raise SystemExit(
                    f"[fig5] {csv_out.name} is being re-swept but a live "
                    f"process is still writing it (pid(s) {' '.join(probe.stdout.split())}); "
                    f"wait for it to finish or kill it, then relaunch")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            partial = csv_out.with_name(csv_out.name + f".partial.{stamp}")
            csv_out.rename(partial)
            stale_sentinel = csv_out.parent / (csv_out.name + ".complete")
            if stale_sentinel.exists():
                stale_sentinel.unlink()
            why = detail if not ok else f"redo requested (--profiles {PROFILES_MODE})"
            log(f"  re-sweeping {lbl} ({why}) -> archived {partial.name}")
        def _mk(gpu_str, _lbl=lbl, _e=e, _csv=csv_out):
            return [PYBIN, str(REPO / "gaussian_profiler" / "gaussian_profiler.py"),
                    "--model", reg[_lbl]["local_path"], "--tasks", _e["task"],
                    "--output_csv", str(_csv),
                    "--perturbation", LOCKED["perturbation"], "--groups", LOCKED["groups"],
                    "--seed", LOCKED["noise_seed"],
                    # Per-job tmp root: the persistent-eval candidate dir has a
                    # FIXED basename inside it, so concurrent jobs sharing a tmp
                    # root collide on (and can cross-contaminate) the candidate
                    # (observed: FileExistsError racing two mirrors).
                    "--tmp_dir", str(run / "tmp" / _lbl),
                    "--eval_split", LOCKED["profiler_eval_split"], "--gpus", gpu_str]
        jobs.append({"label": f"({i}/{len(labels)}) {lbl} · {e['task']}",
                     "n_gpus": N_GPUS_32B if e.get("runner") == "32b" else 1,
                     "make_cmd": _mk,
                     "log": run / "logs" / f"profiler_{lbl}_{RUNSTAMP}.log",
                     "echo": r"\[progress\]|Baseline Accuracy|\[eval-split\]|\[save\] delta|\[delta-save\]|\[eval-failed\]"})
    dispatch(jobs, gpus, dry)


def stage_clustering(run, sets, reg, dry, gpus="0"):
    log("== stage: clustering (threshold %.1f) ==" % LOCKED["baseline_drop_threshold"])
    # clustering.py loads model weights to CUDA for stats; pin it to the FIRST
    # allocated GPU so it doesn't default to GPU 0 (which may be busy with a
    # concurrent profiler/MICR job -> OOM). Single card is plenty for stats.
    cl_gpu = (gpus.split(",")[0].strip() if isinstance(gpus, str) else str(gpus[0])) or "0"
    for rs in runsets_for(sets):
        out = run / "clustering" / rs
        sh([PYBIN, str(REPO / "clustering" / "clustering.py"),
            "--select-labels", *RUNSETS[rs],
            "--baseline-drop-threshold", LOCKED["baseline_drop_threshold"],
            "--out-dir", str(out)],
           dry, log_file=run / "logs" / f"clustering_{rs}_{RUNSTAMP}.log",
           env={"MICR_NOISE_PROFILE_DIR": str(run / "profiler"),
                "CUDA_VISIBLE_DEVICES": cl_gpu},
           echo=r"candidates|Wrote|\[noise-override\]|rows")


def stage_micr(run, sets, reg, dry, gpus):
    log("== stage: MICR (split M, tol %.1f) ==" % LOCKED["drop_tolerance"])
    label_map = {lbl: e["local_path"] for lbl, e in reg.items()
                 if isinstance(e, dict) and "local_path" in e}
    lm_path = run / "label_map.json"
    if not dry:
        lm_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(label_map, open(lm_path, "w"), indent=2)
    jobs = []
    for rs in runsets_for(sets):
        _rl = RUNSETS[rs]
        for i, lbl in enumerate(_rl, 1):
            e = reg[lbl]
            runner = "run_eval_32b.py" if e["runner"] == "32b" else "run_eval.py"
            res = run / "micr" / rs / lbl
            steps = res / "steps.csv"
            if steps.exists() and not dry:
                log(f"  skip (exists)   {steps}")
                continue
            def _mk(gpu_str, _lbl=lbl, _e=e, _rs=rs, _runner=runner, _res=res, _steps=steps):
                return [PYBIN, str(REPO / "micr" / _runner),
                        "--ops_step_csvs_dir", str(run / "clustering" / _rs),
                        "--label_map_json", str(lm_path),
                        "--target_label", _lbl, "--domain", _e["task"],
                        "--eval_split", LOCKED["micr_eval_split"],
                        "--drop_tolerance", LOCKED["drop_tolerance"],
                        "--scaling", RUNSET_SCALING.get(_rs, "auto"),
                        "--working_root", str(run / "work" / _lbl),
                        "--results_csv", str(_steps),
                        "--output_dir", str(_res / "evals"),
                        "--gpu_ids", gpu_str]
            jobs.append({"label": f"({i}/{len(_rl)}) {lbl} · {e['task']} · split M",
                         "n_gpus": N_GPUS_32B if e.get("runner") == "32b" else 1,
                         "make_cmd": _mk,
                         "log": run / "logs" / f"micr_{rs}_{lbl}_{RUNSTAMP}.log",
                         "echo": r"\[progress\]|\[baseline\]|\[final\]|\[eval-split\]|\[scale\]"})
    dispatch(jobs, gpus, dry)


def stage_analysis(run, sets, reg, dry):
    log("== stage: analysis (corrected memory, Pareto, operating-point jsonl) ==")
    for s in sets:
        cfg = {"_family_profiles": {}}
        leaf_names = []
        # one leaf pool per (runset, family) in the native bop schema
        for rs in SETS[s]["runsets"]:
            fams = {}
            for lbl in RUNSETS[rs]:
                fams.setdefault(reg[lbl]["family"], []).append(lbl)
            for fam, labels in fams.items():
                name = f"{s}__{rs}__{fam}"
                cfg[name] = {
                    "models": {lbl: lbl for lbl in labels},
                    "run_dir": str(run / "micr" / rs),
                    "ops_dir": str(run / "clustering" / rs),
                    "memory_profile": fam,
                    "points": [],
                    "out_dir": str(run / "analysis" / s / name),
                }
                leaf_names.append(name)
                if fam not in ("llama-8b", "deepseek-7b", "qwen2.5-7b"):
                    prof = measure_family(Path(reg[labels[0]]["local_path"]))
                    cfg["_family_profiles"][fam] = {
                        "components": prof["component_mib"],
                        "full_model_mib": prof["full_model_mib"],
                        "num_layers": prof["num_layers"],
                    }
        compose = list(leaf_names)
        if SETS[s]["reuse"]:
            compose.append("deepseek")   # built-in pool = the recorded deepseek run
        cfg[s] = {
            "compose": compose,
            "points": [
                {"name": "B", "kind": "acc", "budget": 1.0},
                {"name": "C", "kind": "acc", "budget": 2.0},
            ],
            "out_dir": str(run / "analysis" / s),
        } if (len(compose) > 1 or not leaf_names) else None
        # (a reuse-only set, e.g. fig6a, has no leaf pools: analyze it as a
        # compose-of-one over the built-in recorded pool)
        if cfg[s] is None:                       # single leaf: analyze it directly
            cfg[s] = dict(cfg[leaf_names[0]])
            cfg[s]["points"] = [
                {"name": "B", "kind": "acc", "budget": 1.0},
                {"name": "C", "kind": "acc", "budget": 2.0},
            ]
            cfg[s]["out_dir"] = str(run / "analysis" / s)
        if not cfg["_family_profiles"]:
            cfg.pop("_family_profiles")
        cfg_path = run / "analysis" / f"pool_{s}.json"
        if not dry:
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            json.dump(cfg, open(cfg_path, "w"), indent=2)
        log(f"  pool config -> {cfg_path}")
        # Per-model-cutoff points (Bpm=<=1%, Cpm=<=2%, Kpm=knee) alongside the
        # global B/C: each model at its own deepest in-budget cutoff, so a
        # fragile member (e.g. deepseek in a llama+ds pool) no longer caps a
        # robust one. Additive -- global B/C are unchanged. Plus a paper-closest
        # point P where the figure has a reported memory number.
        analysis_cmd = [PYBIN, str(REPO / "scripts" / "build_operating_points.py"),
            "--pool-config", str(cfg_path), "--pool", s, "--sweep", "--per-model",
            # frontier plotted on the MEAN per-model drop (user choice); the
            # worst-drop view stays available in sweep.csv, and B/C operating
            # points are still selected by worst-drop budgets.
            "--accuracy-metric", "mean",
            "--plot-out", str(run / "analysis" / s / "pareto.png"),
            "--sweep-csv", str(run / "analysis" / s / "sweep.csv")]
        if SETS[s].get("paper_mem") is not None:
            analysis_cmd += ["--paper-mem", str(SETS[s]["paper_mem"])]
        sh(analysis_cmd,
           dry, log_file=run / "logs" / f"analysis_{s}_{RUNSTAMP}.log",
           echo=r"validate|frontier|pareto|Wrote|sweep")


def stage_finaleval(run, sets, reg, dry, gpus):
    """Full-dataset scores for the auto-selected operating points (B, C ...).

    Convention: profiler scores on split P, MICR on split M, FINAL model on the
    full dataset. For every point in analysis/<set>/report.csv with a real
    cutoff, rebuild each member's merged variant at that cutoff via replay
    (no eval in the loop) and let replay's final full-set eval record
    final_full_score into finaleval/<set>/<point>_<label>/micr_replay.json.
    The bulky variant weights are deleted afterwards -- only the score and its
    metadata are the deliverable here; variants for the serving handoff are
    rebuilt at the user's hard-coded cutoffs later.
    """
    log("== stage: finaleval (replay operating points, full-dataset eval) ==")
    label_map = {lbl: e["local_path"] for lbl, e in reg.items()
                 if isinstance(e, dict) and "local_path" in e}
    lm_path = run / "label_map.json"
    if not dry:
        json.dump(label_map, open(lm_path, "w"), indent=2)
    import csv as _csv
    jobs = []
    for s in sets:
        rep = run / "analysis" / s / "report.csv"
        if not rep.exists():
            log(f"  skip {s}: no analysis report yet")
            continue
        def _global_cutoff(r):
            # per-model points (Bpm/Cpm) carry cutoff="pm" -> not a global int;
            # they replay per-model and are handled separately (skipped here for
            # now). Global B/C keep an int cutoff.
            try:
                return int(r["cutoff"]) >= 0
            except (ValueError, TypeError):
                return False
        points = [r for r in _csv.DictReader(open(rep))
                  if r.get("point") not in (None, "", "A") and _global_cutoff(r)]
        if not points:
            log(f"  skip {s}: no mergeable operating points")
            continue
        for rs in SETS[s]["runsets"]:
            for lbl in RUNSETS[rs]:
                e = reg[lbl]
                steps = run / "micr" / rs / lbl / "steps.csv"
                if not steps.exists():
                    log(f"  skip {lbl}: no steps.csv")
                    continue
                runner = "run_eval_32b.py" if e["runner"] == "32b" else "run_eval.py"
                for r in points:
                    pt, cutoff = r["point"], int(r["cutoff"])
                    outd = run / "finaleval" / s / f"{pt}_{lbl}"
                    if (outd / "micr_replay.json").exists() and not dry:
                        log(f"  skip (exists)   {pt}_{lbl}")
                        continue
                    def _mk(gpu_str, _e=e, _lbl=lbl, _rs=rs, _steps=steps,
                            _cut=cutoff, _outd=outd, _runner=runner, _pt=pt):
                        return [PYBIN, str(REPO / "micr" / _runner),
                                "--ops_step_csvs_dir", str(run / "clustering" / _rs),
                                "--label_map_json", str(lm_path),
                                "--target_label", _lbl, "--domain", _e["task"],
                                "--replay_steps_csv", str(_steps),
                                "--replay_cutoff", _cut,
                                "--replay_cutoff_mode", "step_idx",
                                # Replay must re-apply the same per-op arithmetic
                                # the recorded run executed: a runset recorded
                                # with scaling on replays with it forced on (and
                                # emits the scaling_factors.npz handoff next to
                                # micr_replay.json); off/auto sets replay
                                # unscaled, exactly as recorded. OLD recorded
                                # runs replayed via --enable-scaling do not come
                                # through this builder and are unaffected.
                                "--scaling", RUNSET_SCALING.get(_rs, "auto"),
                                "--save_variant_dir", str(_outd),
                                "--working_root", str(run / "work_replay" / f"{_pt}_{_lbl}"),
                                "--results_csv", str(_outd / "replay_steps.csv"),
                                "--output_dir", str(_outd / "evals"),
                                "--gpu_ids", gpu_str]
                    jobs.append({"label": f"{pt}@{cutoff} {lbl} · {e['task']} · full set",
                                 "n_gpus": N_GPUS_32B if e.get("runner") == "32b" else 1,
                                 "make_cmd": _mk,
                                 "log": run / "logs" / f"finaleval_{s}_{pt}_{lbl}_{RUNSTAMP}.log",
                                 "echo": r"\[final\]|\[replay\]|score"})
    if jobs:
        dispatch(jobs, gpus, dry)
    if not dry:
        # keep micr_replay.json/config, drop the weights (10+ GiB per variant)
        for f in (run / "finaleval").rglob("*.safetensors"):
            f.unlink()


def stage_collect(run, sets, reg, dry):
    """Copy the reportable outputs of a run into the repo-level results/ folder
    (the artifact's common results home): shared per-model profiler CSVs
    (results/profiler/, set-independent), per-set step CSVs,
    sweep table, Pareto plot, operating-point jsonls, and a summary.md."""
    log("== stage: collect (populate results/) ==")
    res_root = REPO / "results"
    for s in sets:
        dest = res_root / s
        if not dry:
            dest.mkdir(parents=True, exist_ok=True)
        copied = []
        # analysis outputs (sweep, pareto, operating points, pool config)
        adir = run / "analysis" / s
        for pat in ("sweep.csv", "pareto*.png", "*.jsonl", "*.json", "report.*"):
            for f in sorted(adir.glob(pat)) if adir.exists() else []:
                if not dry:
                    shutil.copy2(f, dest / f.name)
                copied.append(f.name)
        pc = run / "analysis" / f"pool_{s}.json"
        if pc.exists():
            if not dry:
                shutil.copy2(pc, dest / pc.name)
            copied.append(pc.name)
        # full-dataset final scores of the operating points (finaleval stage)
        fdir = run / "finaleval" / s
        if fdir.exists():
            frows = []
            for mj in sorted(fdir.glob("*/micr_replay.json")):
                try:
                    d = json.loads(mj.read_text())
                except Exception:
                    continue
                pt, _, lbl = mj.parent.name.partition("_")
                frows.append((pt, lbl,
                              d.get("cutoff_step_idx",
                                    d.get("replay_cutoff", d.get("cutoff", ""))),
                              d.get("final_full_score")))
            if frows and not dry:
                with open(dest / "final_full_scores.csv", "w") as fh:
                    fh.write("point,label,cutoff,final_full_score\n")
                    for pt, lbl, cut, sc in frows:
                        fh.write(f"{pt},{lbl},{cut},{sc}\n")
                copied.append("final_full_scores.csv")
        # per-model step CSVs + profiler CSVs for the set's fresh run-sets
        for rs in SETS[s]["runsets"]:
            for lbl in RUNSETS[rs]:
                steps = run / "micr" / rs / lbl / "steps.csv"
                if steps.exists():
                    if not dry:
                        (dest / "steps").mkdir(exist_ok=True)
                        shutil.copy2(steps, dest / "steps" / f"{lbl}_steps.csv")
                    copied.append(f"steps/{lbl}_steps.csv")
                # Profiles are per-MODEL and set-independent, so they live in a
                # single SHARED results/profiler/ store -- not duplicated inside
                # every per-set folder (a model in fig5b+5c+5d would otherwise be
                # copied three times). Carry the .complete sentinel with it.
                prof = run / "profiler" / f"gaussian_{lbl}.csv"
                if prof.exists():
                    if not dry:
                        shared_prof = res_root / "profiler"
                        shared_prof.mkdir(exist_ok=True)
                        shutil.copy2(prof, shared_prof / prof.name)
                        sentinel = prof.with_name(prof.name + ".complete")
                        if sentinel.exists():
                            shutil.copy2(sentinel, shared_prof / sentinel.name)
                    copied.append(f"../profiler/{prof.name}")
        # summary
        if not dry:
            lines = [f"# {s} results", "",
                     f"- run: {run.name}", f"- settings: {LOCKED}",
                     f"- paper memory target: {SETS[s]['paper_mem']}%",
                     f"- files: {len(copied)}", ""]
            sweep = dest / "sweep.csv"
            if sweep.exists():
                lines.append("See sweep.csv for the full cutoff table and pareto.png "
                             "for the frontier; B.jsonl / C.jsonl are the operating-point "
                             "merge specs.")
            (dest / "summary.md").write_text("\n".join(lines) + "\n")
        log(f"  {s}: {len(copied)} files -> {dest}")
        _print_result_block(run, s, dest)


def _print_result_block(run, s, dest):
    """Console summary rendered from the collected artifacts. Numbers from the
    MICR trajectory are M-split scores; memory comes from the analysis report."""
    bar = "-" * 68
    lines = [bar, f" RESULT {s}"]
    rep = dest / "report.csv"
    if rep.exists():
        try:
            import csv as _csv
            for row in _csv.DictReader(open(rep)):
                nm = row.get("point") or row.get("name", "?")
                sv = row.get("savings_pct") or row.get("savings", "")
                lines.append(f"   point {nm}: memory {float(sv):.1f}%"
                             + (f" · cutoff {row['cutoff']}" if row.get("cutoff") else ""))
        except Exception as e:
            lines.append(f"   (report.csv unreadable: {e})")
    sw = dest / "sweep.csv"
    if sw.exists():
        try:
            n = sum(1 for _ in open(sw)) - 1
            lines.append(f"   sweep: {n} cutoffs · frontier plot: pareto.png")
        except Exception:
            pass
    ffs = dest / "final_full_scores.csv"
    if ffs.exists():
        try:
            import csv as _csv
            for row in _csv.DictReader(open(ffs)):
                sc = row.get("final_full_score") or "?"
                lines.append(f"   point {row['point']} · {row['label']}: "
                             f"{sc} [full set]")
        except Exception:
            pass
    steps_dir = dest / "steps"
    if steps_dir.exists():
        try:
            import csv as _csv
            for f in sorted(steps_dir.glob("*_steps.csv")):
                rows = list(_csv.DictReader(open(f)))
                base = next((float(r["score"]) for r in rows if r["decision"].strip() == "baseline"), None)
                acc = [float(r["score"]) for r in rows if r["decision"].strip() == "accepted"]
                if base is not None and acc:
                    best = max(acc)
                    lines.append(f"   {f.name.replace('_steps.csv',''):42s} "
                                 f"baseline {base:.2f} · best accepted {best:.2f} "
                                 f"({best-base:+.2f}) [M split]")
        except Exception:
            pass
    lines.append(f" artifacts: {dest}")
    lines.append(bar)
    print("\n".join(lines), flush=True)


STAGES = {"prereq": stage_prereq, "profiler": stage_profiler,
          "clustering": stage_clustering, "micr": stage_micr,
          "analysis": stage_analysis, "finaleval": stage_finaleval,
          "collect": stage_collect}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--registry", default=None,
                    help="Override the model registry JSON (default: "
                         "models_download/hf_repos.json). Point at a variant "
                         "registry to swap a model's benchmark for a run, e.g. "
                         "UltraMedical medqa_4options -> mmlu_professional_medicine.")
    ap.add_argument("--sets", default="fig5b",
                    help="Comma-separated figure sets (fig5a..fig5d, aliases 5a..5d) "
                         "or atomic run-sets (llama5, qwen25_2, qwen32b3).")
    ap.add_argument("--list-sets", action="store_true",
                    help="Print available sets and their members, then exit.")
    ap.add_argument("--stages", default="all",
                    help="Comma-separated subset of: prereq,profiler,clustering,micr,analysis (or 'all')")
    ap.add_argument("--gpus", default="0",
                    help="Candidate GPUs for eval stages: a comma list (e.g. 0,1,3) or "
                         "'auto' = every detected GPU. Busy cards (>=8 GiB used) are culled "
                         "at dispatch time; use --exclude-gpus to always reserve specific cards.")
    ap.add_argument("--exclude-gpus", default="",
                    help="Comma list of GPU ids to never dispatch onto (e.g. a card reserved "
                         "for another user). Applied to both an explicit --gpus and --gpus auto.")
    ap.add_argument("--n-gpus-32b", type=int, default=1,
                    help="GPUs per 32B (runner=32b) profiler/MICR job. Default 1 — a 32B "
                         "eval fits one 140GB card (~62GB resident + ~62GB eval copy + "
                         "activations ~127GB), so e.g. fig5a's 3 models run 1:1 on 3 GPUs. "
                         "Scale to 2 for vLLM/generative 32B tasks that reserve memory at "
                         "boot (e.g. ifeval).")
    ap.add_argument("--hf-home", default=None,
                    help="HF cache dir. Precedence: this flag > HF_HOME env > <repo>/hf_cache. "
                         "On this cluster: /scratch/shared_dir/hf_cache (shared, has all models).")
    ap.add_argument("--dry-run", action="store_true", help="Print every command without executing")
    ap.add_argument("--profiles", choices=["ask", "reuse", "redo"], default="ask",
                    help="When a reusable profile CSV already exists: 'ask' prompts "
                         "per model (interactive only — non-interactive/Docker falls "
                         "back to 'reuse' so it never hangs), 'reuse' always reuses, "
                         "'redo' always re-sweeps. A poisoned/truncated profile is "
                         "re-swept regardless. Default: ask.")
    ap.add_argument("--offline-evals", action="store_true",
                    help="Run profiler/MICR eval jobs with HF_HUB_OFFLINE=1 (cache "
                         "reads only). Use on machines whose HF_HOME is a shared "
                         "cache not writable by you; requires the prereq stage to "
                         "have materialized all models/datasets first. Default: "
                         "online, so evals can download anything they need.")
    args = ap.parse_args()

    # Resolve --gpus: 'auto' -> every detected GPU; then drop --exclude-gpus. The
    # dispatcher's <8 GiB free-filter still culls busy cards from what remains, so
    # 'auto' safely coexists with other users' jobs and with our own concurrent runs.
    def _detected_gpus():
        try:
            _o = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=20).stdout
            return [l.strip() for l in _o.splitlines() if l.strip()]
        except Exception:
            return ["0"]
    _excl = {g.strip() for g in args.exclude_gpus.split(",") if g.strip()}
    _auto = args.gpus.strip().lower() == "auto"
    _cand = _detected_gpus() if _auto else [g.strip() for g in args.gpus.split(",") if g.strip()]
    args.gpus = ",".join(g for g in _cand if g not in _excl) or "0"
    log(f"GPUs: {args.gpus}" + ("  (auto)" if _auto else "")
        + (f"  excluding {sorted(_excl)}" if _excl else ""))

    global OFFLINE_EVALS, PROFILES_MODE, REGISTRY_PATH, N_GPUS_32B
    OFFLINE_EVALS = args.offline_evals
    PROFILES_MODE = args.profiles
    N_GPUS_32B = args.n_gpus_32b
    if args.registry:
        REGISTRY_PATH = Path(args.registry)

    if args.list_sets:
        print(list_sets())
        return
    try:
        sets = [resolve_set(x) for x in args.sets.split(",") if x.strip()]
    except KeyError as e:
        ap.error(f"unknown set {e}; choose from {list(SETS) + list(RUNSETS)} "
                 f"or aliases {list(SET_ALIASES)}")
    stages = list(STAGES) if args.stages == "all" else \
        [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in stages:
        if s not in STAGES:
            ap.error(f"unknown stage {s}; choose from {list(STAGES)}")

    # HF cache precedence: --hf-home > HF_HOME env > <repo>/hf_cache (self-contained
    # default so a fresh clone needs no machine-specific paths). On this cluster pass
    # HF_HOME=/scratch/shared_dir/hf_cache to reuse the shared team cache, which
    # already holds all 12 models and most datasets.
    # The reference image bakes HF_HOME=/cache/huggingface, which is root-owned
    # AND ephemeral under --rm (anything downloaded there vanishes when the
    # container exits). So we IGNORE that baked default and always use the
    # persistent, writable repo-local <repo>/hf_cache -- identical behavior as
    # root or non-root, fully self-contained, no /scratch mount. A DELIBERATE
    # HF_HOME (e.g. a shared cluster cache) or --hf-home is still honored.
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    elif os.environ.get("HF_HOME", "").rstrip("/") in ("", "/cache", "/cache/huggingface"):
        os.environ["HF_HOME"] = str(REPO / "hf_cache")

    def _writable_dir(d):
        try:
            Path(d).mkdir(parents=True, exist_ok=True)
            _t = Path(d) / ".w_probe"
            _t.touch(); _t.unlink()
            return True
        except (PermissionError, OSError):
            return False
    # Safety net: if a DELIBERATE HF_HOME turns out unwritable, still fall back to
    # the repo-local cache rather than crash.
    if not _writable_dir(os.environ["HF_HOME"]):
        os.environ["HF_HOME"] = str(REPO / "hf_cache")
        Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    # Datasets need WRITABLE lock files; a shared HF_HOME can carry lock files
    # owned by other users (observed: EACCES on cais___mmlu locks), which kills
    # datasets.load_dataset even though the hub/ (model) side reads fine. So
    # datasets default to a user-writable cache unless explicitly overridden.
    os.environ.setdefault("HF_DATASETS_CACHE", str(REPO / "hf_cache" / "datasets"))
    Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)
    # Same story for dynamic modules (evaluate's code_eval etc. writes here).
    os.environ.setdefault("HF_MODULES_CACHE", str(REPO / "hf_cache" / "modules"))
    Path(os.environ["HF_MODULES_CACHE"]).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    # lm_eval 0.4.9's get_task_dict crashes on external --include_path task yamls
    # (our split configs). micr/_lmeval_shim/sitecustomize.py recompiles the one
    # offending line in-process; putting its dir on PYTHONPATH makes every child
    # AND every `python -m lm_eval` eval subprocess auto-apply it. Done this way
    # (not by editing site-packages) so NOTHING needs root -- the reference Docker
    # image ships lm_eval root-owned and evaluators run the container as their own
    # UID. sh()/subprocess launches all merge os.environ, so this propagates.
    _shim = str(REPO / "micr" / "_lmeval_shim")
    os.environ["PYTHONPATH"] = _shim + (os.pathsep + os.environ["PYTHONPATH"]
                                        if os.environ.get("PYTHONPATH") else "")
    log(f"HF_HOME: {os.environ['HF_HOME']}")
    log(f"HF_DATASETS_CACHE: {os.environ['HF_DATASETS_CACHE']}")
    reg = load_registry()
    run = REPO / "runs" / args.run_name
    (run / "logs").mkdir(parents=True, exist_ok=True)
    global RUNSTAMP
    RUNSTAMP = datetime.now().strftime("%Y%m%dT%H%M%S")
    log(f"log stamp {RUNSTAMP} — suffixed onto every per-run log filename")

    run_config = {
        "run_name": args.run_name, "created": datetime.now(timezone.utc).isoformat(),
        "sets": sets, "stages": stages, "settings": LOCKED,
        "registry": str(REGISTRY_PATH),
        "note": "memory = distinct-tensor freed / sum(full on-disk N); "
                "profiler baselines measured on P, MICR baselines on M, final on full",
    }
    cfg_path = run / "run_config.json"
    if not args.dry_run:
        existing = json.load(open(cfg_path)) if cfg_path.exists() else {}
        existing.update(run_config)
        json.dump(existing, open(cfg_path, "w"), indent=2)
    bar = "=" * 68
    names = {"fig5a": "3x Qwen3-32B", "fig5b": "5x Llama-3.1-8B",
             "fig5c": "5x Llama + 2x DeepSeek", "fig5d": "12 models"}
    print(bar)
    print(f" SANDHI pipeline · {', '.join(f'{x} ({names.get(x, x)})' for x in sets)} · run {args.run_name}")
    print(f" {LOCKED['perturbation']}-only · noise seed {LOCKED['noise_seed']} · splits P/M "
          f"(seed {LOCKED['split_seed']}) · threshold {LOCKED['baseline_drop_threshold']}% · "
          f"tol {LOCKED['drop_tolerance']}")
    print(f" HF_HOME={os.environ['HF_HOME']} · GPUs {args.gpus} · run dir {run}")
    print(bar, flush=True)

    for i, st in enumerate(stages, 1):
        print(f"\n[{i}/{len(stages)} {st}]", flush=True)
        fn = STAGES[st]
        if st in ("profiler", "micr", "finaleval", "clustering"):
            fn(run, sets, reg, args.dry_run, args.gpus)
        else:
            fn(run, sets, reg, args.dry_run)
    log("done")


if __name__ == "__main__":
    main()
