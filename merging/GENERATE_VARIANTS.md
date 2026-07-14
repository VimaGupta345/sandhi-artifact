# Generate the merged variant models from a run's profiles / MICR results

Given a completed run under `runs/<run>/`, you can rebuild the actual **merged
model** at any operating point — a full, loadable HF model directory (config +
safetensors + tokenizer). This is the same **replay** path `finaleval` uses:
`micr/run_eval.py` re-applies the accepted merges from the run's `steps.csv` up
to a cutoff and copies the result to `--save_variant_dir`. (`finaleval` deletes
the weights afterward because it only needs the score — so to *keep* the model,
run the replay directly, as below.)

**Operating points** (all emitted by `build_operating_points.py`; see
`report.csv`):

| point | rule | cutoff kind |
|---|---|---|
| `B` / `C` | global ≤1% / ≤2% worst-model drop | **one** cutoff for all models (`cutoff` column) |
| `P` | closest to the paper's reported memory | **one** cutoff (`cutoff` column) |
| `Bpm` / `Cpm` / `Kpm` | **per-model** ≤1% / ≤2% / knee | **each model its own** cutoff (`<model>__cutoff` columns; the `cutoff` column reads `pm`) |

Everything needed is already in the run dir — no re-profiling, no re-MICR:

| input | file |
|---|---|
| merge-candidate ops | `runs/<run>/clustering/<runset>/` |
| accept/reject trajectory | `runs/<run>/micr/<runset>/<model>/steps.csv` |
| model label → weights path | `runs/<run>/label_map.json` |
| global cutoff (`B`/`C`/`P`) | `report.csv` `cutoff` column |
| per-model cutoff (`Bpm`/`Cpm`/`Kpm`) | `report.csv` `<model>__cutoff` column |

`--scaling` must match how the run-set was merged: **off** for `llama5`,
`qwen32b3`; **on** for `deepseek2`, `qwen25_2`, `llama5_deepseek2`. For scaled
run-sets a `scaling_factors.npz` serving-handoff is written next to the model.
`--no_eval` builds the weights only (skips scoring); drop it to also get the
full-set accuracy in `micr_replay.json`. `--domain` is the model's registry task
(required by the CLI; unused under `--no_eval`). Uses the `dock()` wrapper from
`REPRODUCE_DOCKER.md`.

## One model
```bash
dock python micr/run_eval.py \
  --ops_step_csvs_dir runs/<run>/clustering/<runset> \
  --label_map_json    runs/<run>/label_map.json \
  --target_label      <model> \
  --domain            <model-task> \
  --replay_steps_csv  runs/<run>/micr/<runset>/<model>/steps.csv \
  --replay_cutoff     <N>  --replay_cutoff_mode step_idx \
  --scaling           <on|off> \
  --save_variant_dir  variants/<run>/<point>_<model> \
  --no_eval \
  --working_root /tmp/vwork --results_csv /tmp/vsteps.csv --output_dir /tmp/veval --gpu_ids 0
```
→ merged model in `variants/<run>/<point>_<model>/`.

## All models at an operating point (B or C), in one pass
```bash
CODE=/path/to/artifacts_worktree; cd "$CODE"
RUN=llama5; RS=llama5; SET=llama5; POINT=B; SCALING=off   # adjust per run-set

# cutoff for this point, read from report.csv:
CUT=$(python3 -c "import csv;print(next(r['cutoff'] for r in \
  csv.DictReader(open(f'runs/$RUN/analysis/$SET/report.csv')) if r['point']=='$POINT'))")

# model -> registry task (for --domain):
declare -A TASK=(
  [Llama-3.1-8B-UltraMedical]=medqa_4options
  [Llama-3.1-Hawkish-8B]=mmlu_econometrics
  [calme-2.3-legalkit-8b]=mmlu_professional_law
  [Llama-SafetyGuard-Content-Binary]=sst2
  [Llama-3.1-8B-Instruct-multi-truth-judge]=truthfulqa_mc2
  [Qwen2.5-Coder-7B-Instruct]=humaneval  [Qwen2.5-Math-7B-Instruct]=gsm8k-cot
  [deepseek-coder-7b-instruct-v1.5]=humaneval  [deepseek-math-7b-instruct]=gsm8k-cot
  [Light-IF-32B]=tinyMMLU  [T-pro-it-2.0]=m_mmlu_ru  [MedGo]=medqa_4options )

for M in $(ls runs/$RUN/micr/$RS); do
  dock python micr/run_eval.py \
    --ops_step_csvs_dir runs/$RUN/clustering/$RS --label_map_json runs/$RUN/label_map.json \
    --target_label "$M" --domain "${TASK[$M]}" \
    --replay_steps_csv runs/$RUN/micr/$RS/$M/steps.csv \
    --replay_cutoff "$CUT" --replay_cutoff_mode step_idx --scaling $SCALING \
    --save_variant_dir "variants/$RUN/${POINT}_$M" --no_eval \
    --working_root /tmp/vwork_$M --results_csv /tmp/vsteps_$M.csv --output_dir /tmp/veval_$M --gpu_ids 0
done
```
Set `POINT=P` for the paper-closest variant — but only for the **single-run**
figures (fig5a=qwen32b3, fig5b=llama5, fig5c=the joint cross-7), where `P` is one
global cutoff shared by every model *in that run*, so the loop above works
unchanged. Cutoffs are **run-specific**: "cutoff N" indexes each MICR run's own
`steps.csv` greedy order, so a model's cutoff in the 5-llama run ≠ its cutoff in
the cross-7 run. The **composed** pools (fig5d=12, fig6e=9) are NOT single runs
and have no single `P` cutoff — their paper-close variant is the additive
per-model composition: reproduce each atom (`5llama+ds`, `2qwen`, `3qwen32b`) at
its own per-model cutoffs (the `Cpm` recipe below already lands fig5d at 37.7% ≈
paper 38%). See `results/FIGURE_COMPOSITIONS.md`.

For a 32B run-set (`qwen32b3`) use `micr/run_eval.py` directly on a 32B-capable
card (`--gpu_ids 0`); each 62 GB model needs a working copy, so run them **one
at a time** (`/tmp` fills otherwise). Do **not** use `run_eval_32b.py` — its CLI
lacks the replay flags. `run_eval.py` replay now reorders blocks to the recorded
`steps.csv` order, so the 32B plans align (the B1 fix; earlier 32B replays
aborted with "misaligned plan").

## Per-model points (`Bpm` / `Cpm` / `Kpm`), each model at its own cutoff
Same as above but read the cutoff **per model** from its `<model>__cutoff`
column instead of the single `cutoff` column:
```bash
RUN=llama5_deepseek2; RS=llama5_deepseek2; SET=llama5_deepseek2; POINT=Cpm; SCALING=on
for M in $(ls runs/$RUN/micr/$RS); do
  CUT=$(python3 -c "import csv;print(next(r['${M}__cutoff'] for r in \
    csv.DictReader(open(f'runs/$RUN/analysis/$SET/report.csv')) if r['point']=='$POINT'))")
  [ "$CUT" -lt 0 ] && continue   # -1 = this model stays unmerged at this point
  dock python micr/run_eval.py \
    --ops_step_csvs_dir runs/$RUN/clustering/$RS --label_map_json runs/$RUN/label_map.json \
    --target_label "$M" --domain "${TASK[$M]}" \
    --replay_steps_csv runs/$RUN/micr/$RS/$M/steps.csv \
    --replay_cutoff "$CUT" --replay_cutoff_mode step_idx --scaling $SCALING \
    --save_variant_dir "variants/$RUN/${POINT}_$M" --no_eval \
    --working_root /tmp/vwork_$M --results_csv /tmp/vsteps_$M.csv --output_dir /tmp/veval_$M --gpu_ids 0
done
```
The per-model cutoffs decouple a fragile member from a robust one (e.g. in the
cross pool llama→c55 while deepseek-coder→c14), which is what recovers the
savings; see `results/VARIANT_SUGGESTIONS.md` for each set's chosen cutoffs.

## Notes
- **Point A** is the unmerged reference (cutoff −1, no merges) — there is no
  variant to build; it's each base model at its `label_map.json` path.
- The merged models here are byte-identical to what the pipeline scored: replay
  re-applies the exact recorded arithmetic (same `--scaling` the run used).
- Point-level operating specs (`B.json`/`B.jsonl`) list which tensors are shared;
  they're the compact recipe, while this produces the materialized weights.
