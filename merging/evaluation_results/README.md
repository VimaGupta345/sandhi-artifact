# Raw evaluation outputs (evidence for `results/full_set_scores.csv`)

The JSON files here are lm_eval output files from the full-dataset
evaluations behind `../results/full_set_scores.csv`. Each file's `config`
block identifies exactly what was evaluated: `config.model_args.pretrained`
is the checkpoint path (an unmerged model for baseline rows, a
`variants/...` replay output for merged rows), and `results.<task>` holds the
score the CSV cites.

## File → CSV row map

| CSV row | side | raw output | score |
|---|---|---|---|
| UltraMedical / medqa | baseline | `medqa_4options_fs_…T07-18-11` | 62.07 |
| UltraMedical / medqa | merged (fig6d_Cpm) | `medqa_4options_fs_…T07-18-53` | 61.76 |
| Multi-Truth / truthfulqa_mc2 | baseline | `truthfulqa_mc2_…T08-51-14` | 73.07 |
| SafetyGuard / sst2 | baseline | `sst2_…T08-51-47` | 88.99 |
| Calme-LegalKit / mmlu-prof-law | baseline | `mmlu_professional_law_…T08-52-42` | 49.48 |
| Hawkish / mmlu-econometrics | baseline | `mmlu_econometrics_…T08-53-12` | 55.26 |
| DS-Coder / humaneval | baseline | `humaneval_…T06-19-22` | 69.51 |
| DS-Coder / humaneval | merged (unscaled) | `humaneval_…T06-30-21` | 67.68 |
| DS-Coder / humaneval | merged (scaled) | `humaneval_…T06-35-21` | 67.68 |
| DS-Math / gsm8k-cot | baseline | `../vendor/math-evaluation-harness/output/gsm8k/` (acc 81.3) | 81.30 |
| DS-Math / gsm8k-cot | merged (scaled) | `gsm8k_cot_dsmath_merged_metrics_…T13-29-09.json` (+ per-sample `.jsonl`) | 79.30 |
| Multi-Truth / truthfulqa_mc2 | merged (fig6d_Cpm) | `truthfulqa_mc2_…T13-26-27` | 73.10 |
| SafetyGuard / sst2 | merged (fig6d_Cpm) | `sst2_…T13-24-03` | 90.71 |
| Calme-LegalKit / mmlu-prof-law | merged (fig6d_Cpm) | `mmlu_professional_law_…T13-25-13` | 49.54 |
| Hawkish / mmlu-econometrics | merged (fig6d_Cpm) | `mmlu_econometrics_…T13-27-11` | 53.51 |
| Light-IF-32B / tinyMMLU | baseline | `tinyMMLU_…T13-29-30` | 76.21 |
| MedGo / medqa | baseline | `medqa_4options_…T13-29-05` | 75.96 |
| MedGo / medqa | merged (c94) | `medqa_4options_…T13-25-44` | 75.96 |
| T-pro-it-2.0 / m_mmlu_ru | baseline | `m_mmlu_ru_…T13-32-25` | 76.71 |
| Light-IF-32B / tinyMMLU | merged (c94, replayed) | `tinyMMLU_…T13-51-40` | 79.30 |
| T-pro-it-2.0 / m_mmlu_ru | merged (c94, replayed) | `m_mmlu_ru_…T13-54-25` | 76.58 |

Console logs for the regenerated outputs are in `rebuild_logs/`, produced by
`rebuild_runner.sh` (the same invocations documented below).

The remaining `humaneval_*` files (scores 82.32, 79.88, 77.44) are the
Qwen2.5-Coder evaluations; that pool has no row in `full_set_scores.csv`
(see `../README.md`).

Every row of `full_set_scores.csv` has a shipped raw output. The two c94
merged variants were rebuilt by replaying the recorded `steps.csv` to cutoff
94 and evaluating the result (`rebuild_logs/*_replay.log`); all regenerated
scores match the CSV exactly. Note the vendored math harness writes to a
fixed path (`vendor/math-evaluation-harness/output/`), so re-running gsm8k
overwrites the shipped DS-Math baseline output — restore it with
`git checkout -- merging/vendor/math-evaluation-harness/output/gsm8k/`.

## Reproducing a merged-side score

Replay the model to the recorded cutoff **with evaluation enabled** (drop
`--no_eval`) — parameters per variant set are in
`../GENERATE_VARIANTS.md` § *Rebuilding the recorded reference variants*; the
full-set score lands in the replay's `micr_replay.json` and an lm_eval output
file here.

## Reproducing a baseline-side score ("registry protocol")

"Registry protocol" means: the model's registry task from
`../models_download/hf_repos.json`, evaluated on the **full** dataset,
temperature 0, with the per-model few-shot and `apply_chat_template` settings
from that registry entry's `_optional_fields`. The faithful reproduction path
is through the evaluation harness (`micr/eval_harness.py`), which reads the
registry and applies those settings automatically — including chat templating,
which the harness applies itself rather than via lm_eval's
`--apply_chat_template` flag (so the lm_eval output config here shows
`chat_template: False`; the 1-shot task variant, e.g. `medqa_4options_fs`,
carries the prompting). Running the replay with evaluation enabled (previous
section) at cutoff −1 conceptually corresponds to the baseline; in practice
the shipped baseline files were produced by the same harness invocation used
for the merged side, so baseline and merged scores are always
protocol-matched — verify any pair by comparing the two files' `config`
blocks, which differ only in the checkpoint path.
