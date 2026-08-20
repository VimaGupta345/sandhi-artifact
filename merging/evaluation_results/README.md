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

The remaining `humaneval_*` files (scores 82.32, 79.88, 77.44) are the
Qwen2.5-Coder evaluations; that pool is under investigation and currently has
no CSV row (see `../README.md`).

## Rows without a shipped raw output

The merged-side outputs for the four Llama models other than UltraMedical
(90.71 / 73.10 / 49.54 / 53.51), the merged DS-Math score (79.30), and both
sides of the three Qwen3-32B rows (the math harness and the 32B evals write
to fixed paths, so later runs replaced these files) are reproducible with the
commands below; the scores are recorded in the CSV. Note the vendored math
harness writes to a fixed path (`vendor/math-evaluation-harness/output/`), so
re-running gsm8k overwrites the shipped DS-Math baseline output — restore it
with `git checkout -- merging/vendor/math-evaluation-harness/output/gsm8k/`.

## Reproducing a merged-side score

Replay the model to the recorded cutoff **with evaluation enabled** (drop
`--no_eval`) — parameters per variant set are in
`../GENERATE_VARIANTS.md` § *Rebuilding the recorded reference variants*; the
full-set score lands in the replay's `micr_replay.json` and an lm_eval output
file here.

## Reproducing a baseline-side score ("registry protocol")

"Registry protocol" means: the model's registry task from
`../models_download/hf_repos.json`, evaluated on the **full** dataset (no
`--limit`), temperature 0, with the per-model `apply_chat_template` /
few-shot settings from that registry entry's `_optional_fields`. Concretely,
via lm_eval against the unmerged checkpoint, e.g. for UltraMedical
(1-shot + chat template per its registry entry):

```bash
lm_eval --model vllm \
  --model_args pretrained=<path-to-unmerged-model>,dtype=bfloat16,gpu_memory_utilization=0.8 \
  --tasks medqa_4options --num_fewshot 1 --apply_chat_template \
  --batch_size auto --output_path merging/evaluation_results/
```

Models whose registry entry sets no chat template (e.g. MedGo, raw 0-shot)
drop `--num_fewshot`/`--apply_chat_template`. The merged side of each pair is
evaluated with the identical invocation, so baseline and merged scores are
always protocol-matched.
