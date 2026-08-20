# Raw evaluation outputs (evidence for `results/full_set_scores.csv`)

The JSON files here are lm_eval output files from the full-dataset
evaluations behind `../results/full_set_scores.csv`. Each file's `config`
block identifies exactly what was evaluated: `config.model_args.pretrained`
is the checkpoint path (an unmerged model for baseline rows, a
`variants/...` replay output for merged rows), and `results.<task>` holds the
score the CSV cites. Files are timestamped; where several outputs exist for
one task, the CSV cites the run at the recorded operating point — match via
the checkpoint path in `config`.

Known gaps, disclosed: this directory was brought under version control after
some runs had already completed, so a few earlier outputs were overwritten by
later runs (e.g. the deepseek-math gsm8k baseline) and the tinyMMLU output
behind Light-IF-32B's merged score is not present. The affected scores are
reproducible with the commands below.

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
