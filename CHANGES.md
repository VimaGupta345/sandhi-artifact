# Changes in response to Artifact Evaluation Review #21A

## Results reconciliation (review §3)

**fig5a.** The M-split scores for this pool depend on the choice of split
(split-selection variability), so the reported operating point is selected
to satisfy the ≤2% accuracy-drop budget on the
**full datasets**: cutoff 94, at which Light-IF is +3.09, T-pro −0.13, and
MedGo 0.00 (`merging/results/full_set_scores.csv`). This point frees
92.6 GB — the paper's absolute figure — reported by the artifact as **48.2%**
(denominator: 192 GB, the sum of full on-disk model sizes; the paper's 49.8%
is the same point against its stated 186 GB total). Documented in
`merging/README.md` § Recorded references, with a signpost README next to
`results/fig5a/report.csv`. The §5.4 cost claim rests on the 92.6 GB freed
and is unaffected. The savings range quoted by the artifact is 26.7%–48.2%.

**fig6a / fig6b.** Confirmed: those `memory_savings.csv` rows used the
earlier, replaced accounting. They are removed; the fig6 pools' memory
numbers now come solely from the pipeline
(`merging/results/FIGURE_COMPOSITIONS.md`, `merging/results/fig6*/report.csv`),
and the recorded serving runs use exactly those specs — e.g. the 2-Qwen spec
frees 0.55 GB and the recorded run correspondingly shows ~1.0×.

## Figure 6 evidence (review §4)

**Recorded results.** `serving/results/` now ships complete recorded runs for
all five deployment scenarios: both arms' server logs, raw benchmark sweeps,
rendered plots, and a summary README. The reference runs (`*_variants`) serve
the materialized merged weights replayed from the recorded MICR journals.
9-model scenario: 2.94× / 2.14× / 1.69× throughput and 1011× / 640× / 294×
P95 TTFT (DeepSeek / Llama / Qwen) vs the paper's 2.93× / 2.11× / 1.72× and
~1000× / ~500× / ~300×.

**Specs are pipeline outputs.** The serving configs now use the Table 2 pool
(`Llama-3.1-8B-UltraMedical`, `Llama-3.1-Hawkish-8B`); both models are in
`merging/models_download/hf_repos.json`. Each shipped spec is the pipeline's
`Cpm` operating point for its pool — group counts match exactly (ds2 21,
qwen2 13, llama5 184, llama_qwen 197, llama_qwen_ds 218), taken from
`merging/results/fig6{a,b,d,e,f}/Cpm.json`.

**Full-set accuracy.** `merging/results/full_set_scores.csv` records the
full-dataset unmerged baseline and merged-variant score, under a matched
protocol per model, for every model that appears in an accuracy figure
(3× Qwen3-32B, 5× Llama, 2× DeepSeek). The Qwen2.5 pair appears in no
standalone accuracy figure; its accuracy enters only through the Fig 5d
composition (M-split footing), and its serving scenarios report
throughput/TTFT — stated in both READMEs. The 32B replay flags in
`run_eval_32b.py` are wired.

**Figures 7–11 and §5.9.** `serving/offloading/` ships the recorded Fig 8/9
runs plus a parameterized launcher. Fig 7 is each pool's analysis output
(`pareto.png`, `sweep.csv`); Fig 10 is covered by `merging/BASELINES.md` and
`merging/plots/data/lora/`; Fig 11 by `merging/micr/run_eval_quantized.py`;
the §5.9 ablations by the recorded layer-level runs under
`merging/clustering/candidates/` and `merging/micr/top_k_experiment.py`.

## Smaller issues (review §5)

- `merging/requirements.txt` documents its lockfile usage
  (`pip install --no-deps`); the `mergekit` and `tinyBenchmarks` lines carry
  correct install instructions; `requirements-native.txt` added.
- Environment-specific eval-split configs are no longer checked in; the
  driver's `prereq` stage regenerates them against the installed lm_eval via
  `TaskManager`'s task index.
- Each scenario config names a distinct spec file, and `server_utils.sh`
  refuses to start sandhi-mode servers unless the spec exists and covers
  every model in the pool.
- kvcached is named and pinned (v0.1.3) in the top-level dependency list and
  `serving/README.md`, with the statement that both arms run with it enabled.
- `merging/README.md` §D rewritten; Figure 6's relationship to `serving/` is
  stated consistently everywhere.
- Exact vLLM versions stated: the serving image ships the SANDHI build
  `0.11.1.dev14+gf0dd2fcb6`; the merging environment pins `vllm==0.11.2`;
  the offloading runs used stock vLLM 0.11.0.
- Plot data: `memory_savings.csv` holds the four Figure 5 rows (the paper's
  reported points); the accuracy tables are regenerated from the artifact
  runs; the operating-point relationship is documented in
  `merging/plots/README.md`. `plots/figures/` contains exactly the four
  panels the script renders.
- The artifact uses the 26.7%–48.2% range throughout (see fig5a above).

## Notes

- **5-Llama serving.** The recorded run exceeds the paper's printed
  improvement (7.1× / 2534× vs 1.1× / 197×); the improvement magnitude
  depends on the configured memory budget, and the recorded configuration is
  fully specified and reproducible as shipped (`serving/results/README.md`).
- **Qwen2.5 pair (Figure 6b).** Under investigation; the pair currently
  carries no accuracy claim and the 2-Qwen serving scenario should be
  skipped. All other scenarios replicate as documented. Noted in
  `merging/README.md` and `serving/README.md`.
- **Figure 6 caption.** The paper's caption misattributes the Llama numbers
  (2.11× / 564×) to Qwen; `serving/README.md` uses §5.3's correct
  per-family assignment, and the caption is on the camera-ready list.
- **Evidence trail.** `merging/evaluation_results/README.md` maps the raw
  lm_eval outputs to `full_set_scores.csv` and documents the exact commands
  ("registry protocol") to reproduce either side of any row.
- **Reference-variant serving.** `MODELS_SANDHI` is documented as a general
  config mechanism; scaling is baked into the replayed weights (no runtime
  scaling needed); `GENERATE_VARIANTS.md` § Rebuilding the recorded
  reference variants gives exact replay parameters for every recorded
  variant set.

## Added

- `serving/paper_plots/` — renders the Figure 6 panels in the paper's style
  from any recorded or fresh results directory, in two commands
  (`parse_bench_logs.py`, then `performance_plots.py`); prints the per-rate
  improvement multipliers quoted in §5.3.
- Top-level README **Dependencies** section: package-manager installation,
  digest-pinned images (no build needed), automated downloads, and gated-model
  instructions with precomputed fallbacks.

## Camera-ready (paper text, not artifact)

- Correct the Fig 5a per-model annotations and the abstract's savings range.
- State in §5.1 that both serving arms run with kvcached, and cite the exact
  vLLM versions above.
- Correct the model name in the Figure 9 description: the recorded offloading
  runs served `fin-llama3.1-8b` (same Llama-3.1-8B architecture; identical
  weight footprint and offload volume) — acknowledged in
  `serving/offloading/README.md`.
