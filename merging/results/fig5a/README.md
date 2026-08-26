# Reading this pool's report.csv / sweep.csv

The M-split scores for this pool depend on the choice of split, so the
reported operating point is selected on **full-dataset** accuracy: cutoff 94, at which every
model satisfies the ≤2% drop budget (`../full_set_scores.csv`), freeing
92.6 GB (48.2% under the pipeline's memory model; the paper quotes the same
point as 49.8% of its 186 GB total). The `P` row (cutoff 104) mechanically
targets the paper's percentage under this pipeline's denominator. All sizes
are binary gigabytes (GiB), written "GB" per the paper's convention. See
`../../README.md` § Accuracy footing (M-split vs. full-set).
