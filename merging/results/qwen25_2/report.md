# Operating points — qwen25_2 pool (Qwen2.5-Coder-7B-Instruct + Qwen2.5-Math-7B-Instruct)

- Source run: `runs/qwen25_2/micr/qwen25_2` (profiling)
- Merge proposals: `runs/qwen25_2/clustering/qwen25_2`
- Memory model: distinct-tensor recipes over ops_step CSVs; sizes/full-model MiB from FAMILY_PROFILES (measured from safetensors); families: qwen2.5-7b
- Pool denominator (FULL on-disk model size, measured from safetensors, 2 models): 30462.4 MB
- **Drop reference = MICR run baseline** (same eval as the merged scores): Coder=87.8, Math=90.7
- Profiling/noise baseline (HARDCODED_BASELINE_SCORES, *different eval* — shown for reference, NOT used for drops): Coder=None, Math=None
- Selection: B = acc drop <= 1.0%; C = acc drop <= 2.0%; Bpm = per-model acc drop <= 1.0%; Cpm = per-model acc drop <= 2.0%; Kpm = per-model knee (savings/accuracy sweet spot, <=2%) (savings is monotonic in cutoff; mem_ge picks the smallest cutoff reaching the target = best accuracy at that savings).
- Single-model (non-merge) groups are excluded from the jsonl: a group with one model is not a merge and saves nothing.
- **Scaled merge groups share the UNSCALED canonical tensor, not the baked bytes**: each member re-applies its per-row scaling factors at load (row_affine_v1; see <point>.scaling_factors.npz next to the jsonl -- a member/group absent from that file is identity/unscaled). The savings figures above already subtract the factor-vector bytes. B: 24 factored slot(s); Bpm: 24 factored slot(s); C: 26 factored slot(s); Cpm: 26 factored slot(s); Kpm: 26 factored slot(s)

| Point | Criterion | Save MB | Save % | Coder score (drop) | Math score (drop) | Mean drop |
|---|---|---|---|---|---|---|
| A | unmerged reference | 0 | 0.0% | 87.80 (+0.00) | 90.70 (+0.00) | +0.00 |
| B | acc drop <= 1.0% | 527 | 1.7% | 87.80 (+0.00) | 90.90 (-0.20) | -0.10 |
| C | acc drop <= 2.0% | 553 | 1.8% | 86.59 (+1.21) | 90.30 (+0.40) | +0.80 |
| Bpm | per-model acc drop <= 1.0% | 527 | 1.7% | 87.80 (+0.00) | 92.00 (-1.30) | -0.65 |
| Cpm | per-model acc drop <= 2.0% | 553 | 1.8% | 86.59 (+1.21) | 92.00 (-1.30) | -0.05 |
| Kpm | per-model knee (savings/accuracy sweet spot, <=2%) | 553 | 1.8% | 86.59 (+1.21) | 92.00 (-1.30) | -0.05 |

## B — exact merged components (12 merge groups, 1.7% savings)
_Each line in B.jsonl is one merged tensor group. `L4(coder+math)` = both at layer 4; `coder:L20+math:L22` = cross-layer._
- `attn.k_proj`: L12(Coder+Math), L5(Coder+Math)
- `attn.o_proj`: L12(Coder+Math), L15(Coder+Math)
- `attn.q_proj`: L5(Coder+Math), L9(Coder+Math)
- `attn.v_proj`: L10(Coder+Math), L12(Coder+Math), L9(Coder+Math)
- `mlp.down_proj`: L19(Coder+Math), L26(Coder+Math)
- `mlp.up_proj`: L6(Coder+Math)

## C — exact merged components (13 merge groups, 1.8% savings)
_Each line in C.jsonl is one merged tensor group. `L4(coder+math)` = both at layer 4; `coder:L20+math:L22` = cross-layer._
- `attn.k_proj`: L12(Coder+Math), L5(Coder+Math)
- `attn.o_proj`: L12(Coder+Math), L15(Coder+Math), L8(Coder+Math)
- `attn.q_proj`: L5(Coder+Math), L9(Coder+Math)
- `attn.v_proj`: L10(Coder+Math), L12(Coder+Math), L9(Coder+Math)
- `mlp.down_proj`: L19(Coder+Math), L26(Coder+Math)
- `mlp.up_proj`: L6(Coder+Math)

## Bpm — exact merged components (12 merge groups, 1.7% savings)
_Each line in Bpm.jsonl is one merged tensor group. `L4(coder+math)` = both at layer 4; `coder:L20+math:L22` = cross-layer._
- `attn.k_proj`: L12(Coder+Math), L5(Coder+Math)
- `attn.o_proj`: L12(Coder+Math), L15(Coder+Math)
- `attn.q_proj`: L5(Coder+Math), L9(Coder+Math)
- `attn.v_proj`: L10(Coder+Math), L12(Coder+Math), L9(Coder+Math)
- `mlp.down_proj`: L19(Coder+Math), L26(Coder+Math)
- `mlp.up_proj`: L6(Coder+Math)

## Cpm — exact merged components (13 merge groups, 1.8% savings)
_Each line in Cpm.jsonl is one merged tensor group. `L4(coder+math)` = both at layer 4; `coder:L20+math:L22` = cross-layer._
- `attn.k_proj`: L12(Coder+Math), L5(Coder+Math)
- `attn.o_proj`: L12(Coder+Math), L15(Coder+Math), L8(Coder+Math)
- `attn.q_proj`: L5(Coder+Math), L9(Coder+Math)
- `attn.v_proj`: L10(Coder+Math), L12(Coder+Math), L9(Coder+Math)
- `mlp.down_proj`: L19(Coder+Math), L26(Coder+Math)
- `mlp.up_proj`: L6(Coder+Math)

## Kpm — exact merged components (13 merge groups, 1.8% savings)
_Each line in Kpm.jsonl is one merged tensor group. `L4(coder+math)` = both at layer 4; `coder:L20+math:L22` = cross-layer._
- `attn.k_proj`: L12(Coder+Math), L5(Coder+Math)
- `attn.o_proj`: L12(Coder+Math), L15(Coder+Math), L8(Coder+Math)
- `attn.q_proj`: L5(Coder+Math), L9(Coder+Math)
- `attn.v_proj`: L10(Coder+Math), L12(Coder+Math), L9(Coder+Math)
- `mlp.down_proj`: L19(Coder+Math), L26(Coder+Math)
- `mlp.up_proj`: L6(Coder+Math)
