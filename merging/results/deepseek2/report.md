# Operating points — deepseek2 pool (deepseek-math-7b-instruct + deepseek-coder-7b-instruct-v1.5)

- Source run: `runs/deepseek2/micr/deepseek2` (profiling)
- Merge proposals: `runs/deepseek2/clustering/deepseek2`
- Memory model: distinct-tensor recipes over ops_step CSVs; sizes/full-model MiB from FAMILY_PROFILES (measured from safetensors); families: deepseek-7b
- Pool denominator (FULL on-disk model size, measured from safetensors, 2 models): 27641.5 MB
- **Drop reference = MICR run baseline** (same eval as the merged scores): math=80.4, coder=74.39
- Profiling/noise baseline (HARDCODED_BASELINE_SCORES, *different eval* — shown for reference, NOT used for drops): math=None, coder=None
- Selection: B = acc drop <= 1.0%; C = acc drop <= 2.0%; Bpm = per-model acc drop <= 1.0%; Cpm = per-model acc drop <= 2.0%; Kpm = per-model knee (savings/accuracy sweet spot, <=2%) (savings is monotonic in cutoff; mem_ge picks the smallest cutoff reaching the target = best accuracy at that savings).
- Single-model (non-merge) groups are excluded from the jsonl: a group with one model is not a merge and saves nothing.
- **Scaled merge groups share the UNSCALED canonical tensor, not the baked bytes**: each member re-applies its per-row scaling factors at load (row_affine_v1; see <point>.scaling_factors.npz next to the jsonl -- a member/group absent from that file is identity/unscaled). The savings figures above already subtract the factor-vector bytes. B: 36 factored slot(s); Bpm: 36 factored slot(s); C: 42 factored slot(s); Cpm: 42 factored slot(s); Kpm: 32 factored slot(s)

| Point | Criterion | Save MB | Save % | math score (drop) | coder score (drop) | Mean drop |
|---|---|---|---|---|---|---|
| A | unmerged reference | 0 | 0.0% | 80.40 (+0.00) | 74.39 (+0.00) | +0.00 |
| B | acc drop <= 1.0% | 1110 | 4.0% | 79.70 (+0.70) | 75.61 (-1.22) | -0.26 |
| C | acc drop <= 2.0% | 1210 | 4.4% | 78.90 (+1.50) | 74.39 (+0.00) | +0.75 |
| Bpm | per-model acc drop <= 1.0% | 1110 | 4.0% | 79.70 (+0.70) | 74.39 (+0.00) | +0.35 |
| Cpm | per-model acc drop <= 2.0% | 1210 | 4.4% | 78.90 (+1.50) | 74.39 (+0.00) | +0.75 |
| Kpm | per-model knee (savings/accuracy sweet spot, <=2%) | 1043 | 3.8% | 80.00 (+0.40) | 74.39 (+0.00) | +0.20 |

## B — exact merged components (18 merge groups, 4.0% savings)
_Each line in B.jsonl is one merged tensor group. `L4(coder+math)` = both at layer 4; `coder:L20+math:L22` = cross-layer._
- `attn.k_proj`: L25(coder+math), L27(coder+math)
- `attn.o_proj`: L20(coder+math), L25(coder+math), L29(coder+math)
- `attn.q_proj`: L20(coder+math)
- `attn.v_proj`: L20(coder+math), L22(coder+math), L25(coder+math)
- `mlp.down_proj`: L14(coder+math), L26(coder+math)
- `mlp.gate_proj`: L22(coder+math), L26(coder+math), L4(coder+math), L8(coder+math), L9(coder+math)
- `mlp.up_proj`: L25(coder+math), L5(coder+math)

## C — exact merged components (21 merge groups, 4.4% savings)
_Each line in C.jsonl is one merged tensor group. `L4(coder+math)` = both at layer 4; `coder:L20+math:L22` = cross-layer._
- `attn.k_proj`: L23(coder+math), L24(coder+math), L25(coder+math), L27(coder+math)
- `attn.o_proj`: L20(coder+math), L25(coder+math), L29(coder+math)
- `attn.q_proj`: L20(coder+math)
- `attn.v_proj`: L20(coder+math), L21(coder+math), L22(coder+math), L25(coder+math)
- `mlp.down_proj`: L14(coder+math), L26(coder+math)
- `mlp.gate_proj`: L22(coder+math), L26(coder+math), L4(coder+math), L8(coder+math), L9(coder+math)
- `mlp.up_proj`: L25(coder+math), L5(coder+math)

## Bpm — exact merged components (18 merge groups, 4.0% savings)
_Each line in Bpm.jsonl is one merged tensor group. `L4(coder+math)` = both at layer 4; `coder:L20+math:L22` = cross-layer._
- `attn.k_proj`: L25(coder+math), L27(coder+math)
- `attn.o_proj`: L20(coder+math), L25(coder+math), L29(coder+math)
- `attn.q_proj`: L20(coder+math)
- `attn.v_proj`: L20(coder+math), L22(coder+math), L25(coder+math)
- `mlp.down_proj`: L14(coder+math), L26(coder+math)
- `mlp.gate_proj`: L22(coder+math), L26(coder+math), L4(coder+math), L8(coder+math), L9(coder+math)
- `mlp.up_proj`: L25(coder+math), L5(coder+math)

## Cpm — exact merged components (21 merge groups, 4.4% savings)
_Each line in Cpm.jsonl is one merged tensor group. `L4(coder+math)` = both at layer 4; `coder:L20+math:L22` = cross-layer._
- `attn.k_proj`: L23(coder+math), L24(coder+math), L25(coder+math), L27(coder+math)
- `attn.o_proj`: L20(coder+math), L25(coder+math), L29(coder+math)
- `attn.q_proj`: L20(coder+math)
- `attn.v_proj`: L20(coder+math), L21(coder+math), L22(coder+math), L25(coder+math)
- `mlp.down_proj`: L14(coder+math), L26(coder+math)
- `mlp.gate_proj`: L22(coder+math), L26(coder+math), L4(coder+math), L8(coder+math), L9(coder+math)
- `mlp.up_proj`: L25(coder+math), L5(coder+math)

## Kpm — exact merged components (16 merge groups, 3.8% savings)
_Each line in Kpm.jsonl is one merged tensor group. `L4(coder+math)` = both at layer 4; `coder:L20+math:L22` = cross-layer._
- `attn.k_proj`: L25(coder+math)
- `attn.o_proj`: L20(coder+math), L25(coder+math)
- `attn.q_proj`: L20(coder+math)
- `attn.v_proj`: L20(coder+math), L22(coder+math), L25(coder+math)
- `mlp.down_proj`: L14(coder+math), L26(coder+math)
- `mlp.gate_proj`: L22(coder+math), L26(coder+math), L4(coder+math), L8(coder+math), L9(coder+math)
- `mlp.up_proj`: L25(coder+math), L5(coder+math)
