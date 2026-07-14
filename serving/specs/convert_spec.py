#!/usr/bin/env python3
"""Convert a merging-pipeline merge spec into the serving-side SHARED_SPEC format.

The merging pipeline emits merge specs in two places:
  - runs/<run>/analysis/<set>/{B,C,Bpm,Cpm,Kpm,P}.json  (analysis operating points;
    these already use HF module names, e.g. self_attn.q_proj — no conversion needed)
  - scripts/build_merge_groups.py output (merged_spec_up_to_cutoff.json etc.;
    these use short names, e.g. attn.q_proj)

The SANDHI serving stack (--shared-layers-spec-path) expects HF module names
(self_attn.*, mlp.*). This script normalizes either input to that convention.

Usage:
    python3 convert_spec.py <input_spec.json> <output_spec.json>
"""
import json
import sys


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    spec = json.load(open(sys.argv[1]))
    out = [
        [
            {
                **entry,
                "component": (
                    "self_" + entry["component"]
                    if entry["component"].startswith("attn.")
                    else entry["component"]
                ),
            }
            for entry in group
        ]
        for group in spec
    ]
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    models = sorted({e["model"] for g in out for e in g})
    print(f"wrote {sys.argv[2]}: {len(out)} shared groups across {len(models)} models")


if __name__ == "__main__":
    main()
