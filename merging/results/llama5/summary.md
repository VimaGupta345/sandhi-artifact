# llama5 results

- run: llama5
- settings: {'perturbation': 'avg', 'groups': 'attn,mlp', 'noise_seed': 1234, 'split_seed': 42, 'profiler_eval_split': 'P', 'micr_eval_split': 'M', 'final_eval_split': 'full', 'baseline_drop_threshold': 5.0, 'drop_tolerance': 2.0, 'batch_size': 'auto', 'temperature': 0.0}
- paper memory target: None%
- files: 19

See sweep.csv for the full cutoff table and pareto.png for the frontier; B.jsonl / C.jsonl are the operating-point merge specs.
