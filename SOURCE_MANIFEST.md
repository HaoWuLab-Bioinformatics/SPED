# Source manifest

This file records how the clean publication package maps to the original working
project at the time of organization.

| Publication file | Original source | Role |
|---|---|---|
| `make_all_results.py` | New publication entry point | Runs and records the complete result pipeline |
| `src/sped/model.py` | `ppline/interaction_model.py` | SPED/additive models, data parsing, training and condition-level evaluation |
| `src/sped/basic_metrics.py` | `ppline/eval_protocol.py` | Backward-compatible base metrics used by model training |
| `src/sped/metrics.py` | `ppline/eval_protocol_v2.py` | Paper-facing delta, DEG, R², WMSE and direction metrics |
| `experiments/empirical_additive.py` | `revcode/empirical_additive.py` | Exact empirical additive baseline |
| `experiments/sped_loss_ablation.py` | `revcode/sped_loss_ablation.py` | Matched single-supervision ablation |
| `experiments/paired_bootstrap.py` | `revcode/paired_bootstrap.py` | Condition-level paired bootstrap |
| `experiments/compare_predictions.py` | `revcode/compare.py` | Hierarchical comparison across splits |
| `experiments/summarize_runs.py` | `revcode/summarize_runs.py` | Repeated-run mean, SD and confidence intervals |
| `protocols/*.json` | `ppline/splits/*.json` | Fixed evaluation protocols |

Only import paths and default locations were changed in copied scripts. The
working notebooks, CPA/GEARS adapters, graph-prior development code, large data,
weights, figures, and cached results remain outside this core package.
