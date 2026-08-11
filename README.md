# SPED publication code

It contains the core SPED model, evaluation metrics, fixed Norman splits, and the
experiments needed to reproduce the main methodological claims. Large datasets,
checkpoints, exploratory notebooks, and generated outputs are excluded.

## Repository layout

```text
publication_code/
├── make_all_results.py         # one-command result pipeline
├── src/sped/                   # reusable model and metrics
├── experiments/               # command-line experiments and statistics
├── protocols/                 # fixed train/test split JSON files
├── tests/                     # lightweight unit tests
├── DATA.md                     # data source, preprocessing and checksums
├── environment.yml            # pinned environment used for the experiments
├── requirements.txt
└── pyproject.toml
```

The primary SPED implementation is `sped.model.AdditiveOnlyModel`. It predicts a
double perturbation as the control reference plus two learned single-gene
effects. `InteractionPerturbationModel` is retained as the nonlinear interaction
extension used during model-development ablations.

## Data contract

The processed Norman AnnData file must contain:

- `adata.obs["guide_identity"]`, with labels such as `ctrl`, `GENE+ctrl`, and
  `GENE1+GENE2`;
- `adata.layers["log_expr"]` (or expression in `adata.X` when configured);
- gene identifiers in `adata.var_names`.

The default location is
`data/Norman/norman_2019_full_adata.h5ad`. Data are not included because of size
and redistribution requirements. Every data-dependent command accepts
`--adata /path/to/file.h5ad`.

## Installation

The exact tested environment is recorded in `environment.yml`; see `DATA.md` for data acquisition and preprocessing. Python 3.10 is recommended.

```bash
cd publication_code
conda env create -f environment.yml
conda activate sped
```

Alternatively, install the reusable package in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For GPU experiments, install the PyTorch build matching the local CUDA version.

## One-command reproduction

Run the complete five-split pipeline (empirical additive, SPED loss experiment, paired comparison, and summary tables):

```bash
python make_all_results.py \
  --adata /path/to/norman_2019_full_adata.h5ad \
  --device cuda:0
```

Inspect every command without running it:

```bash
python make_all_results.py \
  --adata /path/to/norman_2019_full_adata.h5ad \
  --dry-run
```

Run a small end-to-end CPU check:

```bash
python make_all_results.py \
  --adata /path/to/norman_2019_full_adata.h5ad \
  --smoke
```

The runner writes `outputs/run_manifest.json` after a full run and `outputs/smoke/run_manifest.json` after a smoke run. Use `python make_all_results.py --help` for stage selection, seeds, hyperparameters, resume behavior, and other options.

## Individual experiments

Empirical additive baseline, one-split smoke run:

```bash
python experiments/empirical_additive.py \
  --adata /path/to/norman_2019_full_adata.h5ad \
  --mode smoke
```

Empirical additive baseline on all five standard splits:

```bash
python experiments/empirical_additive.py \
  --adata /path/to/norman_2019_full_adata.h5ad \
  --mode standard
```

Matched SPED single-supervision ablation:

```bash
python experiments/sped_loss_ablation.py \
  --adata /path/to/norman_2019_full_adata.h5ad \
  --device cuda:0
```

Use `--smoke --device cpu` for a two-epoch structural check. Outputs are written
under `outputs/` and ignored by Git. Smoke artifacts are isolated under `outputs/smoke/` so they cannot be resumed as full runs.

After the baseline and SPED runs finish, compare aligned predictions across all
splits:

```bash
python experiments/compare_predictions.py
```

Run a paired bootstrap for any two aligned prediction archives:

```bash
python experiments/paired_bootstrap.py \
  --left outputs/loss_ablation/<sped-archive>.npz \
  --right outputs/empirical_additive/<baseline-archive>.npz \
  --left-name SPED \
  --right-name empirical-additive \
  --output outputs/statistics/paired_bootstrap.csv
```

Use `python <script> --help` for all available options.

## Verification

From this directory:

```bash
PYTHONPATH=src python -m compileall -q src experiments tests
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Reproducibility notes

- The versioned split JSON files must not be regenerated for reported runs.
- The main comparison evaluates held-out double perturbations whose component
  genes have observed single-perturbation profiles.
- Pearson delta and DEG Pearson delta subtract the control reference, reducing
  the control-profile bias of raw-expression correlations.
- The primary uncertainty source is the held-out-combination split. Report all
  per-split scores, not only their mean.
- Archive generated prediction matrices for paired, condition-level bootstrap
  comparisons.

## Public-release checklist

Add the final paper citation, author/contact information, and a license chosen by the project owner. Data provenance, checksums, and the tested environment are now recorded in `DATA.md` and `environment.yml`. This directory deliberately does not assert a license.
