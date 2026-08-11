# SPED

**Single-Perturbation Effect Decomposition for data-efficient prediction of unmeasured double-gene perturbations**

SPED is a compositional model for predicting transcriptomic responses to double-gene perturbations. It learns a reusable expression-effect vector for each perturbation gene from measured single-perturbation responses and predicts an unmeasured pair by composing the two learned effects with a control reference.

This repository contains the core SPED implementation, fixed Norman2019 evaluation protocols, evaluation metrics, matched loss-ablation experiments, an empirical additive reference, and scripts for condition-level statistical comparisons. Large expression matrices, trained checkpoints, exploratory notebooks, and generated outputs are not distributed in the repository.

## Method overview

For perturbation genes $A$ and $B$, SPED predicts the double-perturbation expression profile as

```text
predicted(A + B) = control reference + learned effect(A) + learned effect(B)
```

The effect decoder is shared across genes. Single-perturbation observations directly supervise the corresponding learned effects, allowing SPED to make predictions even when no double-perturbation condition is used for training. The principal implementation is `sped.model.AdditiveOnlyModel`. `InteractionPerturbationModel` is retained as the symmetric interaction extension used in the ablation analyses.

The primary evaluation task holds out double-perturbation conditions while retaining measured single-perturbation responses for their component genes. It therefore evaluates extrapolation to **unmeasured combinations of observed genes**, not strict generalization to genes with no perturbation-response measurements.

## Repository structure

```text
.
├── make_all_results.py          # Reproduce the publication-facing analyses
├── src/sped/
│   ├── model.py                 # SPED models, datasets, and training utilities
│   ├── metrics.py               # Paper-facing evaluation metrics
│   └── basic_metrics.py         # Backward-compatible evaluation utilities
├── experiments/
│   ├── empirical_additive.py    # Empirical additive reference
│   ├── sped_loss_ablation.py    # Matched single-supervision ablation
│   ├── compare_predictions.py   # Paired SPED/reference comparison
│   ├── paired_bootstrap.py      # Condition-level paired bootstrap
│   └── summarize_runs.py        # Repeated-run summaries and intervals
├── protocols/                   # Versioned train/test split definitions
├── tests/                       # Lightweight unit tests
├── DATA.md                      # Dataset source, preprocessing, and checksums
├── SOURCE_MANIFEST.md           # Mapping from working code to this release
├── environment.yml              # Reproducibility environment
├── requirements.txt             # Minimal Python dependencies
└── pyproject.toml                # Installable package metadata
```

## Requirements

- Python 3.10
- NumPy
- pandas
- SciPy
- Scanpy
- PyTorch
- `cell-gears==0.1.2` for downloading the GEARS-preprocessed Norman2019 dataset

The reported experiments were run on Linux with Python 3.10.18 and a CUDA 12.4 build of PyTorch. CPU execution is supported for tests and smoke runs; full experiments are substantially faster on a CUDA-capable GPU.

## Installation

### Reproducibility environment

The pinned environment used for the experiments is provided in `environment.yml`:

```bash
git clone https://github.com/HaoWuLab-Bioinformatics/SPED.git
cd SPED
conda env create -f environment.yml
conda activate sped
```

The environment includes the CUDA 12.4 PyTorch build. On systems without a compatible GPU, install an appropriate CPU or CUDA build of PyTorch instead.

### Minimal editable installation

To use the reusable `sped` package without recreating the complete experimental environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Data

The experiments use the Norman2019 K562 combinatorial CRISPR activation Perturb-seq dataset:

> Norman TM, Horlbeck MA, Replogle JM, et al. Exploring genetic interaction manifolds constructed from rich single-cell phenotypes. *Science*. 2019;365(6455):786–793. https://doi.org/10.1126/science.aax4438

The expression matrix is not redistributed because of its size and upstream data terms. The primary archive is [GEO GSE133344](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344). The experiments in this repository start from the preprocessed Norman dataset distributed through the official [GEARS](https://github.com/snap-stanford/GEARS) loader.

Complete download instructions, preprocessing details, expected AnnData fields, dataset dimensions, and SHA256 checksums are provided in [`DATA.md`](DATA.md).

The default processed-data path is:

```text
data/Norman/norman_2019_full_adata.h5ad
```

The file may be stored elsewhere by passing an explicit `--adata` path.

## Quick validation

After installation, run the unit tests from the repository root:

```bash
PYTHONPATH=src python -m compileall -q src experiments tests
PYTHONPATH=src python -m unittest discover -s tests -v
```

Inspect the complete pipeline without launching any experiment:

```bash
python make_all_results.py \
  --adata /path/to/norman_2019_full_adata.h5ad \
  --dry-run
```

Run a small end-to-end CPU smoke test:

```bash
python make_all_results.py \
  --adata /path/to/norman_2019_full_adata.h5ad \
  --smoke \
  --device cpu
```

Smoke-test artifacts are written to `outputs/smoke/` and are isolated from full experimental results.

## Reproducing the analyses

Run the complete five-split pipeline:

```bash
python make_all_results.py \
  --adata /path/to/norman_2019_full_adata.h5ad \
  --device cuda:0
```

The default pipeline executes, in order:

1. the empirical additive reference;
2. the matched SPED single-supervision experiment;
3. paired comparisons of aligned condition-level predictions; and
4. summary-table generation.

Use `--stages` to run selected stages and `--help` to inspect all configuration options:

```bash
python make_all_results.py --help

python make_all_results.py \
  --stages additive sped \
  --adata /path/to/norman_2019_full_adata.h5ad \
  --device cuda:0
```

The standard configuration uses seeds 0--4, 50 training epochs, `lambda_single` values of 0, 0.1, 0.3, 1, and 3, and the versioned split files in `protocols/`.

## Running individual experiments

### Empirical additive reference

One-split smoke run:

```bash
python experiments/empirical_additive.py \
  --adata /path/to/norman_2019_full_adata.h5ad \
  --mode smoke
```

All five standard splits:

```bash
python experiments/empirical_additive.py \
  --adata /path/to/norman_2019_full_adata.h5ad \
  --mode standard
```

### Matched SPED loss ablation

```bash
python experiments/sped_loss_ablation.py \
  --adata /path/to/norman_2019_full_adata.h5ad \
  --device cuda:0
```

For a short structural check, add `--smoke --device cpu`.

### Paired comparison

After the empirical additive and SPED runs have completed:

```bash
python experiments/compare_predictions.py
```

For two explicitly selected aligned prediction archives:

```bash
python experiments/paired_bootstrap.py \
  --left outputs/loss_ablation/<sped-archive>.npz \
  --right outputs/empirical_additive/<reference-archive>.npz \
  --left-name SPED \
  --right-name empirical-additive \
  --output outputs/statistics/paired_bootstrap.csv
```

Run `python <script> --help` for the complete argument list of any experiment.

## Outputs

Generated files are written under `outputs/` and excluded from version control. A complete run produces:

```text
outputs/
├── empirical_additive/          # Per-split reference predictions and metrics
├── loss_ablation/               # SPED predictions and loss-ablation metrics
├── statistics/                  # Paired comparisons and bootstrap summaries
├── tables/                      # Aggregated result tables
└── run_manifest.json            # Commands and settings executed by the runner
```

The runner supports resuming completed SPED configurations. Use `--force` to retrain them. Smoke outputs are always separated under `outputs/smoke/` and cannot be resumed as full runs.

## Reproducibility notes

- Use the versioned JSON files in `protocols/` for reported experiments; do not regenerate the published train/test assignments.
- Standard splits hold out 25 of the 128 measured double-perturbation conditions and retain all 102 single-perturbation conditions in training.
- Predictions and observations are evaluated as condition-level mean expression profiles, so each held-out condition contributes equally to aggregate metrics.
- Pearson delta and DEG Pearson delta are calculated after subtracting the control reference.
- The five standard runs vary the held-out-combination split together with model initialization. Per-split scores should therefore be retained and reported alongside aggregate summaries.
- Prediction archives contain aligned condition-level matrices for paired bootstrap comparisons.
- The exact data provenance and preprocessing decisions, including the highly variable gene selection procedure, are documented in `DATA.md`.

## Citation

If you use this repository, please cite the associated manuscript:

> *SPED: Single-Perturbation Effect Decomposition for Data-Efficient Prediction of Unmeasured Double-Gene Combinations.*

The complete journal citation will be added after publication.
