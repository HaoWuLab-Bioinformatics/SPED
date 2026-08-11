# Data acquisition and preprocessing

This repository does not redistribute the Norman Perturb-seq expression matrix.
The instructions below document the public source, the exact preprocessing used
for the reported SPED experiments, and checksums of the local files from which
the benchmark was produced.

## Dataset and citation

The experiments use the combinatorial CRISPR activation Perturb-seq dataset from:

> Norman TM, Horlbeck MA, Replogle JM, et al. Exploring genetic interaction
> manifolds constructed from rich single-cell phenotypes. *Science*.
> 2019;365(6455):786–793. https://doi.org/10.1126/science.aax4438

Primary archive: NCBI GEO accession
[GSE133344](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344).

This project loaded the preprocessed Norman dataset distributed by the official
[GEARS repository](https://github.com/snap-stanford/GEARS), using
`cell-gears==0.1.2`. In that release, `PertData.load(data_name="norman")`
downloads Harvard Dataverse file
[6154020](https://dataverse.harvard.edu/api/access/datafile/6154020).
The original paper archive remains the authoritative source for the experiment;
the GEARS file is the machine-readable starting point used by this codebase.

Users must comply with the terms attached to the original dataset and should
cite both Norman et al. and GEARS when using the GEARS-preprocessed copy.

## Download

Create the environment first from `environment.yml`, then run from the repository
root:

```bash
conda env create -f environment.yml
conda activate sped
python - <<'PY'
from gears import PertData

pert_data = PertData("data/Norman_full")
pert_data.load(data_name="norman")
print(pert_data.adata)
PY
```

This creates the GEARS input at approximately:

```text
data/Norman_full/norman/perturb_processed.h5ad
```

The direct GEARS download endpoint can also be used, but the loader is preferred
because it creates the expected directory structure:

```text
https://dataverse.harvard.edu/api/access/datafile/6154020
```

## Preprocessing used for the paper

The local preprocessing run used the following procedure:

1. Load `pert_data.adata` with `PertData.load(data_name="norman")`.
2. Copy the AnnData object. The loaded matrix had 89,357 cells, 5,045 genes and
   277 original condition labels.
3. Normalize condition names into `guide_identity`:
   - control: `ctrl`;
   - single perturbation: `GENE+ctrl`, with the gene first;
   - double perturbation: `GENE1+GENE2`, with genes sorted alphabetically.
4. Inspect `adata.X`. Its observed range was 0.000–8.905, so it was already
   log-normalized and no additional normalization was applied. The preprocessing
   code only runs `scanpy.pp.normalize_total(target_sum=1e4)` followed by
   `scanpy.pp.log1p` when `adata.X.max() > 20`.
5. Select 2,000 highly variable genes with:

   ```python
   sc.pp.highly_variable_genes(
       adata,
       n_top_genes=2000,
       flavor="seurat_v3",
       layer="counts",
   )
   adata = adata[:, adata.var["highly_variable"]].copy()
   ```

   This operation was run on all cells. Although an old notebook comment says
   “non-control cells”, the executed code did not subset the cells before HVG
   selection.
6. Parse each normalized condition into `condition_type`, `gene1` and `gene2`.
7. Save the processed object as:

   ```text
   data/Norman/norman_2019_full_adata.h5ad
   ```

8. Use the versioned JSON files in `protocols/` for all reported train/test
   assignments. Do not regenerate these files when reproducing paper results.

The five standard protocols hold out 25 of 128 double-perturbation conditions
per split, retain 103 double conditions for training, and include all 102 single
conditions in training.

## Final AnnData contract

The expected paper input has:

```text
shape:              89,357 cells × 2,000 genes
unique conditions:  231
condition types:    1 control, 102 single, 128 double
X dtype:            float32
layers:             counts
```

Required fields are:

- `obs["guide_identity"]`
- `obs["condition_type"]`
- `obs["gene1"]`
- `obs["gene2"]`
- `var_names`
- `layers["counts"]`

SPED reads `layers["log_expr"]` when present and otherwise uses `adata.X`. The
paper input does not contain `log_expr`, so the reported runs use `adata.X`.

## Checksums

The following checksums describe the exact local files used to build and run the
benchmark. A GEARS release or upstream archive may change in the future, so
verify files before comparing numerical results.

| File | Size (bytes) | SHA256 | Additional checksum |
|---|---:|---|---|
| `data/Norman_full/norman.zip` | 168,758,985 | `c1938353f6f41829c9137f073ecb843ce4d4114528b001b48312b019f739d877` | MD5 `cdc41d6050e619c37fd9dd44d440e2b4` |
| `data/Norman_full/norman/perturb_processed.h5ad` | 2,228,610,012 | `23ffb0fac6a847ff927cf7509d80d85052bfefbfb97610786a2dafaaefa0b6a0` | — |
| `data/Norman/norman_2019_full_adata.h5ad` | 915,489,159 | `a59c7cb0bf9f1a3eb7ccaa018affac770479e05012c9b8de407c532acc215174` | — |

Verify the final file with:

```bash
sha256sum data/Norman/norman_2019_full_adata.h5ad
```

Expected output:

```text
a59c7cb0bf9f1a3eb7ccaa018affac770479e05012c9b8de407c532acc215174
```

## Running the benchmark with an external data location

The large H5AD file does not have to be copied into this repository. Every main
experiment accepts an explicit path:

```bash
python make_all_results.py --stages additive \
  --adata /absolute/path/to/norman_2019_full_adata.h5ad

python make_all_results.py --stages sped \
  --adata /absolute/path/to/norman_2019_full_adata.h5ad \
  --device cuda:0
```
