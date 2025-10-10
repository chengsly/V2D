# Variant2Disease-V2D

This is the repository for Variant2Disease-V2D, a variant-to-disease (V2D) framework that models disease effect sizes using machine-learning algorithms on genome-wide estimates of posterior mean squared causal effect sizes and functional annotations.



# `v2d.py` — Input File Formats

This document describes the expected formats for the annotation (`--annot`) and beta2/target (`--beta2`) files used by `v2d.py`. It also clarifies how records are matched, how to exclude features, and common pitfalls. Examples are provided to make validation easy.

> - Files are tab-delimited by default (set with `--delimiter`, default `"\t"`).  
> - Gzipped files (`.gz`) are supported; plain text is fine too.  
> - Rows are matched by the key `(CHR, BP, SNP, ID)`.  
> - `--annot` contains features (one row per variant); `--beta2` contains the target column(s).  
> - Avoid duplicate keys. Keep consistent row counts *or* enable merging by keys in your codebase.



## 1) Annotation file (`--annot` and `--pred`)

**Purpose:** feature matrix for each variant/row used by the model.

**File type:** TSV/CSV (default delimiter is `\t`), optionally gzipped (`.gz`).  
**Required columns (first four, in this order):**  
- `CHR` — chromosome (integer or string like `"X"`, but prefer integer coding: 1–22, 23 for X, 24 for Y)  
- `BP` — base-pair position (integer)  
- `SNP` — variant identifier (string; e.g., rsID)  
- `ID` — **unique** variant ID to disambiguate duplicates when multiple rows share the same rsID (e.g., you can provide `chr:bp:ref:alt`)


**Feature columns:**  
- One or more numeric feature columns, e.g., `MAFbin_frequent_1`, `GCcontent`, `conserved_phylop`, etc.  
- All features must be numeric (`int`/`float`). If you have categories, one-hot encode them before running `v2d.py`.  
- Missing values should be replaced by the genome-wide average value.

**Header:** A header row is expected.  

### Minimal example (`.tsv`)
```text
CHR	BP	SNP	ID	MAFbin_frequent_1	MAFbin_frequent_2	GCcontent
1	10177	rs367896724	1:10177:A:T	0	0	0.41
1	10352	rs201106462	1:10352:T:C	1	0	0.36
1	10505	rs531730856	1:10505:C:G	0	1	0.39
```


## 2) Beta2/target file (`--beta2`)

**Purpose:** provides the posterior estimates of squared normalized effect sizes for each variant.

**File type:** TSV/CSV (default delimiter is `\t`), optionally gzipped (`.gz`).  
**Required columns (first four, in this order):**  
- `CHR`, `BP`, `SNP`, `ID` — must match the annotation file’s keys.  
- `Y` — the posterior estimates of squared normalized effect sizes (float).
- We recommend removing variants in the MHC locus from this file.

**Header:** A header row is expected.  

### Minimal example (`.tsv`)
```text
CHR	BP	SNP	ID	Y
1	10177	rs367896724	1:10177:A:T	0.0000
1	10352	rs201106462	1:10352:T:C	0.0000
1	10505	rs531730856	1:10505:C:G	0.0125
```




# `v2d.py` Documentation Command-line Interface

## Quick Start
Example of command line to compute V2D scores from the linear model and output MSE
```bash
python v2d.py \
  --model linear \
  --annot annotations/baselineLF.common. \
  --beta2 beta2/15ukbb.beta2_prior.common.txt.gz \
  --print_mse \
  --pred annotations/baselineLF.common. \
  --out V2D/ukbb/v2d_ukbb.common
```

## Required Inputs

| Argument   | Type | Required | Description |
|------------|------|:-------:|-------------|
| `--model`  | `str` in `{linear, rf, mlp, xgb, tree}` | ✅ | Model family to train/evaluate. Enables model-specific flags below. |
| `--annot`  | `str` (path) | ✅ | Path/prefix to annotations/features (e.g., baseline feature matrix). |
| `--beta2`  | `str` (path) | ✅ | Path to response/target file (e.g., β² or label vector). |
| `--out`    | `str` (path/prefix) | ✅ | Output prefix for predictions/logs/artifacts. |

## Reproducibility & Multi-runs

| Argument       | Type | Default | Description |
|----------------|------|:------:|-------------|
| `--init_seed`  | `int` | `0` | Random seed used to initialize the first run. |
| `--nbseed`     | `int` | `1` | Number of repeated runs with different seeds (e.g., seed, seed+1, …). Useful for reporting mean/variance. |

## Feature Exclusion

| Argument    | Type        | Default | Description |
|-------------|-------------|:------:|-------------|
| `--exclude` | `strings…`  | `[]`   | List of feature/annotation column names to drop at load time. Example: `--exclude MAFbin_frequent_1 CpG_Content_50kb_common`. |

## Output Controls

| Argument        | Type | Default | Description |
|-----------------|------|:------:|-------------|
| `--print_mse`   | flag | `False` | Print MSE to stdout (and/or include in logs). |
| `--print_model` | flag | `False` | Print fitted model summary (coefficients, architecture, and/or tree params). |
| `--pred`        | `strings…` | `[]` | Print V2D scores from annotation files (should have the same column names as files provided in `--annot`). |

## Model-Specific Hyperparameters

### MLP (Multi-Layer Perceptron)

| Argument       | Type | Default | Description |
|----------------|------|:------:|-------------|
| `--n_neurons`  | `int` | *none* | Hidden width per layer. |
| `--n_layers`   | `int` | *none* | Number of hidden layers. |

### Tree / Random Forest

| Argument             | Type | Default | Applies to | Description |
|---------------------|------|:------:|-----------|-------------|
| `--min_samples_leaf`| `int` | *none* | `tree`,`rf` | Minimum samples at a leaf node. |
| `--max_depth`       | `int` | *none* | `tree`,`rf`,`xgb` | Maximum tree depth (also used by XGBoost). |
| `--n_estimators`    | `int` | *none* | `rf`,`xgb` | Number of trees (estimators). |

### XGBoost

| Argument             | Type   | Default | Description |
|---------------------|--------|:------:|-------------|
| `--min_child_weight`| `int`   | *none* | Minimum sum of instance weight (Hessian) needed in a child. |
| `--gamma`           | `float` | *none* | Minimum loss reduction to make a split. |
| `--subsample`       | `float` | *none* | Row subsample ratio per tree (0–1]. |
| `--scale_pos_weight`| `int`   | *none* | Positive class weight (useful for imbalance; ignored for pure regression). |
| `--learning_rate`   | `float` | *none* | Step size (a.k.a. `eta`). |

## Compatibility Matrix 

| Flag group | linear | mlp | tree | rf | xgb |
|---|:--:|:--:|:--:|:--:|:--:|
| `--n_neurons`, `--n_layers` |  | ✅ |  |  |  |
| `--min_samples_leaf` |  |  | ✅ | ✅ |  |
| `--max_depth` |  |  | ✅ | ✅ | ✅ |
| `--n_estimators` |  |  |  | ✅ | ✅ |
| `--min_child_weight`, `--gamma`, `--subsample`, `--scale_pos_weight`, `--learning_rate` |  |  |  |  | ✅ |


# Tutorial

## 1) Download data from Cheng et al. medrxiv

Formated beta2 and annotation files are available in https://zenodo.org/records/17257765.

## 2) Hyperparameter search and MSE (LEOCO)

Compute MSE under a leave-even/odd-chromosomes-out (LEOCO) scheme to pick hyperparameters across models. We evaluate a grid of choices for each model and select the configuration with the minimal aggregated LEOCO MSE (for MLP, averages are taken over multiple seeds).


### Paper grids

**Decision Tree** 
- `LEAFS = 100000 50000 25000 10000 5000`
- `DEPTH = 3 4 5 6 7 8 9 10`

**Random Forest** 
- `N_EST = 100 200 500 1000 2500 5000`
- `MAX_DEPTH = 5 10 20 30 40 50 100 200`

**XGBoost** 
- `N_EST = 25 40 50 100 200 500 1000`
- `MAX_DEPTH = 1 2 5 10 15 25`
- `GAMMA = 0 1 10`
- `MIN_CHILD_WEIGHT = 1 6 8 10`
- `SUBSAMPLE = 0.6 0.8 1.0`

**MLP** 
- `N_LAYERS = 1 2 3 4 5 6 10`
- `N_NEURONS = 1 2 3 4 5 6 10 15`


```bash
BETA2="beta2/15ukbb.beta2_prior.common.txt.gz"
ANNOT="annotations/baselineLF.common."
OUTDIR="runs_paper"
mkdir -p "$OUTDIR"

# --- Decision Tree ---
for leaf in 100000 50000 25000 10000 5000; do
  for depth in 3 4 5 6 7 8 9 10; do
    tag="tree_leaf${leaf}_depth${depth}"
    python v2d.py --beta2 "$BETA2" --annot "$ANNOT" \
      --model tree \
      --min_samples_leaf "$leaf" \
      --max_depth "$depth" \
      --print_mse \
      --out "$OUTDIR/$tag"
  done
done

# --- Random Forest ---
for ne in 100 200 500 1000 2500 5000; do
  for md in 5 10 20 30 40 50 100 200; do
    tag="rf_ne${ne}_md${md}"
    python v2d.py --beta2 "$BETA2" --annot "$ANNOT" \
      --model rf \
      --n_estimators "$ne" \
      --max_depth "$md" \
      --print_mse \
      --out "$OUTDIR/$tag"
  done
done

# --- XGBoost ---
for ne in 25 40 50 100 200 500 1000; do
  for md in 1 2 5 10 15 25; do
    for g in 0 1 10; do
      for mcw in 1 6 8 10; do
        for ss in 0.6 0.8 1.0; do
          tag="xgb_ne${ne}_md${md}_g${g}_mcw${mcw}_ss${ss}"
          python v2d.py --beta2 "$BETA2" --annot "$ANNOT" \
            --model xgb \
            --n_estimators "$ne" \
            --max_depth "$md" \
            --gamma "$g" \
            --min_child_weight "$mcw" \
            --subsample "$ss" \
            --learning_rate 0.05 \
            --print_mse \
            --out "$OUTDIR/$tag"
        done
      done
    done
  done
done

# --- MLP ---
for w in 1 2 3 4 5 6 10 15; do
  for L in 1 2 3 4 5 6 10; do
    tag="mlp_w${w}_L${L}"
    python v2d.py --beta2 "$BETA2" --annot "$ANNOT" \
      --model mlp \
      --n_neurons "$w" \
      --n_layers "$L" \
      --nbseed 10 \
      --print_mse \
      --out "$OUTDIR/$tag"
  done
done

# --- Linear baseline ---
python v2d.py --beta2 "$BETA2" --annot "$ANNOT" \
  --model linear \
  --print_mse \
  --out "$OUTDIR/linear_baseline"

```


## 3) Predicting V2D scores
To predict V2D scores, use the `--pred` option followed by annotation files in the same format as above.
If you want to compute V2D scores for the same SNPs used in training, provide the same annotation files to `--annot` and `--pred`.
If you want to compute V2D scores for a different set of SNPs, provide different annotation files for `--annot` and `--pred`.



```bash
python v2d.py \
  --beta2 beta2/15ukbb.beta2_prior.common.txt.gz \
  --annot annotations/baselineLF.common. \
  --model mlp \
  --n_neurons 6 \
  --n_layers 5 \
  --nbseed 10 \
  --pred annotations/baselineLF.common. \
  --print_model \
  --out runs/v2d_mlp_common
```

This writes per-chromosome files: `runs/v2d_mlp_common.<CHR>.csv` with columns `CHR, BP, SNP, ID, V2D`, and (optionally) saves fitted models (e.g., `runs/v2d_mlp_common.even.joblib`, `runs/v2d_mlp_common.odd.joblib`).

## 4) Plotting a tree of annotations

## Utility: `plot_tree.py` (generate tree figures)

Command line to plot a tree of depth 5 with a minimum 25K SNPs per leaf.

**Usage**
```bash
python plot_tree.py \
  --beta2 beta2/15ukbb.beta2_prior.common.txt.gz \
  --annot annotations/baselineLF.common. \
  --max_depth 5 \
  --min_samples_leaf 25000 \
  --out tree_result.pdf
```

### Requirements

Ensure dependencies are installed:
```bash
pip install numpy pandas scikit-learn xgboost joblib scipy matplotlib
```

**Data availability**  
https://zenodo.org/records/17257765

