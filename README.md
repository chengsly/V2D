# A machine-learning framework to characterize functional disease architectures and prioritise disease variants

Modeling disease effect sizes from genome-wide association studies (GWAS) is critical for advancing our understanding of human disease functional architectures, but also for providing priors improving the power and accuracy of fine-mapping. Here, we propose the variant-to-disease (V2D) framework, a novel approach that models disease effect sizes using machine-learning algorithms on genome-wide estimates of posterior mean squared causal effect sizes and functional annotations. We benchmarked the V2D framework using simulations and real data analysis, demonstrating that it provides reliable estimates of heritability (h2) functional enrichment. By applying the V2D framework with linear trees on 15 UK Biobank traits, we identified non-linear relationships between constraint and regulatory annotations, highlighting constrained regulatory variants as the main functional component of disease functional architecture (h2 enrichment = 17.3 ± 1.0x across 79 independent GWAS). By applying the V2D framework with neural networks, we constructed GWAS prioritization scores (V2D-MLP), which were extremely enriched in h2 (enrichment of the top 1% V2D-MLP scores = 20.6 ± 0.7x), outperformed existing prioritization scores in the analysis of different GWAS datasets, were transportable to analyze gene expression and non-European datasets, and improved variant prioritization in GWAS fine-mapping studies.

# Variant2Disease-V2D

This is the repository for Variant2Disease-V2D, a variant-to-disease (V2D) framework that models disease effect sizes using machine-learning algorithms on genome-wide estimates of posterior mean squared causal effect sizes and functional annotations.


# `v2d.py` Documentation Command-line Interface

## Quick Start

```bash
python v2d.py \
  --beta2 beta2/15ukbb.beta2_prior.common.txt.gz \
  --annot annotations/baselineLF.common \
  --model mlp \
  --n_neurons 64 \
  --n_layers 3 \
  --nbseed 10 \
  --print_mse --print_model \
  --out V2D/ukbb/v2d_ukbb.common
```


## Required Inputs

| Argument   | Type | Required | Description |
|------------|------|:-------:|-------------|
| `--model`  | `str` in `{linear, rf, mlp, xgb, tree}` | ✅ | Model family to train/evaluate. Enables model‑specific flags below. |
| `--annot`  | `str` (path) | ✅ | Path/prefix to annotations/features (e.g., baseline feature matrix). |
| `--beta2`  | `str` (path) | ✅ | Path to response/target file (e.g., β² or label vector). |
| `--out`    | `str` (path/prefix) | ✅ | Output file or prefix for logs/artifacts. |



## Reproducibility & Multi‑runs

| Argument       | Type | Default | Description |
|----------------|------|:------:|-------------|
| `--init_seed`  | `int` | `0` | Random seed used to initialize the first run. |
| `--nbseed`     | `int` | `1` | Number of repeated runs with different seeds (e.g., seed, seed+1, …). Useful for reporting mean/variance. |



## Feature Exclusion

| Argument    | Type        | Default | Description |
|-------------|-------------|:------:|-------------|
| `--exclude` | `strings…`  | `[]`   | List of feature/annotation column names to drop at load time. Example: `--exclude MAFbin_frequent_1 GCcontent`. |


## Output Controls

| Argument        | Type | Default | Description |
|-----------------|------|:------:|-------------|
| `--print_mse`   | flag | `False` | Print MSE to stdout (and/or include in logs). |
| `--print_model` | flag | `False` | Print fitted model summary (coefficients, architecture, and/or tree params). |



## Model‑Specific Hyperparameters

### MLP (Multi‑Layer Perceptron)

| Argument       | Type | Default | Description |
|----------------|------|:------:|-------------|
| `--n_neurons`  | `int` | *none* | Hidden width per layer. |
| `--n_layers`   | `int` | *none* | Number of hidden layers. |

### Tree / Random Forest

| Argument           | Type | Default | Applies to | Description |
|-------------------|------|:------:|-----------|-------------|
| `--min_samples_leaf` | `int` | *none* | `tree`,`rf` | Minimum samples at a leaf node. |
| `--max_depth`        | `int` | *none* | `tree`,`rf`,`xgb` | Maximum tree depth (also used by XGBoost). |
| `--n_estimators`     | `int` | *none* | `rf`,`xgb` | Number of trees (estimators). |

### XGBoost

| Argument            | Type   | Default | Description |
|--------------------|--------|:------:|-------------|
| `--min_child_weight` | `int`   | *none* | Minimum sum of instance weight (Hessian) needed in a child. |
| `--gamma`            | `float` | *none* | Minimum loss reduction to make a split. |
| `--subsample`        | `float` | *none* | Row subsample ratio per tree (0–1]. |
| `--scale_pos_weight` | `int`   | *none* | Positive class weight (useful for imbalance; ignored for pure regression). |
| `--learning_rate`    | `float` | *none* | Step size (a.k.a. `eta`). |



## Compatibility Matrix 

| Flag group | linear | mlp | tree | rf | xgb |
|---|:--:|:--:|:--:|:--:|:--:|
| `--n_neurons`, `--n_layers` |  | ✅ |  |  |  |
| `--min_samples_leaf` |  |  | ✅ | ✅ |  |
| `--max_depth` |  |  | ✅ | ✅ | ✅ |
| `--n_estimators` |  |  |  | ✅ | ✅ |
| `--min_child_weight`, `--gamma`, `--subsample`, `--scale_pos_weight`, `--learning_rate` |  |  |  |  | ✅ |



## Examples

### 1) MLP 
```bash
python v2d.py \
  --beta2 beta2/15ukbb.beta2_prior.common.txt.gz \
  --annot annotations/baselineLF.common \
  --model mlp \
  --n_neurons 64 \
  --n_layers 3 \
  --nbseed 10 \
  --print_mse --print_model \
  --out V2D/ukbb/v2d_ukbb.common
```

### 2) Random Forest 
```bash
python v2d.py \
  --beta2 beta2/15ukbb.beta2_prior.common.txt.gz \
  --annot annotations/baselineLF.common \
  --model rf \
  --n_estimators 500 \
  --max_depth 12 \
  --min_samples_leaf 5 \
  --exclude MAFbin_frequent_1 MAFbin_frequent_2 \
  --out V2D/ukbb/rf_depth12_leaf5
```

### 3) XGBoost 
```bash
python v2d.py \
  --beta2 beta2/15ukbb.beta2_prior.common.txt.gz \
  --annot annotations/baselineLF.common \
  --model xgb \
  --n_estimators 800 \
  --max_depth 8 \
  --learning_rate 0.05 \
  --subsample 0.8 \
  --min_child_weight 3 \
  --gamma 0.0 \
  --out V2D/ukbb/xgb_mw3_eta005_ss08_md8
```

### 4) Single decision tree 
```bash
python v2d.py \
  --beta2 beta2/15ukbb.beta2_prior.common.txt.gz \
  --annot annotations/baselineLF.common \
  --model tree \
  --max_depth 4 \
  --min_samples_leaf 20 \
  --print_model \
  --out V2D/ukbb/tree_depth4_leaf20
```

### 5) Linear 
```bash
python v2d.py \
  --beta2 beta2/15ukbb.beta2_prior.common.txt.gz \
  --annot annotations/baselineLF.common \
  --model linear \
  --print_mse --print_model \
  --out V2D/ukbb/linear_baseline
```
