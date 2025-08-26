# A machine-learning framework to characterize functional disease architectures and prioritise disease variants

Modeling disease effect sizes from genome-wide association studies (GWAS) is critical for advancing our understanding of human disease functional architectures, but also for providing priors improving the power and accuracy of fine-mapping. Here, we propose the  We benchmarked the V2D framework using simulations and real data analysis, demonstrating that it provides reliable estimates of heritability (h2) functional enrichment. By applying the V2D framework with linear trees on 15 UK Biobank traits, we identified non-linear relationships between constraint and regulatory annotations, highlighting constrained regulatory variants as the main functional component of disease functional architecture (h2 enrichment = 17.3 ± 1.0x across 79 independent GWAS). By applying the V2D framework with neural networks, we constructed GWAS prioritization scores (V2D-MLP), which were extremely enriched in h2 (enrichment of the top 1% V2D-MLP scores = 20.6 ± 0.7x), outperformed existing prioritization scores in the analysis of different GWAS datasets, were transportable to analyze gene expression and non-European datasets, and improved variant prioritization in GWAS fine-mapping studies.

# Variant2Disease-V2D

This is the repository for Variant2Disease-V2D, a variant-to-disease (V2D) framework that models disease effect sizes using machine-learning algorithms on genome-wide estimates of posterior mean squared causal effect sizes and functional annotations.

The paper corresponding to this repository has been published at xxxxxx. The exact code version used in the paper can be found at xxxxxxxx. If you find V2D useful, please consider to cite: 
- xxxxxxxxxxx

Please direct any questions to xxxxxxxxx.

## Installation


### Testing


## Usage

### Prediction



### Prediction File Format

As an example:



