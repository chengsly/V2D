# Computing Trait-Specific V2Dx Scores

V2Dx extends the V2D framework by integrating:

- genome-wide V2D scores
- trait-relevant cell-type-specific (CTS) annotations
- interaction terms between V2D and CTS annotations

This tutorial demonstrates how to compute schizophrenia V2Dx scores using publicly available GWAS summary statistics and cell-type-specific annotations prioritized by CT-FM.

This tutorial uses European LD scores and is appropriate for European-ancestry GWAS. Other ancestries require matched LD and reference files.

The schizophrenia GWAS summary statistics included in this tutorial are already in LDSC-munged format. For transforming your GWAS summary statistics to a LDSC-friendy format, you can refer to the script 1_munge_sumstats in our CT-FM github: https://github.com/ArtemKimUSC/CTFM

## Requirements

- bedtools: https://bedtools.readthedocs.io/en/latest/
- Python 3
- R 4.4 or later

Bedtools, Python and R should be available from the command line. In practice, the tutorial assumes that commands such as `python XX.py`, `Rscript XX.R`, and `bedtools ...` can be run from the active shell environment.


- S-LDSC: https://github.com/bulik/ldsc - we will use a newer version compatible with python3 proposed by Stephen Dorn: https://github.com/svdorn/ldsc-2.0.1

To install S-LDSC:

```bash
git clone https://github.com/svdorn/ldsc-2.0.1.git
cd ldsc-2.0.1
pip install .
```

Test the installation with:

```bash
python ldsc.py -h
```

## Main Steps

1. Download reference files and V2D scores.
2. Create S-LDSC annotation files.
3. Compute LD scores.
4. Estimate V2Dx coefficients using S-LDSC in a leave-even/odd-chromosome framework.
5. Generate final V2Dx scores.

## 1. Download Reference Files and Scripts

```bash
mkdir V2Dx
cd V2Dx
wget https://zenodo.org/records/20499413/V2Dx.tgz?download=1 -O V2Dx.tgz
```

Unpack files:

```bash
tar -xvzf V2Dx.tgz
```

Move tutorial files outside the `v2dx_ref/` directory:

```bash
mv v2dx_ref/tutorial_v2dx_scz .
```

Remove the compressed file:

```bash
rm V2Dx.tgz
```

You should now have two directories in your `V2Dx/` folder:

- `v2dx_ref/`, containing V2D scores, reference files for S-LDSC analyses, and scripts
- `tutorial_v2dx_scz/`, containing:
  - schizophrenia GWAS summary statistics: `PASS.Schizophrenia.Trubetskoy2022.sumstats.gz`
  - BED files for three CTS annotations
  - `bed.list`, where the first column is the annotation name and the second column is the path to the BED file


Define environment variables for downstream analyses.

Generic LDSC path:

```bash
LDSC="python /path_to_ldsc/ldsc.py"
```

Gazal lab example:

```bash
LDSC="python /project2/gazal_569/DATA/ldsc/ldsc-2.0.1/ldsc.py"
```

Reference directory for any V2Dx analysis:

```bash
REF_DIR="v2dx_ref"
```

Trait-specific V2Dx directory:

```bash
V2DX_DIR="tutorial_v2dx_scz"
```

## 2. Create Annotation Files

```bash
bash $REF_DIR/annotate_snps.sh $REF_DIR $V2DX_DIR
```

Outputs:

- annotation files in LDSC format in `$V2DX_DIR/annots/1000G.*`
- annotation files for approximately 20M UK Biobank variants in `$V2DX_DIR/annots/ukbb.*`

Inspect the 1000G annotation files:

```bash
zcat $V2DX_DIR/annots/1000G.22.annot.gz | head
```

Example output:

```text
base	v2d	ENCFF790XQN_blood_bodily_fluid_naive_thymus_derived_CD4_positive_alpha_beta_T_cell	ENCFF790XQN_blood_bodily_fluid_naive_thymus_derived_CD4_positive_alpha_beta_T_cell_v2d	Zhang_Glutamatergic_2	Zhang_Glutamatergic_2_v2d	Zhang_Fetal_Excitatory_Neuron_3	Zhang_Fetal_Excitatory_Neuron_3_v2d
1	0.318078749655231	0	0	0	0	0	0
1	0.196371948775364	0	0	0	0	0	0
1	0.365908260521204	0	0	0	0	0	0
1	0.324856578844164	0	0	0	0	0	0
1	0.225358843608109	0	0	0	0	0	0
1	0.196371948775364	0	0	0	0	0	0
1	0.227408791585773	0	0	0	0	0	0
1	0.227700525002234	0	0	0	0	0	0
1	0.42394108592759	0	0	0	0	0	0
```

Inspect the UK Biobank annotation files:

```bash
zcat $V2DX_DIR/annots/ukbb.22.annot.gz  | head
```

Example output:

```text
base	v2d	ENCFF790XQN_blood_bodily_fluid_naive_thymus_derived_CD4_positive_alpha_beta_T_cell	ENCFF790XQN_blood_bodily_fluid_naive_thymus_derived_CD4_positive_alpha_beta_T_cell_v2d	Zhang_Glutamatergic_2	Zhang_Glutamatergic_2_v2d	Zhang_Fetal_Excitatory_Neuron_3	Zhang_Fetal_Excitatory_Neuron_3_v2d
1	0.140633603489702	0	0	0	0	0	0
1	0.140633603489702	0	0	0	0	0	0
1	0.140633603489702	0	0	0	0	0	0
1	0.140633603489702	0	0	0	0	0	0
1	0.140633603489702	0	0	0	0	0	0
1	0.565471293070404	0	0	0	0	0	0
1	0.140633603489702	0	0	0	0	0	0
1	0.140633603489702	0	0	0	0	0	0
1	0.565471293070404	0	0	0	0	0	0
```

## 3. Compute LD Scores Using LDSC

The following command computes LD scores for the tutorial annotations.

```bash
for CHR in {1..22}; do
    if [ ! -f "$V2DX_DIR/annots/1000G.$CHR.l2.ldscore.gz" ]; then
        echo "$CHR";
        $LDSC \
                      --l2 \
                      --bfile $REF_DIR/1000G_EUR_Phase3_plink/1000G.EUR.QC.$CHR \
                      --ld-wind-cm 1 \
                      --annot $V2DX_DIR/annots/1000G.$CHR.annot.gz \
                      --thin-annot \
                      --out $V2DX_DIR/annots/1000G.$CHR \
                      --print-snps $REF_DIR/hm3_no_MHC.list.txt;
    fi;
done
```

Outputs:

- LD scores will be generated in `$V2DX_DIR/annots/1000G.*.l2.ldscore.gz`


Example output:

```bash
zcat $V2DX_DIR/annots/1000G.22.l2.ldscore.gz | head
```

```text
CHR	SNP	BP	baseL2	v2dL2	ENCFF790XQN_blood_bodily_fluid_naive_thymus_derived_CD4_positive_alpha_beta_T_cellL2	ENCFF790XQN_blood_bodily_fluid_naive_thymus_derived_CD4_positive_alpha_beta_T_cell_v2dL2	Zhang_Glutamatergic_2L2	Zhang_Glutamatergic_2_v2dL2	Zhang_Fetal_Excitatory_Neuron_3L2	Zhang_Fetal_Excitatory_Neuron_3_v2dL2
22	rs9617528	16061016	7.336	1.591	0.000	0.000	0.000	0.000	0.000	0.000
22	rs4911642	16504399	62.101	21.955	-0.003	-0.022	0.000	0.000	0.000	0.000
22	rs7287144	16886873	152.659	51.039	0.006	0.066	0.000	0.000	0.000	0.000
22	rs5748662	16892858	130.432	43.412	0.002	0.030	0.000	0.000	0.000	0.000
22	rs5994034	16894090	50.683	16.225	0.106	0.986	0.000	0.000	0.000	0.000
22	rs4010554	16894264	154.912	51.744	0.001	0.021	0.000	0.000	0.000	0.000
22	rs4010558	16896762	155.911	51.985	0.001	0.023	0.000	0.000	0.000	0.000
22	rs3954571	16953560	170.452	50.255	0.029	0.267	-0.007	-0.007	0.000	0.000
```

## 4. Estimate V2Dx Coefficients Using S-LDSC

We use a leave-even/odd-chromosome framework to avoid information leakage between coefficient estimation and score construction.

### 4a. Split GWAS Summary Statistics

Create headers:

```bash
zcat $V2DX_DIR/PASS.Schizophrenia.Trubetskoy2022.sumstats.gz | head -1 > $V2DX_DIR/trait.odd.sumstats
cp $V2DX_DIR/trait.odd.sumstats $V2DX_DIR/trait.even.sumstats
```

Extract odd/even SNPs:

```bash
zcat $V2DX_DIR/PASS.Schizophrenia.Trubetskoy2022.sumstats.gz | grep -w -f $REF_DIR/odd.list - >> $V2DX_DIR/trait.odd.sumstats
zcat $V2DX_DIR/PASS.Schizophrenia.Trubetskoy2022.sumstats.gz | grep -w -f $REF_DIR/even.list - >> $V2DX_DIR/trait.even.sumstats
```

Compress files:

```bash
gzip $V2DX_DIR/trait.*.sumstats
```

### 4b. Run S-LDSC

```bash
OPTIONS="--overlap-annot --print-coefficients";
FREQ="--frqfile-chr $REF_DIR/1000G_Phase3_frq/1000G.EUR.QC.";
WEIGHTS="--w-ld-chr $REF_DIR/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC.";
# Odd chromosomes:
$LDSC \
                    --h2 $V2DX_DIR/trait.odd.sumstats.gz \
                    --ref-ld-chr $V2DX_DIR/annots/1000G. \
                    $FREQ $WEIGHTS $OPTIONS \
                    --out $V2DX_DIR/trait.odd
# Even chromosomes:
$LDSC \
                    --h2 $V2DX_DIR/trait.even.sumstats.gz \
                    --ref-ld-chr $V2DX_DIR/annots/1000G. \
                    $FREQ $WEIGHTS $OPTIONS \
                    --out $V2DX_DIR/trait.even
```

Outputs:

- V2Dx coefficients in `$V2DX_DIR/trait.{even,odd}.results`

## 5. Generate Final V2Dx Scores

The leave-even/odd framework ensures that coefficients used to compute V2Dx scores are estimated independently from the chromosomes being scored.

Specifically, `create_v2dx.r` applies:

- coefficients from `trait.odd.results` to even chromosomes
- coefficients from `trait.even.results` to odd chromosomes

```bash
Rscript $REF_DIR/create_v2dx.r $REF_DIR $V2DX_DIR
```

Outputs:

- V2Dx scores in LDSC format in `$V2DX_DIR/V2Dx/1000G/v2dx.*`
- V2Dx scores for approximately 20M UK Biobank variants in `$V2DX_DIR/V2Dx/ukbb/v2dx.*`

Inspect the 1000G-format V2Dx scores:

```bash
zcat $V2DX_DIR/V2Dx/1000G/v2dx.22.txt.gz | head
```

Example output:

```text
CHR	RSID	POS	V2D	V2Dx
22	rs587616822	16050840	0.318078749655231	0.441650272017
22	rs62224609	16051249	0.196371948775364	0.345674085096999
22	rs587646183	16052463	0.365908260521204	0.479367918376762
22	rs139918843	16052684	0.324856578844164	0.446995168082347
22	rs587743102	16052837	0.225358843608109	0.368532722548525
22	rs376238049	16052962	0.196371948775364	0.345674085096999
22	rs200777521	16052986	0.227408791585773	0.370149281268812
22	rs587710177	16053139	0.227700525002234	0.37037933793538
22	rs587701155	16053254	0.42394108592759	0.525131747993783
```

Inspect the UK Biobank-format V2Dx scores:

```bash
zcat tutorial_v2dx_scz/V2Dx/ukbb/v2dx.22.txt.gz | head
```

Example output:

```text
CHR	BP	SNP	ID	V2D	V2Dx
22	16050115	rs587755077	22:16050115_G_A	0.140633603489702	0.217217362608461
22	16050213	rs587654921	22:16050213_C_T	0.140633603489702	0.217217362608461
22	16050527	rs587769434	22:16050527_C_A	0.140633603489702	0.217217362608461
22	16050678	rs2186465	22:16050678_C_T	0.140633603489702	0.217217362608461
22	16050739	22:16050739_TA_T	22:16050739_TA_T	0.140633603489702	0.217217362608461
22	16050822	rs12172168	22:16050822_G_A	0.565471293070404	0.748540590509574
22	16050840	rs587616822	22:16050840_C_G	0.140633603489702	0.217217362608461
22	16050847	rs587702478	22:16050847_T_C	0.140633603489702	0.217217362608461
22	16051249	rs62224609	22:16051249_T_C	0.565471293070404	0.748540590509574
```

### Which Output Format Should I Use?

The `1000G` output is intended for analyses that use the 1000 Genomes/S-LDSC reference SNP set and LDSC-style files. This format is useful for inspecting the trait-specific scores on the same SNP universe used to estimate LD scores and S-LDSC coefficients.

The `ukbb` output contains V2Dx scores for the larger UK Biobank/PolyFun-style variant panel, approximately 20M variants. This format is the relevant one for genome-wide variant prioritization and downstream analyses that need V2Dx scores on the broader variant set.

## Cleanup

After confirming that the output files were created correctly, annotation files can be removed:

```bash
rm -rf $V2DX_DIR/annots/
```

## Final Notes

- This tutorial uses schizophrenia as an example, but the same workflow can be applied to other traits with GWAS summary statistics and trait-relevant cell-type-specific annotations.
- We provide the codes without a cluster configuration as it may differ depending on your setting. The full workflow typically uses SLURM for LDSC-related analyses and completes within a few hours on standard compute nodes.