# Performing functionally-informed fine-mapping using V2D and V2Dx

This tutorial demonstrates how to leverage V2D and V2Dx as priors to re-weight Height UK Biobank fine-mapping results.

This tutorial uses European LD scores and is appropriate for European-ancestry GWAS. Other ancestries require matched LD and reference files.

## Requirements
- Download Reference Files and Scripts from the V2Dx tutorial [https://github.com/chengsly/V2D/blob/main/tutorial.md].

- R 4.4 or later
- R libraries data.table R.utils optparse zoo
- Python 3 (to compute V2Dx scores)

R and Python should be available from the command line. In practice, the tutorial assumes that commands such as `Rscript XX.R` and `python XX.py` can be run from the active shell environment.


## Main Steps

1. Download files fine-mapping data (fine-mapping results, and annotations and sumstats for V2Dx) .
2. Perform functionally informed fine-mapping using V2D.
3. Perform functionally informed fine-mapping using V2Dx.

## 1. Download Reference Files and Scripts

```bash
mkdir Height_finemapping
cd Height_finemapping
wget https://zenodo.org/records/20499413/V2Dx.tgz?download=1 -O V2Dx.tgz
wget https://zenodo.org/records/20499413/tutorial_finemapping_height.tgz?download=1 -O tutorial_finemapping_height.tgz
```

Unpack files:

```bash
tar -xvzf V2Dx.tgz
tar -xvzf tutorial_finemapping_height.tgz
```

Remove the compressed file:

```bash
rm V2Dx.tgz
rm tutorial_finemapping_height.tgz
```

You should now have two directories in your `Height_finemapping/` folder:

- `v2dx_ref/`, containing V2D scores, reference files for S-LDSC analyses, and scripts to compute V2Dx scores
- `tutorial_finemapping_height/`, containing:
  - Height fine-mapping results: `finemapping/UKBB.Height.SuSiE.tsv.bgz`
  - Height annotations for V2Dx (created using step 2 of V2Dx tutorial): `annots/*`
  - Height GWAS summary statistics for V2Dx: `sumstats/Height.sumstats.gz`
  - Script to perform fine-mapping: `v2d_func_finemap.R`


Define environment variables for downstream analyses.

Generic LDSC path:

```bash
LDSC="python /path_to_ldsc/ldsc.py"
```

Gazal lab example:

```bash
LDSC="python /project2/gazal_569/DATA/ldsc/ldsc-2.0.1/ldsc.py"
```

Reference directory for any V2D and V2Dx analysis:

```bash
V2D_DIR="v2dx_ref"
```

Directory with fine-mapping scripts and trait-specific data:

```bash
FM_DIR="tutorial_finemapping_height"
```

## 2. Perform functionally informed fine-mapping using V2D. 

The script below reweight SuSiE posterior probabilities from the file tutorial_finemapping_height/finemapping/UKBB.Height.SuSiE.tsv.bgz, with V2D scores in files v2dx_ref/V2D/1000G_EUR/v2d_1000G.. Variants are matched between the V2D and fine-mapping files using the variant-ID column specified by --V2Did, and scores are read from the column specified by --V2Dscore.
This script was adapted from cV2F_func_finemap_EMS.R, developed by Tabassum Fabiha and Kushal Dey for cV2F ([preprint](https://www.biorxiv.org/content/10.1101/2024.11.07.622307v2); [code](https://github.com/Deylab999MSKCC/cv2f/blob/main/SuSIE_finemap/cV2F_func_finemap_EMS.R)). We thank the cV2F authors for making their implementation publicly available. Our modified implementation adapts the original procedure to V2D/V2Dx scores. The original and modified code are distributed under the GNU General Public License.

```bash
mkdir results_v2d
Rscript $FM_DIR/v2d_func_finemap.R \
          --trait UKBB.Height.SuSiE \
          --finemappath $FM_DIR/finemapping \
          --V2Dpath $V2D_DIR/V2D/1000G_EUR/v2d_1000G. \
          --V2Did RSID \
          --V2Dscore V2D \
          --output_finemap results_v2d
```

Outputs:

- fine-mapping results in file `results_v2d/UKBB.Height.SuSiE.V2D.txt`

Inspect fine-mapping results:

```bash
head results_v2d/UKBB.Height.SuSiE.V2D.txt | cut -f6,32-34
```

Example output:

```text
rsid	pip	V2D.pip	V2D
rs75267490	0.040514700605134	0.02411092167266	1.03766448252645
rs3109210	0.043229704733426	0.0279760910277148	1.12838178385859
rs115173026	0.0556406519168324	0.192837312474129	6.04275136270287
rs3121553	0.0172584619359627	0.031323251291332	3.16527411437422
rs2799064	0.0248635164057694	0.0119681169362669	0.839384707209619
rs2001744	0.300215003702945	0.277542165693999	1.61171318681599
rs2341363	0.0629281792885735	0.0430625751281126	1.19311256525234
rs2710890	0.0723811471587477	0.063929696732849	1.53991946447
rs2710889	0.313474912375072	0.309357957035223	1.72047849317531
```


## 3. Perform functionally informed fine-mapping using V2Dx

The detailed pipeline to compute V2Dx scores is provided here [URL].

### 3a. Split GWAS Summary Statistics

Create headers:

```bash
zcat $FM_DIR/sumstats/Height.sumstats.gz | head -1 > $FM_DIR/sumstats/Height.odd.sumstats
cp $FM_DIR/sumstats/Height.odd.sumstats $FM_DIR/sumstats/Height.even.sumstats
```

Extract odd/even SNPs:

```bash
zcat $FM_DIR/sumstats/Height.sumstats.gz | grep -w -f $V2D_DIR/odd.list - >> $FM_DIR/sumstats/Height.odd.sumstats
zcat $FM_DIR/sumstats/Height.sumstats.gz | grep -w -f $V2D_DIR/even.list - >> $FM_DIR/sumstats/Height.even.sumstats
```

Compress files:

```bash
gzip $FM_DIR/sumstats/Height.*.sumstats
```

### 3b. Run S-LDSC

```bash
OPTIONS="--overlap-annot --print-coefficients";
FREQ="--frqfile-chr $V2D_DIR/1000G_Phase3_frq/1000G.EUR.QC.";
WEIGHTS="--w-ld-chr $V2D_DIR/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC.";
# Odd chromosomes:
$LDSC \
                    --h2 $FM_DIR/sumstats/Height.odd.sumstats.gz \
                    --ref-ld-chr $FM_DIR/annots/Height. \
                    $FREQ $WEIGHTS $OPTIONS \
                    --out $FM_DIR/trait.odd
# Even chromosomes:
$LDSC \
                    --h2 $FM_DIR/sumstats/Height.even.sumstats.gz \
                    --ref-ld-chr $FM_DIR/annots/Height. \
                    $FREQ $WEIGHTS $OPTIONS \
                    --out $FM_DIR/trait.even
```

Outputs:

- V2Dx coefficients in `$FM_DIR/trait.{even,odd}.results`

## 3c. Generate Height V2Dx Scores

The leave-even/odd framework ensures that coefficients used to compute V2Dx scores are estimated independently from the chromosomes being scored.

Specifically, `create_v2dx.r` applies:

- coefficients from `Height.odd.results` to even chromosomes
- coefficients from `Height.even.results` to odd chromosomes

```bash
Rscript $V2D_DIR/create_v2dx.r $V2D_DIR $FM_DIR
```

Outputs:

- V2Dx scores in LDSC format in `$FM_DIR/V2Dx/1000G/v2dx.*`
- V2Dx scores for approximately 20M UK Biobank variants in `$FM_DIR/V2Dx/ukbb/v2dx.*`

Inspect the 1000G-format V2Dx scores:

```bash
zcat $FM_DIR/V2Dx/1000G/v2dx.1.txt.gz | head
```

Example output:

```text
CHR	RSID	POS	V2D	V2Dx
1	rs575272151	11008	1.00478220594496	0.768087888290296
1	rs544419019	11012	1.00488510837983	0.76818306525673
1	rs540538026	13110	0.186092855661174	0.0108621830796367
1	rs62635286	13116	0.186092855661174	0.0108621830796367
1	rs200579949	13118	0.186092855661174	0.0108621830796367
1	rs531730856	13273	0.186092855661174	0.0108621830796367
1	rs554008981	13550	0.186092855661174	0.0108621830796367
1	rs546169444	14464	1.22464022623925	0.971439922130549
1	rs531646671	14599	1.43782364094006	1.16861845081189
```

## 3d. Perform fine-mapping using V2Dx scores

```bash
Rscript $FM_DIR/v2d_func_finemap.R \
          --trait UKBB.Height.SuSiE \
          --finemappath $FM_DIR/finemapping \
          --V2Dpath $FM_DIR/V2Dx/1000G/v2dx. \
          --V2Did RSID \
          --V2Dscore V2Dx \
          --output_finemap results_v2d
```

Outputs:

- fine-mapping results in file `results_v2d/UKBB.Height.SuSiE.V2Dx.txt`

Inspect fine-mapping results:

```bash
head results_v2d/UKBB.Height.SuSiE.V2Dx.txt | cut -f6,32-34 
```

Example output:

```text
rsid	pip	V2Dx.pip	V2Dx
rs75267490	0.040514700605134	0.0148989284392568	0.798501506184213
rs3109210	0.043229704733426	0.017568023706142	0.882408146049781
rs115173026	0.0556406519168324	0.4680121191755	18.2632607567123
rs3121553	0.0172584619359627	0.0219831382077272	2.76637943611021
rs2799064	0.0248635164057694	0.00704268689145571	0.615107716291559
rs2001744	0.300215003702945	0.183838588376521	1.32945312388907
rs2341363	0.0629281792885735	0.0273099324495066	0.94227922116848
rs2710890	0.0723811471587477	0.0421063304086148	1.26304936278689
rs2710889	0.313474912375072	0.20648406936758	1.43005280285379
```

