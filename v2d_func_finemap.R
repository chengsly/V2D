# Adapted from code developed for V2D:
# Fabiha et al., "A consensus variant-to-function score to
# functionally prioritize variants for disease"
# [https://www.biorxiv.org/content/10.1101/2024.11.07.622307v2.full]
#
# Original code: <https://github.com/Deylab999MSKCC/V2D/blob/main/SuSIE_finemap/V2D_func_finemap_EMS.R>
#
# This version was modified by Cheng et al. for V2D analyses.
# Modifications include:
# - Using raw scores rather than binarizing them
# - Changing every cv2f/cV2F to V2D
#
# The original cV2F code is distributed under the GNU General Public License.
# This modified version is distributed under the same license.
#
# Copyright (c) 2024 Tabassum Fabiha (original code)
# Modifications copyright (c) 2026 Cheng et al.

library(data.table)
library(R.utils)
library(optparse)
library(zoo)

options(echo=TRUE) # if you want see commands in output file
args <- commandArgs(trailingOnly = TRUE)
print(args)

option_list <- list(
  make_option("--trait", type="character", default = "UKBB.FEV1FVC.SuSiE", help="Biosample names"),
  make_option("--finemappath", type="character", default = "../data/ukbb-finemapping",
	      help="Path to directory of finemapped traits"),
#  make_option("--bimpath", type="character", default = "../data/1000G_BIMS_hg38/1000G.EUR.QC.",
#	      help="Path and prefix of the bimfile of the BIMFILE"),
  make_option("--V2Dpath", type="character", default = "../results/score/mwe_baseline.",
	      help="Path and prefix of V2D scores files"),
  make_option("--V2Did", type = "character", default = "SNP",
          help = "Column name containing variant IDs to match with fine-mapping results"),
  make_option("--V2Dscore", type = "character", default = "V2D",
          help = "Column name containing variant prioritization scores"),
  make_option("--output_finemap", type="character", default = "../results/SuSIE_finemap/EMS_all_V2D",
	      help="Output path to directory of results")
)

opt <- parse_args(OptionParser(option_list=option_list))
dput(opt)

traitname = opt$trait
dff = read.delim2(paste0(opt$finemappath, "/", traitname, ".tsv.bgz"))

V2D_tabb = c()
for(numchr in unique(dff$chromosome)){
  V2D_scores = data.frame(fread(paste0(opt$V2Dpath, numchr, ".txt.gz")))
  V2D_tabb = rbind(V2D_tabb, V2D_scores)
  cat("We are at chr:", numchr, "\n")
}
V2D_tabb = data.frame(
    SNP   = V2D_tabb[[opt$V2Did]],
    score = V2D_tabb[[opt$V2Dscore]]
)

u_regions = unique(dff$region)

out_merged = c()
for(numu in 1:length(u_regions)){
  dff_temp = dff[which(dff$region == u_regions[numu]), ]
  dff_temp2 = dff_temp[which(dff_temp$cs_id != -1), ]
  if(nrow(dff_temp2) == 0){
    cat("We are at region:", numu, "\n")
    next
  }
  xx = cbind.data.frame(as.numeric(dff_temp2$alpha1),
                        as.numeric(dff_temp2$alpha2),
                        as.numeric(dff_temp2$alpha3),
                        as.numeric(dff_temp2$alpha4),
                        as.numeric(dff_temp2$alpha5),
                        as.numeric(dff_temp2$alpha6),
                        as.numeric(dff_temp2$alpha7),
                        as.numeric(dff_temp2$alpha8),
                        as.numeric(dff_temp2$alpha9),
                        as.numeric(dff_temp2$alpha10))

  V2D_temp = V2D_tabb$score[match(dff_temp2$rsid, V2D_tabb$SNP)]
  Cz = zoo(V2D_temp)
  Cz_approx2 <- na.approx(Cz, na.rm=FALSE, rule=2)
  Cz_approx = Cz_approx2
  # if(length(Cz_approx2) ==1){
  #   Cz_approx = (exp(Cz_approx2))/max(exp(Cz_approx2))
  # }else{
  #   Cz_approx = (exp(Cz_approx2)-min(exp(Cz_approx2)))/(max(exp(Cz_approx2)) - min(exp(Cz_approx2)))
  # }

  xx2 = cbind.data.frame(as.numeric(dff_temp2$alpha1)*Cz_approx,
                        as.numeric(dff_temp2$alpha2)*Cz_approx,
                        as.numeric(dff_temp2$alpha3)*Cz_approx,
                        as.numeric(dff_temp2$alpha4)*Cz_approx,
                        as.numeric(dff_temp2$alpha5)*Cz_approx,
                        as.numeric(dff_temp2$alpha6)*Cz_approx,
                        as.numeric(dff_temp2$alpha7)*Cz_approx,
                        as.numeric(dff_temp2$alpha8)*Cz_approx,
                        as.numeric(dff_temp2$alpha9)*Cz_approx,
                        as.numeric(dff_temp2$alpha10)*Cz_approx)

  ucs = unique(dff_temp2$cs_id)
  for(num_cs in 1:length(ucs)){

    idx= which(dff_temp2$cs_id == ucs[num_cs])
    xx_filtered = xx[idx, ]
    pip2 = apply(xx_filtered, 1, function(z) return(1 - prod(1-as.numeric(z))))

    effects_to_choose = as.numeric(which(colSums(xx_filtered) > 0.90))
    xx3 = xx2[idx, ]
    xx4 = sweep(xx3, 2, colSums(xx3), "/")
    xx5 = cbind.data.frame(xx4[,effects_to_choose])
    pip3 = apply(xx5, 1, function(z) return(1 - prod(1-as.numeric(z))))

    V2D_out = Cz_approx[idx]
    dff_temp3 = cbind.data.frame(dff_temp2[idx, ], pip2, pip3, V2D_out)

    numm = length(which(cumsum(sort(xx4[,effects_to_choose], decreasing = T)) < 0.95))+1
    denn = length(which(cumsum(sort(xx[,effects_to_choose], decreasing = T)) < 0.95))+1
    cs_shrink = (1 - numm/denn)
    dff_temp3$cs_shrink_percentage = cs_shrink

    colnames(dff_temp3) = c(colnames(dff_temp2), "pip", "V2D.pip",
                            "V2D", "CS.shrink.percent")
    out_merged = rbind(out_merged, dff_temp3)
  }
  cat("We are at region:", numu, "\n")
}
colnames(out_merged)[33] = paste0(opt$V2Dscore,".pip")
colnames(out_merged)[34] = opt$V2Dscore
write.table(out_merged, file = paste0(opt$output_finemap, "/", traitname, ".", opt$V2Dscore,".txt"),
            row.names=F, col.names=T, sep = "\t", quote=F)
