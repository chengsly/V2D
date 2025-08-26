import numpy as np
import pandas as pd
import sys

mode = sys.argv[-1]

def is_common_row(df):
    annot_sum = 0 
    annot_sum += df["MAFbin_frequent_1"]
    annot_sum += df["MAFbin_frequent_2"]
    annot_sum += df["MAFbin_frequent_3"]
    annot_sum += df["MAFbin_frequent_4"]
    annot_sum += df["MAFbin_frequent_5"]
    annot_sum += df["MAFbin_frequent_6"]
    annot_sum += df["MAFbin_frequent_7"]
    annot_sum += df["MAFbin_frequent_8"]
    annot_sum += df["MAFbin_frequent_9"]
    annot_sum += df["MAFbin_frequent_10"]
    return annot_sum >= 1

def is_lowfreq_row(df):
    annot_sum = 0 
    annot_sum += df["MAFbin_lowfreq_1"]
    annot_sum += df["MAFbin_lowfreq_2"]
    annot_sum += df["MAFbin_lowfreq_3"]
    annot_sum += df["MAFbin_lowfreq_4"]
    annot_sum += df["MAFbin_lowfreq_5"]
    annot_sum += df["MAFbin_lowfreq_6"]
    annot_sum += df["MAFbin_lowfreq_7"]
    annot_sum += df["MAFbin_lowfreq_8"]
    annot_sum += df["MAFbin_lowfreq_9"]
    annot_sum += df["MAFbin_lowfreq_10"]
    return annot_sum >= 1


def is_common_col(col):
    if "SNP" in col:
        return True
    if col == "MAFbin_frequent_1":
        return False
    elif ("MAFbin_frequent" in col or 
        "_common" in col or col == "Y" or
        col == "CHR" or col == "BP" or col == "denom_p"):
        return True
    elif "Y_" in col:
        return True
    else:
        return False 

def is_lowfreq_col(col):
    if "SNP" in col:
        return True
    if "MAFbin_lowfreq_1" == col:
        return False
    elif ("lowfreq" in col or 
        col == "Y" or
        col == "CHR" or col == "BP" or col == "denom_p"):
        return True
    elif "Y_" in col: # sepcific for this script
        return True
    else:
        return False 


baseline_file = "/project/gazal_569/helen/h2ML_prior/baselineLF.annot"
baseline_data =  pd.read_csv(baseline_file, sep = '\t').reset_index()

with open("../trait_indep.txt", "r") as f:
    traits = [elem.rstrip("\n") for elem in f.readlines()]


def obtain_mean_y(baseline_data=baseline_data.copy(),
                  traits=traits, 
                  prior_mode=True):
    for trait_name in traits:        
        root_path = "/project/gazal_569/DATA/Weissbrod_2020/beta2/"
        if not prior_mode:
            root_path = "/project/gazal_569/DATA/Weissbrod_2020/beta2_noprior/" 
        beta2_fname = root_path + trait_name + ".all.txt"
        beta2_data = pd.read_csv(beta2_fname, sep = '\t', header= None)
        baseline_data[f'Y_{trait_name}'] = beta2_data[3]

    if mode == "common":
        baseline_data = baseline_data.loc[is_common_row(baseline_data)]
        common_cols = [c for c in baseline_data.columns if is_common_col(c)]
        baseline_data = baseline_data[common_cols]
    else:
        baseline_data = baseline_data.loc[is_lowfreq_row(baseline_data)]
        lowfreq_cols = [c for c in baseline_data.columns if is_lowfreq_col(c)]
        baseline_data = baseline_data[lowfreq_cols]

    # step 4: remove MCH region
    baseline_data = baseline_data.loc[~((baseline_data["CHR"]== 6) &
                                            (baseline_data["BP"]>=25000000) &
                                            (baseline_data["BP"]<=34000000))]
    # QC 
    for trait_name in traits:                                     
        column_name = f"Y_{trait_name}"
        Ys = baseline_data[column_name]

        # # step 5: set to NaN if out of bound
        
        Y_sum = baseline_data[column_name].sum()
        Ys.values[Ys.values > 0.01 * Y_sum] = np.nan
        # step 5: do it twice
        Y_sum = Ys.sum(skipna=True)
        Ys.values[Ys.values > 0.01 * Y_sum] = np.nan

        # step 6: normalize and ignore NaNs
        Ysum = Ys.sum(skipna=True)
        Ys = Ys / Ysum
        baseline_data[column_name] = Ys 
    # step 7: row-wise mean for beta2 
    y_cols = [column for column in baseline_data.columns if "Y_" in column]
    y_mean = baseline_data[y_cols].mean(axis=1,skipna=True) # mean ignoring NaN values

    # step 8: renormalize beta2
    y_mean = y_mean / y_mean.sum(skipna=True)
    print(baseline_data.shape)
    baseline_data.drop(columns=y_cols, inplace=True)

    return y_mean

prior_y_mean = obtain_mean_y()
noprior_y_mean = obtain_mean_y(prior_mode=False)

# make sure same number of rows after processing
if mode == "common":
    baseline_data = baseline_data.loc[is_common_row(baseline_data)]
    common_cols = [c for c in baseline_data.columns if is_common_col(c)]
    baseline_data = baseline_data[common_cols]
else:
    baseline_data = baseline_data.loc[is_lowfreq_row(baseline_data)]
    lowfreq_cols = [c for c in baseline_data.columns if is_lowfreq_col(c)]
    baseline_data = baseline_data[lowfreq_cols]

# step 4: remove MCH region
baseline_data = baseline_data.loc[~((baseline_data["CHR"]== 6) &
                                        (baseline_data["BP"]>=25000000) &
                                        (baseline_data["BP"]<=34000000))]
baseline_data["beta2_prior"] = prior_y_mean 
baseline_data["beta2_noprior"] = noprior_y_mean

baseline_data = baseline_data[["CHR", "BP", "SNP", "beta2_prior", "beta2_noprior"]]
baseline_data.to_csv(f"df_annot_mean_snps_{mode}.csv")