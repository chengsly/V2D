import numpy as np
import os 
import math
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


# step 1: read X 
baseline_file = "/project/gazal_569/helen/h2ML_prior/baselineLF.annot"
baseline_data =  pd.read_csv(baseline_file, sep = '\t').reset_index()

with open("../trait_indep.txt", "r") as f:
    traits = [elem.rstrip("\n") for elem in f.readlines()]
# step 2: read all Ys 
# for trait_name in traits:
#     print(f"current trait is: {trait_name}")
#     beta2_fname = "/project/gazal_569/DATA/Weissbrod_2020/beta2/" + trait_name + ".all.txt"
#     beta2_data = pd.read_csv(beta2_fname, sep = '\t', header= None)
#     beta2_data.columns = ["CHR","BP","SNP",f"Y_{trait_name}"]
#     baseline_data = baseline_data.merge(beta2_data, on=["SNP", "BP", "CHR"])

for trait_name in traits:
    print(f"current trait is: {trait_name}")
    beta2_fname = "/project/gazal_569/DATA/Weissbrod_2020/beta2/" + trait_name + ".all.txt"
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

baseline_data["Y"] = y_mean 

# generate all data
X, Y = baseline_data.iloc[:, 3:-1].to_numpy(), baseline_data.iloc[:, -1].to_numpy()
if mode == "common":
    np.save("X_common_mean15/X_common.npy", X)
    np.save("Y_common_mean15/Y_common.npy", Y)
else:
    np.save("X_lowfreq_mean15/X_lowfreq.npy", X) 
    np.save("Y_lowfreq_mean15/Y_lowfreq.npy", Y)


if mode == "common":
   baseline_data.to_csv("baseline_mean_common.annot", sep="\t")
   baseline_data["SNP"].to_csv("baseline_mean_common_snps.annot", sep=",")
else:
   baseline_data.to_csv("baseline_mean_lowfreq.annot", sep="\t")
   baseline_data["SNP"].to_csv("baseline_mean_lowfreq_snps.annot", sep=",")
   

odd_chrs = [1,3,5,7,9,11,13,15,17,19,21]
even_chrs = [2,4,6,8,10,12,14,16,18,20,22]

# save individuals
for i in range(1,23):
    chr_i_x = baseline_data.loc[baseline_data.CHR == i].to_numpy()[:, 3:-1]
    chr_i_y = baseline_data.loc[baseline_data.CHR == i].to_numpy()[:, -1]
    chr_noti_x = baseline_data.loc[baseline_data.CHR != i].to_numpy()[:, 3:-1]
    chr_noti_y = baseline_data.loc[baseline_data.CHR != i].to_numpy()[:, -1]

    if mode == "lowfreq":
        np.save(f"X_lowfreq_mean15/X_lowfreq_CHR={i}.npy", chr_i_x)
        np.save(f"Y_lowfreq_mean15/Y_lowfreq_CHR={i}.npy", chr_i_y)
        np.save(f"X_lowfreq_mean15/X_lowfreq_not_CHR={i}.npy", chr_noti_x)
        np.save(f"Y_lowfreq_mean15/Y_lowfreq_not_CHR={i}.npy", chr_noti_y)
    else:
        np.save(f"X_common_mean15/X_common_CHR={i}.npy", chr_i_x)
        np.save(f"Y_common_mean15/Y_common_CHR={i}.npy", chr_i_y)
        np.save(f"X_common_mean15/X_common_not_CHR={i}.npy", chr_noti_x)
        np.save(f"Y_common_mean15/Y_common_not_CHR={i}.npy", chr_noti_y)

# odd and even
data_odd = baseline_data.loc[baseline_data.CHR.isin(odd_chrs)].to_numpy()
data_even = baseline_data.loc[baseline_data.CHR.isin(even_chrs)].to_numpy()

# save 
X_even, Y_even = data_even[:, 3:-1], data_even[:,-1]
X_odd, Y_odd = data_odd[:, 3:-1], data_odd[:,-1]
print(f"X_even = {X_even.shape}, Y_even = {Y_even.shape}, X_odd = {X_odd.shape}, Y_odd = {Y_odd.shape}")

if mode == "lowfreq":
    np.save("X_lowfreq_mean15/X_lowfreq_even.npy", X_even)
    np.save("Y_lowfreq_mean15/Y_lowfreq_even.npy", Y_even)
    np.save("X_lowfreq_mean15/X_lowfreq_odd.npy", X_odd)
    np.save("Y_lowfreq_mean15/Y_lowfreq_odd.npy", Y_odd)
else:
    np.save("X_common_mean15/X_common_even.npy", X_even)
    np.save("Y_common_mean15/Y_common_even.npy", Y_even)
    np.save("X_common_mean15/X_common_odd.npy", X_odd)
    np.save("Y_common_mean15/Y_common_odd.npy", Y_odd)
