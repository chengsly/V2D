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
    if "MAFbin_lowfreq_1" in col:
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
with open("../trait_indep.txt", "r") as f:
    traits = [elem.rstrip("\n") for elem in f.readlines()]

for trait_name in traits:
    os.makedirs(f"X_common_traits/{trait_name}", exist_ok=True)
    os.makedirs(f"Y_common_traits/{trait_name}", exist_ok=True)
    os.makedirs(f"X_lowfreq_traits/{trait_name}", exist_ok=True)
    os.makedirs(f"Y_lowfreq_traits/{trait_name}", exist_ok=True)
    baseline_data =  pd.read_csv(baseline_file, sep = '\t').reset_index()
    print(f"current trait is: {trait_name}")
    beta2_fname = "/project/gazal_569/DATA/Weissbrod_2020/beta2/" + trait_name + ".all.txt"
    beta2_data = pd.read_csv(beta2_fname, sep = '\t', header= None)
    baseline_data['Y'] = beta2_data[3]

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
    Y_sum = baseline_data["Y"].sum()
    baseline_data = baseline_data.loc[baseline_data["Y"] <= 0.01 * Y_sum]
    Y_sum = baseline_data["Y"].sum()
    baseline_data = baseline_data.loc[baseline_data["Y"] <= 0.01 * Y_sum]
    # # step 5: set to NaN if out of bound
    baseline_data["Y"] = baseline_data["Y"] / baseline_data["Y"].sum() 
    baseline_data.head(10).to_csv(f"test_baseline_{trait_name}.csv")


    # generate all data
    X, Y = baseline_data.iloc[:, 3:-1].to_numpy(), baseline_data.iloc[:, -1].to_numpy()
    if mode == "common":
        np.save(f"X_common_traits/{trait_name}/X_common.npy", X)
        np.save(f"Y_common_traits/{trait_name}/Y_common.npy", Y)
    else:
        np.save(f"X_lowfreq_traits/{trait_name}/X_lowfreq.npy", X) 
        np.save(f"Y_lowfreq_traits/{trait_name}/Y_lowfreq.npy", Y)

    odd_chrs = [1,3,5,7,9,11,13,15,17,19,21]
    even_chrs = [2,4,6,8,10,12,14,16,18,20,22]

    # save individuals
    for i in range(1,23):
        chr_i_x = baseline_data.loc[baseline_data.CHR == i].to_numpy()[:, 3:-1]
        chr_i_y = baseline_data.loc[baseline_data.CHR == i].to_numpy()[:, -1]
        chr_noti_x = baseline_data.loc[baseline_data.CHR != i].to_numpy()[:, 3:-1]
        chr_noti_y = baseline_data.loc[baseline_data.CHR != i].to_numpy()[:, -1]

        if mode == "lowfreq":
            np.save(f"X_common_traits/{trait_name}/X_lowfreq_CHR={i}.npy", chr_i_x)
            np.save(f"X_common_traits/{trait_name}/Y_lowfreq_CHR={i}.npy", chr_i_y)
            np.save(f"X_common_traits/{trait_name}/X_lowfreq_not_CHR={i}.npy", chr_noti_x)
            np.save(f"X_common_traits/{trait_name}/Y_lowfreq_not_CHR={i}.npy", chr_noti_y)
        else:
            np.save(f"X_common_traits/{trait_name}/X_common_CHR={i}.npy", chr_i_x)
            np.save(f"Y_common_traits/{trait_name}/Y_common_CHR={i}.npy", chr_i_y)
            np.save(f"X_common_traits/{trait_name}/X_common_not_CHR={i}.npy", chr_noti_x)
            np.save(f"Y_common_traits/{trait_name}/Y_common_not_CHR={i}.npy", chr_noti_y)

    # odd and even
    data_odd = baseline_data.loc[baseline_data.CHR.isin(odd_chrs)].to_numpy()
    data_even = baseline_data.loc[baseline_data.CHR.isin(even_chrs)].to_numpy()

    # save 
    X_even, Y_even = data_even[:, 3:-1], data_even[:,-1]
    X_odd, Y_odd = data_odd[:, 3:-1], data_odd[:,-1]
    print(f"X_even = {X_even.shape}, Y_even = {Y_even.shape}, X_odd = {X_odd.shape}, Y_odd = {Y_odd.shape}")

    if mode == "lowfreq":
        np.save(f"X_lowfreq_traits/{trait_name}/X_lowfreq_even.npy", X_even)
        np.save(f"Y_lowfreq_traits/{trait_name}/Y_lowfreq_even.npy", Y_even)
        np.save(f"X_lowfreq_traits/{trait_name}/X_lowfreq_odd.npy", X_odd)
        np.save(f"Y_lowfreq_traits/{trait_name}/Y_lowfreq_odd.npy", Y_odd)
    else:
        np.save(f"X_common_traits/{trait_name}/X_common_even.npy", X_even)
        np.save(f"Y_common_traits/{trait_name}/Y_common_even.npy", Y_even)
        np.save(f"X_common_traits/{trait_name}/X_common_odd.npy", X_odd)
        np.save(f"Y_common_traits/{trait_name}/Y_common_odd.npy", Y_odd)
