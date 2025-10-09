import numpy as np
import pandas as pd
import os, sys
import shlex
from sklearn.linear_model import LinearRegression
from sklearn.tree import plot_tree
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import argparse
from scipy import stats
import scipy.stats as st
from sklearn import preprocessing
import joblib
from joblib import dump, load
import time
from datetime import datetime

parser = argparse.ArgumentParser()

#model selected
parser.add_argument("--model", type=str, choices=['linear', 'rf', 'mlp', 'xgb', 'tree'], required=True)

#model arguments
#MLP arguments
parser.add_argument("--n_neurons", type=int, dest="n_neurons")
parser.add_argument("--n_layers", type=int, dest="n_layers")
#tree and/or rf arguments
parser.add_argument("--min_samples_leaf", type=int, dest="min_samples_leaf")
parser.add_argument("--max_depth", type=int, dest="max_depth") ## also used by xgboost
parser.add_argument("--n_estimators", type=int, dest="n_estimators") ## also used by xgboost
# xgboost params
parser.add_argument("--min_child_weight", type=int, dest="min_child_weight")
parser.add_argument("--gamma", type=float, dest="gamma")
parser.add_argument("--subsample", type=float, dest="subsample")
parser.add_argument("--scale_pos_weight", type=int, dest="scale_pos_weight")
parser.add_argument("--learning_rate", type=float, dest="learning_rate")


#paths
parser.add_argument("--annot", type=str, required=True)
parser.add_argument("--beta2", type=str, required=True)
parser.add_argument("--pred", type=str, default=None)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--delimiter", type=str, default="\t")

#seeds
parser.add_argument("--init_seed", type=int, default=0)
parser.add_argument("--nbseed", type=int, default=1)

#exclude
parser.add_argument("--exclude", nargs="*", default=[])

#outputs
parser.add_argument("--print_mse", action='store_true')
parser.add_argument("--print_model", action='store_true')

args = parser.parse_args()

version_string = """*********************************************************************
* Variant-to-disease (V2D) framework
* Version 1.0.0
* (C) 2025 Siliangyu Cheng, Steven Gazal
* University of Southern California
* GNU General Public License v3
*********************************************************************
"""

even_CHRs = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
odd_CHRs = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
all_CHRs = list(range(1,23))
even_CHRs = [22]
odd_CHRs = [21]
all_CHRs = [21, 22]


def sse(Y_true, Y_pred):
    return np.sum(np.square(Y_true-Y_pred))

def myprint(mytext):
    print(f"{mytext}")
    with open(log_filename, "a") as f:
        f.writelines(f"{mytext}\n")

def find_annot_for_chr(chr, prefix=args.annot):
    filename = f"{prefix}{chr}.annot.gz"   # e.g. annotations/baselineLF.common.21.annot.gz
    path = Path(filename)
    if not path.exists():
        myprint(f"WARNING: annotation for chromosome {chr} not found at {path}")
        df = pd.DataFrame()
        return df
    df = pd.read_csv(path, delimiter=args.delimiter)
    return df

# train_leoco function
def train_leoco(X_tr, X_te, Y_tr, Y_te, Xpred, even_odd, CHRs):
    seeds = args.init_seed + np.arange(args.nbseed)
    minMSE = np.inf
    bestmodel = None
    tr_count = X_tr.shape[0]
    te_count = X_te.shape[0]
    mse_test = []
    normalizer = preprocessing.Normalizer()
    X_tr_np = normalizer.fit_transform(X_tr.iloc[:,4:].to_numpy()) #skip the first 4 columns
    X_te_np = normalizer.transform(X_te.iloc[:,4:].to_numpy())
    if args.pred:
        Xpred_np = normalizer.transform(Xpred.iloc[:,4:].to_numpy())
    
    for seed in seeds:
        #define the model
        if (args.model == "mlp"):
            model = MLPRegressor(random_state=seed, 
                                    max_iter=500, 
                                    hidden_layer_sizes=[args.n_neurons] * args.n_layers,
                                    shuffle=False,
                                    verbose=True, 
                                    n_iter_no_change=100,
                                    early_stopping=True)
        #other models here
        elif args.model == "linear":
            model = LinearRegression()

        elif args.model == "tree":
            model = DecisionTreeRegressor(random_state=seed, min_samples_leaf=args.min_samples_leaf, max_depth=args.max_depth)
        
        elif args.model == "rf":
            model = RandomForestRegressor(n_estimators=args.n_estimators,
                                          max_depth=args.max_depth, 
                                          min_samples_leaf=args.min_samples_leaf,
                                          random_state=seed)
        else:
            n_estimators = args.n_estimators
            max_depth = args.max_depth
            learning_rate = args.learning_rate
            gamma = args.gamma
            one_to_left = st.beta.rvs(10, 1, size=1)[0]
            min_child_weight = args.min_child_weight
            from_zero_positive = st.expon.rvs(0, 50,size=1)[0]
            nthread = 1
            scale_pos_weight = args.scale_pos_weight
            subsample = args.subsample
            learning_rate = args.learning_rate 
            model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                                    gamma=gamma, min_child_weight=min_child_weight, subsample=subsample,
                                    scale_pos_weight=scale_pos_weight, reg_alpha=from_zero_positive, 
                                    nthread=nthread, random_state=seed,
                                    colsample_bytree=one_to_left)
        model.fit(X_tr_np, Y_tr)
        Y_pred_tr = model.predict(X_tr_np)
        CHR_sse_tr = sse(Y_tr, Y_pred_tr)
        CHR_mse_tr = CHR_sse_tr / tr_count

        Y_pred = model.predict(X_te_np)
        CHR_sse_te = sse(Y_te, Y_pred)
        CHR_mse_te = CHR_sse_te / te_count
        mse_test.append(CHR_mse_te)
        
        if CHR_mse_tr < minMSE:
            minMSE = CHR_mse_tr
            bestmodel = model

        if args.print_mse:
            with open(mse_filename, "a") as f:
                f.writelines(f"{seed}\t{even_odd}\t{len(Y_te)+1}\t{CHR_mse_tr}\t{CHR_mse_te}\n")

    if args.print_mse:
            myprint(f"  Print mse to {mse_filename}.")
    
    # save prediction file
    if args.pred:
        myprint(f"  Print prediction for {even_odd} chromosomes to {filename_prefix}.*.csv.")
        Y_pred = bestmodel.predict(Xpred_np)
        info_pred = Xpred.iloc[:,:4]
        Y_pred_df = info_pred.copy()
        Y_pred_df["V2D"] = Y_pred / Y_pred.mean()
        # loop over chromosomes and output
        for chr in CHRs:
            pred_filename = f"{filename_prefix}.{chr}.csv"
            Y_pred_df_chr = Y_pred_df.loc[Y_pred_df.CHR == chr]
            Y_pred_df_chr.to_csv(pred_filename, sep='\t', index=False)

    # save model 
    if args.print_model:
        model_filename = f"{filename_prefix}.{even_odd}.joblib"
        myprint(f"  Print model for {even_odd} chromosomes to {filename_prefix}.{even_odd}.joblib")
        dump(bestmodel, model_filename)


#1) print out all the options in the log
print(args)

# file save names
filename_prefix = f"{args.out}"
log_filename = filename_prefix + ".log"
        
if os.path.exists(log_filename):
    os.remove(log_filename)

if args.print_mse:
    mse_filename = f"{filename_prefix}.mse.txt"
    if os.path.exists(mse_filename):
        os.remove(mse_filename)
    with open(mse_filename, "w") as f:
        f.write("seed\tchr_test\tnb_snps\tmse_train\tmse_test\n")

myprint(version_string)
cmd = " ".join(shlex.quote(arg) for arg in sys.argv)
myprint("Call:")
myprint(cmd)

start_time = time.time()
start_str = datetime.now().strftime("%a %b %d %H:%M:%S %Y")
myprint(f"\nBeginning analysis at {start_str}\n")

#2) comprehensive check error in options
def _is_set(v):
    return v is not None

allowed_by_model = {
    "linear": {"init_seed", "nbseed"},
    "mlp":    {"n_neurons", "n_layers", "init_seed", "nbseed"},
    "tree":   {"min_samples_leaf", "max_depth", "init_seed", "nbseed"},
    "rf":     {"min_samples_leaf", "max_depth", "n_estimators", "init_seed", "nbseed"},
    "xgb":{"max_depth", "n_estimators", "min_child_weight", "gamma",
               "subsample", "scale_pos_weight", "learning_rate", "init_seed", "nbseed"},
}

all_params = {
    "n_neurons", "n_layers",
    "min_samples_leaf", "max_depth", "n_estimators",
    "min_child_weight", "gamma", "subsample",
    "scale_pos_weight", "learning_rate",
    "init_seed", "nbseed",
}

model = args.model
if model not in allowed_by_model:
    raise ValueError(f"Unknown --model {model}")

# --- Forbid params that were set but not allowed for this model ---
for p in all_params:
    if _is_set(getattr(args, p, None)) and p not in allowed_by_model[model]:
        flag = "--" + p.replace("_", "-")
        allowed_flags = ", ".join("--" + a.replace("_", "-") for a in sorted(allowed_by_model[model]))
        raise ValueError(
            f"{flag} cannot be used with --model {model}; "
            f"allowed options for {model} are: {allowed_flags or '(none)'}"
        )

# --- Required params per model ---
if model == "mlp":
    if not _is_set(args.n_neurons) or not _is_set(args.n_layers):
        raise ValueError("--model mlp requires --n_neurons and --n_layers")

# Random forest usually requires n_estimators (a single tree doesn’t)
if model == "rf":
    if not _is_set(args.n_estimators):
        raise ValueError("--model rf requires --n_estimators")

# Decision tree must NOT have n_estimators (that would imply an ensemble)
if model == "tree" and _is_set(getattr(args, "n_estimators", None)):
    raise ValueError("--n_estimators is not valid for --model tree; did you mean --model rf?")

# --- Value/range checks (when provided) ---
if _is_set(getattr(args, "n_neurons", None)) and args.n_neurons < 1:
    raise ValueError("--n_neurons must be >= 1")

if _is_set(getattr(args, "n_layers", None)) and args.n_layers < 1:
    raise ValueError("--n_layers must be >= 1")

if _is_set(getattr(args, "min_samples_leaf", None)) and args.min_samples_leaf < 1:
    raise ValueError("--min_samples_leaf must be >= 1")

if _is_set(getattr(args, "max_depth", None)) and args.max_depth < 1:
    raise ValueError("--max_depth must be >= 1 (or omit to use None)")

if _is_set(getattr(args, "n_estimators", None)) and args.n_estimators < 1:
    raise ValueError("--n_estimators must be >= 1")

if _is_set(getattr(args, "min_child_weight", None)) and args.min_child_weight < 0:
    raise ValueError("--min_child_weight must be >= 0")

if _is_set(getattr(args, "gamma", None)) and args.gamma < 0.0:
    raise ValueError("--gamma must be >= 0.0")

if _is_set(getattr(args, "subsample", None)) and not (0.0 < args.subsample <= 1.0):
    raise ValueError("--subsample must be in (0, 1]")

if _is_set(getattr(args, "scale_pos_weight", None)) and args.scale_pos_weight <= 0:
    raise ValueError("--scale_pos_weight must be > 0")

if _is_set(getattr(args, "learning_rate", None)) and args.learning_rate <= 0.0:
    raise ValueError("--learning_rate must be > 0.0")

if model == "linear":
    myprint("Selected model: linear.\n")
elif model == "mlp":
    myprint(f"Selected model: mlp.\nOptions: n_neurons={args.n_neurons}, n_layers={args.n_layers}\n")
elif model == "tree":
    myprint(f"Selected model: tree.\nOptions: max_depth={args.max_depth}, "
            f"min_samples_leaf={args.min_samples_leaf}\n")
elif model == "rf":
    myprint(f"Selected model: rf.\nOptions: n_estimators={args.n_estimators}, "
            f"max_depth={args.max_depth}, min_samples_leaf={args.min_samples_leaf}\n")
elif model == "xgb":
    myprint(
        "Selected model: xgboost.\n"
        f"Options: n_estimators={args.n_estimators}, max_depth={args.max_depth}, "
        f"min_child_weight={args.min_child_weight}, gamma={args.gamma}, "
        f"subsample={args.subsample}, scale_pos_weight={args.scale_pos_weight}, "
        f"learning_rate={args.learning_rate}\n"
    )
    
    
#3) read the data
myprint(f"Reading beta2 files from {args.beta2} ...")
Y = pd.read_csv(args.beta2, delimiter=args.delimiter)
myprint(f"  Read beta2 values for {len(Y)} SNPs.")

# auto-rename last column to beta2 if needed
if Y.columns[-1] != "beta2":
    Y = Y.rename(columns={Y.columns[-1]: "beta2"})

# read all annots
from pathlib import Path


myprint(f"Reading annotation files for training from {args.annot}[1..22] ...")
X = pd.concat([find_annot_for_chr(chr) for chr in all_CHRs])

#check colnames for the files (4 first should CHR POS RS ID)
assert X.columns[:4].tolist() == ["CHR", "POS", "RS", "ID"] or X.columns[:4].tolist() == ["CHR", "BP", "SNP", "ID"] or X.columns[:4].tolist() == ["CHR", "BP", "rsid", "ID"]
assert Y.columns[:4].tolist() == ["CHR", "POS", "RS", "ID"] or Y.columns[:4].tolist() == ["CHR", "BP", "SNP", "ID"] or Y.columns[:4].tolist() == ["CHR", "BP", "rsid", "ID"]
first_three_cols = X.columns[:4].tolist()
#remove column if --remove
if len(args.exclude) != 0:
    myprint(f"  Removing annotation {args.exclude}.")
    X = X.drop(args.exclude, axis=1, errors="ignore")
myprint(f"  Annotation file for training has {X.shape[0]} SNPs and {X.shape[1] - 3} annotations.")
#update Y so that they have all the SNPs in X (match on CHR POS RS)
X_and_Y = Y.merge(X, on=first_three_cols, how="inner")
myprint(f"After merging beta2 and annotation files, {X_and_Y.shape[0]} SNPs remain.")
#
Xpred_odd, Xpred_even = None, None

if args.pred:
    myprint(f"Reading annotation files for prediction from {args.pred}[1..22] ...")
    assert args.pred is not None, "--pred requires a valid path"

    if args.pred == args.annot:
        Xpred = X
    else:
        Xpred = pd.concat([find_annot_for_chr(chr, args.pred) for chr in all_CHRs])
        # check colnames for the files (4 first should be CHR POS RS ID)
        assert Xpred.columns[:4].tolist() == ["CHR", "POS", "RS", "ID"] or Xpred.columns[:4].tolist() == ["CHR", "BP", "SNP", "ID"] or Xpred.columns[:4].tolist() ==["CHR", "BP", "rsid", "ID"]

        # remove excluded columns if any
        if len(args.exclude) != 0:
            Xpred = Xpred.drop(args.exclude, axis=1, errors="ignore")
    
    myprint(f"  Annotation file for prediction has {X.shape[0]} SNPs and {X.shape[1] - 3} annotations.")
    
    # make sure CHR is int
    Xpred.CHR = Xpred.CHR.astype(int)

    # split into odd/even chromosome sets
    Xpred_odd = Xpred.loc[Xpred.CHR.isin(odd_CHRs)]
    Xpred_even = Xpred.loc[Xpred.CHR.isin(even_CHRs)]

    # optional cleanup
    del Xpred


X_and_Y.CHR = X_and_Y.CHR.astype(int)

myprint(f"\nSplitting files in even/odd chromosomes for training.")
# split into odd/even sets
X_odd = X_and_Y.loc[X_and_Y.CHR.isin(odd_CHRs)].drop(columns=["beta2"])
Y_odd = X_and_Y.loc[X_and_Y.CHR.isin(odd_CHRs), "beta2"]

X_even = X_and_Y.loc[X_and_Y.CHR.isin(even_CHRs)].drop(columns=["beta2"])
Y_even = X_and_Y.loc[X_and_Y.CHR.isin(even_CHRs), "beta2"]


#normalise Y_odd and Y_even so that sum = number of SNPs
Y_odd_np = Y_odd.to_numpy() * 10e7#/ Y_odd.mean()
Y_even_np = Y_even.to_numpy() * 10e7 #/ Y_even.mean()
del Y_odd, Y_even

myprint(f"  Even chromosomes have {X_even.shape[0]} SNPS, odd chromosomes have {X_odd.shape[0]} SNPs.")


#4) train leoco 

myprint(f"\nStart training on odd chromsomes.")
train_leoco(X_odd, X_even, Y_odd_np, Y_even_np, Xpred_even, "even", even_CHRs)

myprint(f"\nStart training on even chromsomes.")
train_leoco(X_even, X_odd, Y_even_np, Y_odd_np, Xpred_odd, "odd", odd_CHRs)

end_time = time.time()
end_str = datetime.now().strftime("%a %b %d %H:%M:%S %Y")
elapsed = end_time - start_time
mins, secs = divmod(elapsed, 60)

myprint(f"\nAnalysis finished at {end_str}")
myprint(f"Total time elapsed: {mins:.0f}m:{secs:.6f}s")
####END              
