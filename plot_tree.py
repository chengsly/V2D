
"""
plot_tree.py
Re-plot a saved DecisionTreeRegressor from v2d.py as a PDF/PNG.
"""

import numpy as np
import pandas as pd
import os, sys
import shlex
from sklearn.tree import plot_tree
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
import argparse
from scipy import stats
import scipy.stats as st
from sklearn import preprocessing
import joblib
from joblib import dump, load
import time
from datetime import datetime
from sklearn.tree import export_graphviz
import graphviz



parser = argparse.ArgumentParser()

#paths
parser.add_argument("--annot", type=str, required=True)
parser.add_argument("--beta2", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--delimiter", type=str, default="\t")

#tree arguments
parser.add_argument("--min_samples_leaf", type=int, dest="min_samples_leaf", required=True)
parser.add_argument("--max_depth", type=int, dest="max_depth", required=True)

#exclude
parser.add_argument("--exclude", nargs="*", default=[])

parser.add_argument("--seed", type=int, default=0)


args = parser.parse_args()

version_string = """*********************************************************************
* Plot tree
* Version 1.0.0
* (C) 2025 Siliangyu Cheng, Steven Gazal
* University of Southern California
* GNU General Public License v3
*********************************************************************
"""

all_CHRs = list(range(1,23))
#all_CHRs = [21, 22]


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
    
#1) print out all the options in the log
print(args)

# file save names
filename_prefix = f"{args.out}"
log_filename = filename_prefix + ".log"
        
if os.path.exists(log_filename):
    os.remove(log_filename)

myprint(version_string)
cmd = " ".join(shlex.quote(arg) for arg in sys.argv)
myprint("Call:")
myprint(cmd)

start_time = time.time()
start_str = datetime.now().strftime("%a %b %d %H:%M:%S %Y")
myprint(f"\nBeginning analysis at {start_str}\n")

myprint(f"Selected model: tree.\nOptions: max_depth={args.max_depth}, "
            f"min_samples_leaf={args.min_samples_leaf}\n")
        
#2) read the data
myprint(f"Reading beta2 file from {args.beta2} ...")
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
first_four_cols = X.columns[:4].tolist()
#remove column if --remove
if len(args.exclude) != 0:
    myprint(f"  Removing annotation {args.exclude}.")
    X = X.drop(args.exclude, axis=1, errors="ignore")
myprint(f"  Annotation file for training has {X.shape[0]} SNPs and {X.shape[1] - 3} annotations.")
#update Y so that they have all the SNPs in X (match on CHR POS RS)
X_and_Y = Y.merge(X, on=first_four_cols, how="inner")
myprint(f"After merging beta2 and annotation files, {X_and_Y.shape[0]} SNPs remain.")
#
X_all = X_and_Y.drop(columns=["CHR", "BP", "SNP", "ID", "beta2"])
Y_all = X_and_Y["beta2"]
Y_all = Y_all / Y_all.mean()

myprint(f"\nRunning regression tree\n")
model = DecisionTreeRegressor(random_state=args.seed, min_samples_leaf=args.min_samples_leaf, max_depth=args.max_depth)
model.fit(X_all, Y_all)

feature_names = list(X.columns[4:])

# Plot
dot = export_graphviz(
    model,
    out_file=None,
    feature_names=feature_names,
    filled=True,
    rounded=True,
    impurity=False,
    proportion=True,
    max_depth=args.max_depth,
    precision=3,
)

# (optional) slightly larger node font
dot = dot.replace('node [shape=box]', 'node [shape=box, fontsize=10]')

# honor args.out extension if it's one of: pdf/png/jpg/jpeg
ext = args.out.lower().rsplit(".", 1)[-1]
fmt = ext if ext in {"pdf", "png", "jpg", "jpeg"} else "pdf"
base = args.out.rsplit(".", 1)[0] if fmt != "pdf" or "." in args.out else args.out

graphviz.Source(dot).render(base, format=fmt, cleanup=True)
myprint(f"Plot saved in {base}.{fmt}")

end_time = time.time()
end_str = datetime.now().strftime("%a %b %d %H:%M:%S %Y")
elapsed = end_time - start_time
mins, secs = divmod(elapsed, 60)

myprint(f"\nAnalysis finished at {end_str}")
myprint(f"Total time elapsed: {mins:.0f}m:{secs:.6f}s")
