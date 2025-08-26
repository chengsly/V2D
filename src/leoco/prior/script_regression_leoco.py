import numpy as np
import pandas as pd
import os, sys
from sklearn.linear_model import LinearRegression
from sklearn.tree import plot_tree
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import argparse
from six import StringIO
import pydot
from scipy import stats
import scipy.stats as st
import xgboost as xgb
import time

parser = argparse.ArgumentParser(description='Regression script arguments')
parser.add_argument('--model', type=str, dest='model', default="tree")
parser.add_argument("--min_sample", type=int, dest="min_sample", default=1000)
parser.add_argument("--max_depths", type=int, dest="max_depths", default=35)
parser.add_argument("--n_estimators", type=int, dest="n_estimators", default=100)
parser.add_argument("--trait_name", type=str, dest="trait_name", default="mean")
parser.add_argument("--n_neurons", type=int, dest="n_neurons", default=10)
parser.add_argument("--n_layers", type=int, dest="n_layers", default=4)
parser.add_argument("--lr", type=float, dest="lr", default=1e-6)
parser.add_argument("--seed", type=int, dest="seed", default=0)
# boolean flags
parser.add_argument("--lowfreq", action="store_true")
parser.add_argument("--try_seeds", action="store_true")
# xgboost params
parser.add_argument("--max_depth_x", type=int, dest="max_depth_x", default=3)
parser.add_argument("--min_child_weight", type=int, dest="min_child_weight", default=1)
parser.add_argument("--gamma", type=float, dest="gamma", default=0)
parser.add_argument("--subsample", type=float, dest="subsample", default=0.5)
parser.add_argument("--scale_pos_weight", type=int, dest="scale_pos_weight", default=1)
parser.add_argument("--n_estimators_x", type=int, dest="n_estimators_x",default=20)
parser.add_argument("--learning_rate", type=float, dest="learning_rate", default=0.01)
args = parser.parse_args()


if args.lowfreq:
    print("lowfreq")
else:
    print("common")


feature_lst = []
if not args.lowfreq:
    feature_file = "../columns_common.txt"
else:
    feature_file = "../lowfreq_cols.txt"

with open(feature_file, "r") as f:
    feature_lst = f.readlines()
    feature_lst = [item.rstrip("\n") for item in feature_lst]
trait_zhang_allergy = ["Zhang_Fetal_T_Lymphocyte_2_Cytotoxic", "Zhang_Natural_Killer_T", "Zhang_Fetal_T_Lymphocyte_1_CD4", "Zhang_T_Lymphocyte_1_CD8", "Zhang_T_lymphocyte_2_CD4"]
trait_zhang_heightz = ["Zhang_Cardiac_Fibroblast", "Zhang_Pericyte_General_3", "Zhang_Vasc_Sm_Muscle_1", "Zhang_Fibro_General", "Zhang_Vasc_Sm_Muscle_2"]


def SSE(Y_true, Y_pred):
    return np.sum(np.square(Y_true-Y_pred))

def print_params(model_name, pred=[], Y_te=[]):
    if model_name == "linear":
        return 
    if model_name == "rf":
        print("random forest")
        print(f"max_depth = {args.max_depths}")
        print(f"min_samples_leaf = {args.min_sample}")
        print(f"n_estimators = {args.n_estimators}")

    if model_name == "tree":
        print("tree")
        print(f"max_depth = {args.max_depths}")
        print(f"min_samples_leaf = {args.min_sample}")
    if model_name == "mlp":
        print("mlp")
        print(f"n_neurons = {args.n_neurons}")
        print(f"n_layers = {args.n_layers}")
    if model_name == "xgboost":
        n_estimators = args.n_estimators_x
        max_depth = args.max_depth_x
        learning_rate = args.learning_rate
        gamma = args.gamma
        one_to_left = st.beta.rvs(10, 1, size=1)[0]
        min_child_weight = args.min_child_weight
        from_zero_positive = st.expon.rvs(0, 50,size=1)[0]
        nthread = 1
        scale_pos_weight = args.scale_pos_weight
        subsample = args.subsample
        learning_rate = args.learning_rate
        print(f"n_estimators = {n_estimators}")
        print(f"max_depth = {max_depth}")
        print(f"learning_rate = {learning_rate}")
        print(f"gamma = {gamma}")
        print(f"min_child_weight = {min_child_weight}")
        print(f"subsample = {subsample}")
        print(f"scale_pos_weight = {scale_pos_weight}")
        print(f"reg_alpha = {from_zero_positive}")
        print(f"colsample_bytree = {one_to_left}")
        if len(Y_te) > 0 and len(pred) > 0:
            print(f"MSE on test split = {mean_squared_error(pred, Y_te)}")
        # print(f"seed = {args.seed}")


def train_leoco(model_name, args):
    mode = "common" if not args.lowfreq else "lowfreq"
    X_even = np.load(f"X_{mode}_mean15/X_{mode}_even.npy",allow_pickle=True).astype(float)
    X_odd = np.load(f"X_{mode}_mean15/X_{mode}_odd.npy", allow_pickle=True).astype(float)
    Y_even = np.load(f"Y_{mode}_mean15/Y_{mode}_even.npy", allow_pickle=True).astype(float) * 10e7
    Y_odd = np.load(f"Y_{mode}_mean15/Y_{mode}_odd.npy", allow_pickle=True).astype(float) * 10e7
    
    # remove 54
    # X_even = np.delete(X_even, 54, 1) 
    # X_odd = np.delete(X_odd, 54, 1)
    
    if model_name == "mlp" or (model_name == "xgboost" and args.try_seeds):
        seeds = [i for i in range(10)]
    else:
        seeds = [1]
    bestseed = seeds[0]
    best_mse_te = np.inf
    best_mse_tr = np.inf

    for seed in seeds:
        sse_even_tr, sse_even_te = train_leoco_model_selection_single_pass(model_name, args, X_train=X_even, Y_train=Y_even, X_test=X_odd, Y_test=Y_odd, seed=seed)
        sse_odd_tr, sse_odd_te = train_leoco_model_selection_single_pass(model_name, args, X_train=X_odd, Y_train=Y_odd, X_test=X_even, Y_test=Y_even, seed=seed)
        mse_tr = (sse_even_tr + sse_odd_tr) / (len(Y_even) + len(Y_odd))
        mse_te = (sse_even_te + sse_odd_te) / (len(Y_even) + len(Y_odd))
        print(f"seed = {seed}: leoco mse test = {mse_te}, leoco mse train = {mse_tr}")
        if mse_tr <= best_mse_tr:
            best_mse_te = mse_te
            best_mse_tr = mse_tr
            bestseed = seed

    print_params(model_name)
    print(f"best seed = {bestseed}, best_mse_tr = {best_mse_tr}")
    print(f"regression for {model_name}: leoco mse test = {best_mse_te}, leoco mse train = {best_mse_tr}")



def train_leoco_model_selection_single_pass(model_name, args, X_train, Y_train, X_test, Y_test, seed):    
    if model_name == "linear":
        model = LinearRegression()
    elif model_name == "tree":
        model = DecisionTreeRegressor(random_state=0, min_samples_leaf=args.min_sample, max_depth=args.max_depths)
    elif model_name == "mlp":
        model = MLPRegressor(random_state=seed, 
                            max_iter=100, 
                            hidden_layer_sizes=(args.n_neurons, args.n_layers),
                            alpha=args.lr,
                            shuffle=False,
                            verbose=True, 
                            n_iter_no_change=10,
                            early_stopping=True)
    elif model_name == "rf":
        model = RandomForestRegressor(max_depth=args.max_depths, 
                                    min_samples_leaf=args.min_sample)
    else:
        n_estimators = args.n_estimators_x
        max_depth = args.max_depth_x
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

    model.fit(X_train, Y_train)
    if model_name == "mlp":
        converged = model.n_iter_ < model.max_iter
        print(f"model convergence = {converged}: actual iters = {model.n_iter_}, max iters = {model.max_iter}")
    Y_pred = model.predict(X_test)
    Y_pred2 = model.predict(X_train)
    sse_te = SSE(Y_pred, Y_test)
    sse_tr = SSE(Y_pred2, Y_train)
    if model_name == "mlp":
        print(f"seed = {seed}, mlp result sse test = {SSE(Y_pred, Y_test)}, mlp result sse train = {SSE(Y_pred2, Y_train)}")
    return sse_tr, sse_te


if __name__ == "__main__":
    train_leoco(args.model, args)
