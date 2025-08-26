import numpy as np
import os, sys
import xgboost as xgb
import scipy.stats as st


learning_rate=0.05
gamma=10
scale_pos_weight=1


n_estimators, max_depth, min_child_weight, subsample = sys.argv[-4],sys.argv[-3],sys.argv[-2],sys.argv[-1]

n_estimators = int(n_estimators)
max_depth = int(max_depth)
min_child_weight = float(min_child_weight)
subsample = float(subsample)

mode = "lowfreq"

def SSE(Y_true, Y_pred):
    return np.sum(np.square(Y_true-Y_pred))

def print_params():
        print(f"n_estimators = {n_estimators}")
        print(f"max_depth = {max_depth}")
        print(f"min_child_weight = {min_child_weight}")
        print(f"subsample = {subsample}")

def train_leoco():
    X_even = np.load(f"X_{mode}_mean15/X_{mode}_even.npy",allow_pickle=True).astype(float)
    X_odd = np.load(f"X_{mode}_mean15/X_{mode}_odd.npy", allow_pickle=True).astype(float)
    Y_even = np.load(f"Y_{mode}_mean15/Y_{mode}_even.npy", allow_pickle=True).astype(float) * 10e7
    Y_odd = np.load(f"Y_{mode}_mean15/Y_{mode}_odd.npy", allow_pickle=True).astype(float) * 10e7
    
    one_to_left = st.beta.rvs(10, 1, size=1)[0]
    from_zero_positive = st.expon.rvs(0, 50,size=1)[0]
    nthread = 2
    seed=42
    model_even = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                            gamma=gamma, min_child_weight=min_child_weight, subsample=subsample,
                            scale_pos_weight=scale_pos_weight, reg_alpha=from_zero_positive, 
                            nthread=nthread, random_state=seed,
                            colsample_bytree=one_to_left)

    model_odd = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                            gamma=gamma, min_child_weight=min_child_weight, subsample=subsample,
                            scale_pos_weight=scale_pos_weight, reg_alpha=from_zero_positive, 
                            nthread=nthread, random_state=seed,
                            colsample_bytree=one_to_left)

    model_even.fit(X_even, Y_even)
    model_odd.fit(X_odd,Y_odd)
    model_even_pred = model_even.predict(X_odd)
    model_odd_pred = model_odd.predict(X_even)
    sse_even = SSE(model_odd_pred, Y_even)
    sse_odd = SSE(model_even_pred, Y_odd)

    mse = (sse_even + sse_odd) / (len(Y_even) + len(Y_odd))
    print_params()    
    print(f"test mse = {mse}")



if __name__ == "__main__":
    train_leoco()
