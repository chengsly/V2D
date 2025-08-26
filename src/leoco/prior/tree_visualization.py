import numpy as np
import os
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from six import StringIO
import pydot
import sys

max_leaf_nodes = 3 
trait_name = "mean"
feature_lst = []
mode = sys.argv[-1]

if mode == "common":
    feature_file = "../columns_common.txt"
else:
    feature_file = "../lowfreq_cols.txt"

with open(feature_file, "r") as f:
    feature_lst = f.readlines()
    feature_lst = [item.rstrip("\n") for item in feature_lst]

def visualize_tree(clf, max_depths):
    dot_data = StringIO()
    print(f"{trait_name}, {len(feature_lst)} features")
    tree.export_graphviz(clf, out_file=dot_data, feature_names=feature_lst)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    filename = f"tree_mean_max_depths={max_depths}_mean_sample_leaf=25000_{mode}_prior"
    graph[0].write_pdf(f"{filename}.pdf")


if __name__ == "__main__":
    folderX = "X_common_mean15" if mode == "common" else "X_lowfreq_mean15"
    folderY = "Y_common_mean15" if mode == "common" else "Y_lowfreq_mean15"
    x_prefix = mode
    Y_tr_name = os.path.join(folderY, f"Y_{x_prefix}.npy")
    X_tr_name = os.path.join(folderX, f"X_{x_prefix}.npy")
    X_tr = np.load(X_tr_name, allow_pickle=True)[:,1:]
    Y_tr = np.load(Y_tr_name, allow_pickle=True)
    Y_tr *= len(Y_tr)
    model = DecisionTreeRegressor(random_state=0, max_depth=20, min_samples_leaf=25000)
    model.fit(X_tr, Y_tr)
    visualize_tree(model, 20)
    # model = DecisionTreeRegressor(random_state=0, max_depth=20, min_samples_leaf=25000)
    # model.fit(X_tr, Y_tr)
    # visualize_tree(model, 5)