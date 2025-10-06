
"""
plot_tree.py
Re-plot a saved DecisionTreeRegressor from v2d.py as a PDF/PNG.
"""

import argparse
import pandas as pd
from joblib import load
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(
        description="Render a saved DecisionTreeRegressor (.joblib) from v2d.py"
    )
    ap.add_argument("--model_path", required=True,
                    help="Path to a saved DecisionTreeRegressor (e.g., runs/tree_md4_leaf10k.even.joblib)")
    ap.add_argument("--annot_csv", required=True,
                    help="Annotation TSV/CSV for feature names; first 4 cols must be CHR,BP,SNP,ID")
    ap.add_argument("--delimiter", default="\t",
                    help="Delimiter for the annotation file (default: tab)")
    ap.add_argument("--max_depth", type=int, default=None,
                    help="Optional display depth (visualization only; does not modify the model)")
    ap.add_argument("--out", required=True, help="Output figure path (e.g., .pdf or .png)")
    ap.add_argument("--dpi", type=int, default=300, help="DPI for raster outputs like PNG")
    args = ap.parse_args()

    # Load model
    model = load(args.model_path)
    if not isinstance(model, DecisionTreeRegressor):
        raise TypeError("Loaded artifact is not a sklearn.tree.DecisionTreeRegressor. "
                        "If you trained RF/XGB, extract a single tree or train with --model tree.")

    # Read a small sample to infer feature names (after 4 key columns)
    df = pd.read_csv(args.annot_csv, delimiter=args.delimiter, nrows=100)
    if df.shape[1] < 5:
        raise ValueError("Expected first 4 columns to be CHR,BP,SNP,ID followed by feature columns.")
    feature_names = list(df.columns[4:])

    # Plot
    plt.figure(figsize=(12, 8))
    plot_tree(
        model,
        feature_names=feature_names,
        filled=True,
        impurity=False,
        proportion=True,
        max_depth=args.max_depth,
        fontsize=6,
    )
    plt.tight_layout()

    ext = args.out.lower().rsplit(".", 1)[-1]
    if ext in {"png", "jpg", "jpeg"}:
        plt.savefig(args.out, dpi=args.dpi)
    else:
        plt.savefig(args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
