import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier


def load_csv(path: str, sep: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=sep)


def make_labels(y_quality: np.ndarray, mode: str):
    yq = pd.Series(y_quality).astype(int).to_numpy()

    if mode == "binary":
        y = (yq >= 7).astype(int)
        names = ["bad(<=6)", "good(>=7)"]
    elif mode == "3class":
        y = np.where(yq <= 5, 0, np.where(yq == 6, 1, 2))
        names = ["low(<=5)", "mid(=6)", "high(>=7)"]
    else:
        raise ValueError("mode must be 'binary' or '3class'")
    return y, names


def save_confusion(cm, label_names, out_png):
    fig = plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(label_names)), label_names, rotation=25, ha="right")
    plt.yticks(range(len(label_names)), label_names)
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def save_feature_importance(importances, feature_names, out_png):
    idx = np.argsort(importances)[::-1]
    fig = plt.figure(figsize=(8, 5))
    plt.bar(range(len(feature_names)), importances[idx])
    plt.xticks(range(len(feature_names)), np.array(feature_names)[idx], rotation=35, ha="right")
    plt.title("RF Feature Importance")
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="e.g., Data/processed/df.csv")
    ap.add_argument("--label_mode", default="3class", choices=["binary", "3class"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--n_iter", type=int, default=40)
    ap.add_argument("--outdir", default="Models/RF/results")
    ap.add_argument("--target_col", default="quality")
    ap.add_argument("--sep", default=",")
    ap.add_argument("--drop_cols", default="", help="comma-separated columns to drop (e.g., index,Id)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_csv(args.data, args.sep)

    if args.drop_cols.strip():
        drop_list = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
        exist = [c for c in drop_list if c in df.columns]
        if exist:
            df = df.drop(columns=exist)

    if args.target_col not in df.columns:
        raise ValueError(f"target_col '{args.target_col}' not found. columns={df.columns.tolist()}")

    missing = int(df.isna().sum().sum())
    dup = int(df.duplicated().sum())

    X = df.drop(columns=[args.target_col]).copy()
    y_raw = df[args.target_col].values
    y, label_names = make_labels(y_raw, args.label_mode)

    X = X.apply(pd.to_numeric, errors="coerce")
    if X.isna().sum().sum() > 0:
        X = X.fillna(X.median(numeric_only=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y
    )

    base = RandomForestClassifier(
        n_estimators=500,
        random_state=args.seed,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    model = base

    if args.tune:
        # tuning + regularization parameters
        param_dist = {
            "n_estimators": [300, 500, 800, 1200],
            "max_depth": [None, 6, 8, 10, 12, 16],
            "min_samples_split": [2, 4, 6, 10],
            "min_samples_leaf": [1, 2, 4, 6, 10],
            "max_features": ["sqrt", "log2", 0.5, 0.7],
            "bootstrap": [True, False],
            "class_weight": [None, "balanced", "balanced_subsample"],
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

        search = RandomizedSearchCV(
            estimator=base,
            param_distributions=param_dist,
            n_iter=args.n_iter,
            scoring="f1_macro",   
            cv=cv,
            random_state=args.seed,
            n_jobs=-1,
            verbose=1
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_

        with open(os.path.join(args.outdir, "best_params.json"), "w") as f:
            json.dump(search.best_params_, f, indent=2)

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, pred, average="weighted")),
        "label_mode": args.label_mode,
        "missing_values_total": missing,
        "duplicate_rows": dup,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "train_class_counts": {str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
        "test_class_counts": {str(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
    }

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    rep = classification_report(y_test, pred, target_names=label_names, digits=4)
    with open(os.path.join(args.outdir, "classification_report.txt"), "w") as f:
        f.write(rep)

    cm = confusion_matrix(y_test, pred)
    np.savetxt(os.path.join(args.outdir, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")
    save_confusion(cm, label_names, os.path.join(args.outdir, "confusion_matrix.png"))

    if hasattr(model, "feature_importances_"):
        save_feature_importance(
            model.feature_importances_,
            X.columns.tolist(),
            os.path.join(args.outdir, "feature_importance.png")
        )

    print("Saved results to:", args.outdir)
    print("Metrics:", metrics)
    print(rep)


if __name__ == "__main__":
    main()
