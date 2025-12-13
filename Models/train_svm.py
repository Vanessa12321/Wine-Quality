"""
Train SVM models (3-class wine quality) and print metrics.

- Uses only sklearn for ML (plus numpy/pandas/matplotlib as helpers).
- Expects the dataset at: ../Data/winequality-red.csv (when run from Models/).
"""

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt


RANDOM_STATE = 42
TEST_SIZE = 0.2


def make_labels(y_quality: np.ndarray, mode: str = "3class"):
    """Map original quality scores to binary or 3-class labels."""
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


def evaluate_model(name, model, X_test, y_test, class_names=None):
    """Evaluate a fitted model and return metrics dict."""
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n===== {name} =====")
    print("Accuracy        :", acc)
    print("Macro precision :", precision)
    print("Macro recall    :", recall)
    print("Macro F1-score  :", f1)
    print("Confusion matrix:\n", cm)
    print("\nClassification report:\n")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=class_names if class_names is not None else None,
            zero_division=0,
        )
    )

    return {
        "model": name,
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }, cm


def main(args):
    # --------- 1. Load data ---------
    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    cols = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
        "quality",
    ]

    # UCI wine-quality data is semicolon-separated; first line is the header.
    df = pd.read_csv(
        data_path, sep=";", skiprows=1, header=None, names=cols
    )

    print("First 5 rows:")
    print(df.head())
    print("\nQuality value counts:")
    print(df["quality"].value_counts().sort_index())

    # --------- 2. Labels (3-class) ---------
    y, class_names = make_labels(df["quality"].to_numpy(), mode="3class")
    X = df.drop(columns=["quality"])

    print("\nClass distribution (0=low, 1=mid, 2=high):")
    print(pd.Series(y).value_counts().sort_index())
    print("Class names:", class_names)

    # --------- 3. Train-test split ---------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTrain shape:", X_train.shape, " Test shape:", X_test.shape)

    # --------- 4. Baseline SVM (RBF) ---------
    svm_basic = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", random_state=RANDOM_STATE)),
        ]
    )
    svm_basic.fit(X_train, y_train)
    scores_basic, cm_basic = evaluate_model(
        "SVM_RBF_basic", svm_basic, X_test, y_test, class_names
    )

    # --------- 5. Tuned SVM (RBF + GridSearchCV) ---------
    svm_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", random_state=RANDOM_STATE)),
        ]
    )

    param_grid = {
        "svc__C": [0.1, 1, 10, 100],
        "svc__gamma": ["scale", "auto", 0.01, 0.1, 1.0],
    }

    grid_search = GridSearchCV(
        estimator=svm_pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    print("\nBest params:", grid_search.best_params_)
    print("Best CV macro F1:", grid_search.best_score_)

    best_svm = grid_search.best_estimator_
    scores_tuned, cm_tuned = evaluate_model(
        "SVM_RBF_tuned", best_svm, X_test, y_test, class_names
    )

    # --------- 6. Linear SVM (extra comparison) ---------
    svm_linear = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="linear", random_state=RANDOM_STATE)),
        ]
    )
    svm_linear.fit(X_train, y_train)
    scores_linear, cm_linear = evaluate_model(
        "SVM_linear_basic", svm_linear, X_test, y_test, class_names
    )

    # --------- 7. RBF with class_weight="balanced" (extra) ---------
    svm_balanced = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    svm_balanced.fit(X_train, y_train)
    scores_balanced, cm_balanced = evaluate_model(
        "SVM_RBF_balanced", svm_balanced, X_test, y_test, class_names
    )

    # --------- 8. Save metrics + confusion matrix for tuned model ---------
    results_dir = Path(args.outdir)
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "basic": scores_basic,
        "tuned": scores_tuned,
        "linear": scores_linear,
        "balanced": scores_balanced,
    }

    with open(results_dir / "metrics_svm.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(results_dir / "classification_report_svm_tuned.txt", "w") as f:
        y_pred_tuned = best_svm.predict(X_test)
        report_text = classification_report(
            y_test, y_pred_tuned, target_names=class_names, zero_division=0
        )
        f.write(report_text)

    # tuned confusion matrix plot
    fig, ax = plt.subplots()
    im = ax.imshow(cm_tuned, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix – Tuned SVM (3-class)")
    for i in range(cm_tuned.shape[0]):
        for j in range(cm_tuned.shape[1]):
            ax.text(
                j,
                i,
                cm_tuned[i, j],
                ha="center",
                va="center",
                color="black",
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.savefig(results_dir / "confusion_matrix_svm_tuned.png")
    plt.close(fig)

    print(f"\nSaved metrics and confusion matrix in: {results_dir.resolve()}")


if __name__ == "__main__":
    # Default paths assuming this file is in Models/
    this_dir = Path(__file__).resolve().parent
    default_data = this_dir.parent / "Data" / "winequality-red.csv"
    default_out = this_dir / "results_svm_tuned"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default=str(default_data),
        help="Path to winequality-red.csv",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(default_out),
        help="Directory to save SVM results",
    )
    args = parser.parse_args()

    main(args)