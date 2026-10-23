"""Cross-validated comparison: dropping incomplete rows vs median imputation.

About half of the records in heart.csv have a missing value. train_model.py
drops them (leaving 298 rows). This script measures what is gained by keeping
all 597 rows and filling gaps with the training-fold median instead.

For both strategies it runs repeated stratified 5-fold cross-validation for
Logistic Regression and Random Forest (and the ANN with --ann). Imputation and
scaling live inside each fold's pipeline, so no information leaks from the
validation fold.

Run:
    python cross_validate.py          # Logistic Regression + Random Forest
    python cross_validate.py --ann    # also the ANN (slower, needs TensorFlow)

Writes results/cv_results.md.
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}
METRICS = list(SCORING)


def load_data():
    df = pd.read_csv("heart.csv")
    df = df.replace("?", np.nan).apply(pd.to_numeric)
    y = (df["target"] > 0).astype(int)
    X = df.drop(columns="target")
    return X, y


def preprocessing():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])


def sklearn_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=SEED),
    }


def cv_sklearn(model, X, y):
    pipe = Pipeline(preprocessing().steps + [("model", model)])
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    scores = cross_validate(pipe, X, y, cv=cv, scoring=SCORING)
    return {m: scores[f"test_{m}"] for m in METRICS}


def build_ann(n_features):
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.models import Sequential

    ann = Sequential([
        Dense(16, activation="relu", input_dim=n_features),
        Dropout(0.3),
        Dense(8, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    ann.compile(optimizer="adam", loss="binary_crossentropy",
                metrics=["accuracy"])
    return ann


def cv_ann(X, y):
    import tensorflow as tf

    tf.random.set_seed(SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    out = {m: [] for m in METRICS}
    for train_idx, test_idx in cv.split(X, y):
        pre = preprocessing()
        X_train = pre.fit_transform(X.iloc[train_idx])
        X_test = pre.transform(X.iloc[test_idx])
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        ann = build_ann(X_train.shape[1])
        ann.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
        prob = ann.predict(X_test, verbose=0).ravel()
        pred = (prob >= 0.5).astype(int)

        out["accuracy"].append(accuracy_score(y_test, pred))
        out["precision"].append(precision_score(y_test, pred, zero_division=0))
        out["recall"].append(recall_score(y_test, pred))
        out["f1"].append(f1_score(y_test, pred))
        out["roc_auc"].append(roc_auc_score(y_test, prob))
    return {m: np.array(v) for m, v in out.items()}


def fmt(scores):
    return " | ".join(f"{scores[m].mean():.3f} ± {scores[m].std():.3f}"
                      for m in METRICS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", action="store_true",
                        help="also cross-validate the neural network")
    args = parser.parse_args()

    X, y = load_data()
    complete = X.notna().all(axis=1)
    strategies = {
        f"Drop incomplete rows (n={int(complete.sum())})": (X[complete], y[complete]),
        f"Median imputation (n={len(X)})": (X, y),
    }

    lines = [
        "Mean ± standard deviation across folds. Random Forest and Logistic "
        "Regression: 5-fold stratified CV repeated 3 times (15 folds). "
        "ANN: 5-fold stratified CV (5 folds). Seed 42.",
        "",
        "| Strategy | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for strategy, (Xs, ys) in strategies.items():
        for name, model in sklearn_models().items():
            lines.append(f"| {strategy} | {name} | {fmt(cv_sklearn(model, Xs, ys))} |")
        if args.ann:
            lines.append(f"| {strategy} | ANN (dropout) | {fmt(cv_ann(Xs, ys))} |")

    report = "\n".join(lines) + "\n"
    os.makedirs("results", exist_ok=True)
    with open("results/cv_results.md", "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
