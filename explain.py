"""Explainability: which clinical features drive the predictions?

Two complementary views, both using the same cleaning as train_model.py
(incomplete rows dropped, binary target, StandardScaler):

1. Permutation importance (model-agnostic) for Logistic Regression and
   Random Forest. Inside each fold of a 5-fold stratified cross-validation,
   the model is fitted on the training part and each feature is shuffled in
   the held-out part. The drop in ROC-AUC shows how much the model relies on
   that feature. Measuring on held-out data avoids rewarding features the
   model merely memorised.
2. Logistic Regression coefficients on standardized features, which show the
   direction of each effect (raises or lowers the predicted risk).

Run:  python explain.py
Writes results/feature_importance.md and results/feature_importance.png.
"""
import os

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
N_REPEATS = 20

FEATURE_NAMES = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting blood sugar",
    "restecg": "Resting ECG",
    "thalach": "Max heart rate",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (oldpeak)",
    "slope": "ST slope",
    "ca": "Major vessels (ca)",
    "thal": "Thalassemia (thal)",
}


def load_data():
    df = pd.read_csv("heart.csv")
    df = df.replace("?", np.nan).dropna().apply(pd.to_numeric)
    y = (df["target"] > 0).astype(int)
    X = df.drop(columns="target")
    return X, y


def models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=SEED),
    }


def cv_permutation_importance(model, X, y):
    """Mean and std of the ROC-AUC drop per feature across held-out folds."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    per_fold = []
    for train_idx, test_idx in cv.split(X, y):
        pipe = Pipeline([("scale", StandardScaler()), ("model", model)])
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        result = permutation_importance(
            pipe, X.iloc[test_idx], y.iloc[test_idx], scoring="roc_auc",
            n_repeats=N_REPEATS, random_state=SEED)
        per_fold.append(result.importances_mean)
    per_fold = np.array(per_fold)
    return pd.DataFrame({"mean": per_fold.mean(axis=0),
                         "std": per_fold.std(axis=0)}, index=X.columns)


def main():
    X, y = load_data()

    importances = {name: cv_permutation_importance(m, X, y)
                   for name, m in models().items()}

    lr = Pipeline([("scale", StandardScaler()),
                   ("model", LogisticRegression(max_iter=1000))]).fit(X, y)
    coefs = pd.Series(lr.named_steps["model"].coef_[0], index=X.columns)

    order = importances["Random Forest"]["mean"].sort_values(ascending=False).index

    # ---------------- Report ----------------
    lines = [
        f"Permutation importance: mean drop in ROC-AUC when a feature is shuffled "
        f"in the held-out fold (5-fold stratified CV, {N_REPEATS} shuffles per fold, "
        f"seed {SEED}, n={len(X)} complete records). Higher means the model relies "
        f"on it more. Logistic Regression coefficients are on standardized features: "
        f"positive raises the predicted risk, negative lowers it.",
        "",
        "| Feature | Random Forest | Logistic Regression | LR coefficient |",
        "| --- | --- | --- | --- |",
    ]
    rf_imp, lr_imp = importances["Random Forest"], importances["Logistic Regression"]
    for f in order:
        lines.append(
            f"| {FEATURE_NAMES.get(f, f)} "
            f"| {rf_imp.loc[f, 'mean']:.3f} ± {rf_imp.loc[f, 'std']:.3f} "
            f"| {lr_imp.loc[f, 'mean']:.3f} ± {lr_imp.loc[f, 'std']:.3f} "
            f"| {coefs[f]:+.2f} |")

    top_rf = set(rf_imp["mean"].nlargest(5).index)
    top_lr = set(lr_imp["mean"].nlargest(5).index)
    shared = [FEATURE_NAMES[f] for f in order if f in top_rf & top_lr]
    lines += ["", f"Features in the top five for both models: {', '.join(shared)}."]

    report = "\n".join(lines) + "\n"
    os.makedirs("results", exist_ok=True)
    with open("results/feature_importance.md", "w") as fh:
        fh.write(report)
    print(report)

    # ---------------- Plot ----------------
    labels = [FEATURE_NAMES.get(f, f) for f in order][::-1]
    ypos = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
    for ax, (name, imp) in zip(axes, [("Random Forest", rf_imp),
                                      ("Logistic Regression", lr_imp)]):
        vals = imp.loc[order[::-1]]
        ax.barh(ypos, vals["mean"], xerr=vals["std"], color="#1B3358",
                ecolor="#8A97A8", capsize=2)
        ax.axvline(0, color="#999999", linewidth=0.8)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Drop in ROC-AUC when shuffled")
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(labels)
    fig.suptitle("Permutation feature importance (held-out folds)", fontsize=12)
    plt.tight_layout()
    plt.savefig("results/feature_importance.png", dpi=150)


if __name__ == "__main__":
    main()
