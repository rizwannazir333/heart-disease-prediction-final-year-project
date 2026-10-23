"""Reproducible evaluation of the three heart disease models.

Uses the same cleaning and 80/20 split as train_model.py, but fixes every
random seed so the reported numbers can be reproduced exactly.
Writes metrics and confusion matrix plots to the results/ folder.

Run:  python evaluate.py
"""
import os
import random

import numpy as np
import pandas as pd

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

tf.random.set_seed(SEED)

# ---------------- Data (same cleaning as train_model.py) ----------------
df = pd.read_csv("heart.csv")
df.replace("?", pd.NA, inplace=True)
rows_before = len(df)
df.dropna(inplace=True)
df = df.apply(pd.to_numeric)
rows_after = len(df)

X = df.drop("target", axis=1)
y = df["target"].apply(lambda v: 1 if v > 0 else 0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- Models ----------------
lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
rf = RandomForestClassifier(random_state=SEED).fit(X_train, y_train)

ann = Sequential([
    Dense(16, activation="relu", input_dim=X_train.shape[1]),
    Dropout(0.3),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid"),
])
ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
ann.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

probs = {
    "Logistic Regression": lr.predict_proba(X_test)[:, 1],
    "Random Forest": rf.predict_proba(X_test)[:, 1],
    "ANN (dropout)": ann.predict(X_test, verbose=0).ravel(),
}
probs["Ensemble (average)"] = np.mean(list(probs.values()), axis=0)

# ---------------- Metrics ----------------
os.makedirs("results", exist_ok=True)
lines = [
    "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
    "| --- | --- | --- | --- | --- | --- |",
]
fig, axes = plt.subplots(1, len(probs), figsize=(4 * len(probs), 3.8))
for ax, (name, p) in zip(axes, probs.items()):
    pred = (p >= 0.5).astype(int)
    lines.append(
        f"| {name} | {accuracy_score(y_test, pred):.3f} "
        f"| {precision_score(y_test, pred):.3f} "
        f"| {recall_score(y_test, pred):.3f} "
        f"| {f1_score(y_test, pred):.3f} "
        f"| {roc_auc_score(y_test, p):.3f} |"
    )
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred),
                           display_labels=["No disease", "Disease"]
                           ).plot(ax=ax, colorbar=False)
    ax.set_title(name, fontsize=10)
plt.tight_layout()
plt.savefig("results/confusion_matrices.png", dpi=150)

summary = (
    f"Rows in heart.csv: {rows_before}; after dropping incomplete rows: {rows_after}\n"
    f"Train/test: {len(y_train)}/{len(y_test)} (seed {SEED}); "
    f"positive class share in test set: {y_test.mean():.3f}\n\n"
    + "\n".join(lines) + "\n"
)
with open("results/metrics.md", "w") as f:
    f.write(summary)
print(summary)
