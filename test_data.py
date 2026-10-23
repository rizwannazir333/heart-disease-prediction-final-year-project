"""Basic data checks for heart.csv.

Uses only the standard library, so it runs under pytest (as in CI) or
directly with:  python test_data.py
"""
import csv
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "heart.csv")

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]


def load_clean_rows():
    """Mirror the cleaning in train_model.py: drop rows with '?' or blank values."""
    with open(DATA_PATH, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = [r for r in reader
                if not any(v in ("?", "") for v in r.values())]
    return header, rows


def test_expected_columns():
    header, _ = load_clean_rows()
    assert header == FEATURES + ["target"]


def test_enough_rows_after_cleaning():
    _, rows = load_clean_rows()
    assert len(rows) > 100


def test_all_values_numeric():
    _, rows = load_clean_rows()
    for row in rows:
        for value in row.values():
            float(value)


def test_binary_label_has_both_classes():
    _, rows = load_clean_rows()
    labels = {1 if float(r["target"]) > 0 else 0 for r in rows}
    assert labels == {0, 1}


def test_age_and_sex_ranges():
    _, rows = load_clean_rows()
    assert all(18 <= float(r["age"]) <= 100 for r in rows)
    assert {float(r["sex"]) for r in rows} <= {0.0, 1.0}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("passed:", name)
