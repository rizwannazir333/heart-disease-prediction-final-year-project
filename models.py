import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
lr_model = joblib.load("lr_model.pkl")
rf_model = joblib.load("rf_model.pkl")

# ANN is optional (loaded only if the file exists)
try:
    from tensorflow.keras.models import load_model
    ann_model = load_model("ann_model.h5")
except Exception:
    ann_model = None


def predict(data):
    data = scaler.transform(np.array(data).reshape(1, -1))

    lr_prob = float(lr_model.predict_proba(data)[0][1])
    rf_prob = float(rf_model.predict_proba(data)[0][1])

    probs = [lr_prob, rf_prob]
    ann_prob = None
    if ann_model is not None:
        ann_prob = float(ann_model.predict(data, verbose=0)[0][0])
        probs.append(ann_prob)

    avg_prob = sum(probs) / len(probs)

    # Multi-class risk: Low / Medium / High
    if avg_prob < 0.3:
        risk = "Low"
    elif avg_prob < 0.7:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "risk": risk,
        "avg_prob": round(avg_prob * 100, 2),
        "lr_prob": round(lr_prob * 100, 2),
        "rf_prob": round(rf_prob * 100, 2),
        "ann_prob": round(ann_prob * 100, 2) if ann_prob is not None else None,
    }
