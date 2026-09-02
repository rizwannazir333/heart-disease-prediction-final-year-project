import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ---------------- Load dataset ----------------
df = pd.read_csv("heart.csv")

# ---------------- Clean ----------------
df.replace("?", pd.NA, inplace=True)
df.dropna(inplace=True)
df = df.apply(pd.to_numeric)

# ---------------- Features & Target ----------------
X = df.drop("target", axis=1)
y = df["target"].apply(lambda v: 1 if v > 0 else 0)   # 0 = no disease, 1 = disease

# ---------------- Split & Scale ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")

# ---------------- Logistic Regression ----------------
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
joblib.dump(lr, "lr_model.pkl")
lr_acc = accuracy_score(y_test, lr.predict(X_test))
joblib.dump(lr_acc, "lr_accuracy.pkl")
print("Logistic Regression Accuracy:", lr_acc)

# ---------------- Random Forest ----------------
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
joblib.dump(rf, "rf_model.pkl")
rf_acc = accuracy_score(y_test, rf.predict(X_test))
joblib.dump(rf_acc, "rf_accuracy.pkl")
print("Random Forest Accuracy:", rf_acc)

# ---------------- ANN (with Dropout) ----------------
ann = Sequential()
ann.add(Dense(16, activation='relu', input_dim=X_train.shape[1]))
ann.add(Dropout(0.3))
ann.add(Dense(8, activation='relu'))
ann.add(Dense(1, activation='sigmoid'))
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

loss, ann_acc = ann.evaluate(X_test, y_test, verbose=0)
joblib.dump(ann_acc, "ann_accuracy.pkl")
ann.save("ann_model.h5")
print("ANN Accuracy:", ann_acc)

print("Models + accuracy saved!")
