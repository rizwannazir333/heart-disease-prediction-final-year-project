# Heart Disease Prediction Using Machine Learning and Deep Learning

A healthcare AI web application that estimates a patient's heart disease risk from 13 clinical measurements. It combines three models (Logistic Regression, Random Forest, and a neural network built in TensorFlow/Keras) and reports a **Low / Medium / High** risk level together with each model's probability.

Final year project, BSc Software Engineering, Virtual University of Pakistan.

> **Disclaimer:** this is an academic project, not a medical device. It must not be used for real diagnosis or treatment decisions.

## Features

- Three models trained on the same preprocessed data and compared side by side
- Neural network (ANN) with dropout regularization
- Averaged-probability risk score mapped to Low, Medium and High
- Flask web app with a login page, dashboard, prediction form, result page and about page
- Prediction history stored in MySQL, with a page to review and delete records

## How it works

1. **Data cleaning:** the dataset (`heart.csv`, 13 clinical features plus a target column) has `?` placeholders replaced with missing values; incomplete rows are dropped and all columns are converted to numeric.
2. **Target:** the original target (0 to 4) is converted to a binary label: `0` means no disease, anything above 0 means disease.
3. **Split and scaling:** an 80/20 train/test split (`random_state=42`), then `StandardScaler` fitted on the training set only.
4. **Models:**
   - Logistic Regression
   - Random Forest
   - ANN: `Dense(16, relu)` → `Dropout(0.3)` → `Dense(8, relu)` → `Dense(1, sigmoid)`, trained with Adam and binary cross-entropy for 50 epochs (batch size 16)
5. **Prediction:** the three predicted probabilities are averaged. Below 30% is Low risk, 30% to 70% is Medium, and 70% or above is High.

### Input features

`age`, `sex`, `cp` (chest pain type), `trestbps` (resting blood pressure), `chol` (cholesterol), `fbs` (fasting blood sugar), `restecg` (resting ECG), `thalach` (max heart rate), `exang` (exercise-induced angina), `oldpeak` (ST depression), `slope`, `ca` (major vessels), `thal`

## Tech stack

Python, TensorFlow/Keras, scikit-learn, Pandas, NumPy, Flask, MySQL, HTML/CSS (Jinja templates)

## Project structure

```text
├── app.py              # Flask app: routes, MySQL history
├── models.py           # Loads trained models and makes predictions
├── train_model.py      # Cleans data, trains and saves all three models
├── heart.csv           # Dataset
├── *_model.pkl / ann_model.h5, scaler.pkl, *_accuracy.pkl   # Saved models and scores
├── templates/          # HTML pages
├── static/             # Images
└── requirements.txt
```

## Getting started

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Make sure MySQL is running, then set your MySQL username and password in `DB_CONFIG` in `app.py`. The `heart_db` database and `history` table are created automatically on first run.

3. (Optional) Retrain the models. Trained models are already included, so this is only needed if you change the data or model settings:

   ```bash
   python train_model.py
   ```

   This prints the test accuracy of each model and saves the results.

4. Start the app from the project folder:

   ```bash
   python app.py
   ```

   Then open `http://127.0.0.1:5000`. The demo login is `admin` / `123`.

## Limitations

- The login is a hard-coded demo account, not real authentication, and the database password is set in the source file. Do not deploy this as is.
- Training uses a single train/test split and no hyperparameter search, so the reported accuracy is a rough estimate.
- The dataset is small, and the three-level risk bands are a simple threshold on the averaged probability rather than a trained multiclass model.

## Author

**Rizwan Ali**, Entry-Level AI Engineer  
[GitHub](https://github.com/rizwannazir333) · [LinkedIn](https://linkedin.com/in/rizwan-ali-ai-engineer)
