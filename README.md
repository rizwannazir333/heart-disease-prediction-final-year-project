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

## Dataset

`heart.csv` has 597 patient records with 13 clinical features and a target graded 0 (no disease) to 4. Checked directly from the file:

| | Records |
| --- | --- |
| Total records | 597 |
| Records with at least one missing value | 299 |
| Complete records used for training and testing | 298 |
| Complete records, no disease (0) / disease (1 to 4) | 160 / 138 |

Because about half of the records are dropped, the 20% test set holds only around 60 patients, so single-split accuracy figures are noisy. Imputing missing values instead of dropping them is a planned improvement.

## How it works

1. **Data cleaning:** any row with a missing value (`?` or blank) is dropped, and all columns are converted to numeric. See the Dataset section for how much data this removes.
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
├── evaluate.py         # Seeded, reproducible evaluation (metrics + confusion matrices)
├── test_data.py        # Dataset sanity checks (run with pytest)
├── MODEL_CARD.md       # Intended use, data, limitations
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

2. Make sure MySQL is running. The app reads its settings from environment variables, and the defaults suit a local setup with an empty root password. The `heart_db` database and `history` table are created automatically on first run.

   | Variable | Default | Purpose |
   | --- | --- | --- |
   | `DB_HOST` / `DB_USER` / `DB_PASSWORD` / `DB_NAME` | `localhost` / `root` / empty / `heart_db` | MySQL connection |
   | `ADMIN_USERNAME` / `ADMIN_PASSWORD` | `admin` / `123` | Demo login |
   | `FLASK_DEBUG` | `1` | Set to `0` to turn off debug mode |

   Example: `export DB_PASSWORD="your-password" ADMIN_PASSWORD="a-stronger-password"`

3. (Optional) Retrain the models. Trained models are already included, so this is only needed if you change the data or model settings:

   ```bash
   python train_model.py
   ```

   This prints the test accuracy of each model and saves the results.

4. Start the app from the project folder:

   ```bash
   python app.py
   ```

   Then open `http://127.0.0.1:5000`. Unless you set `ADMIN_USERNAME` and `ADMIN_PASSWORD`, the demo login is `admin` / `123`.

5. Run the data checks (they need only Python, no TensorFlow or MySQL):

   ```bash
   pytest
   ```

## Evaluation

`evaluate.py` repeats the cleaning and 80/20 split from `train_model.py` with every random seed fixed (seed 42), so results can be reproduced exactly. It reports accuracy, precision, recall, F1 and ROC-AUC for Logistic Regression, Random Forest, the ANN and the averaged ensemble, and saves the results to `results/metrics.md` and `results/confusion_matrices.png`.

```bash
python evaluate.py
```

For a screening task like this, recall (how many true disease cases are caught) matters at least as much as accuracy, so check the confusion matrices rather than accuracy alone.

## Responsible use

- Predictions are probabilities from small models trained on limited data. They support learning and discussion, not clinical decisions.
- Model performance has not been checked across age or sex groups, or on data from other hospitals or countries, so it may not generalize.
- Patient names and inputs are stored in a local MySQL database. A real deployment would need consent, access control and data-protection compliance (for example GDPR in Europe).

## Limitations

- The login is a single shared demo account with a plain-text password, not real authentication. Do not deploy this as is.
- Training uses a single train/test split and no hyperparameter search, so reported accuracy is a rough estimate.
- Only 298 complete records are used and the test set is about 60 patients, so results will vary a lot between splits.
- The three-level risk bands are a simple threshold on the averaged probability, not a trained multiclass model.

## Planned improvements

- Stratified k-fold cross-validation and hyperparameter search instead of a single split
- Probability calibration and a proper multiclass or ordinal model for risk levels
- Explainability (for example SHAP feature importance) so predictions can be inspected
- Fairness checks across age and sex, and validation on an external dataset
- Impute missing values instead of dropping about half of the records
- Real authentication with hashed passwords and per-user accounts

## Author

**Rizwan Ali**, Entry-Level AI Engineer  
[GitHub](https://github.com/rizwannazir333) · [LinkedIn](https://linkedin.com/in/rizwan-ali-ai-engineer)
