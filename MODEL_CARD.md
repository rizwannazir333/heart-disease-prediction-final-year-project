# Model Card: Heart Disease Risk Prediction

## Overview

Three models estimate whether a patient has heart disease from 13 clinical measurements, and their probabilities are averaged into a Low / Medium / High risk level.

| | |
| --- | --- |
| **Models** | Logistic Regression, Random Forest, ANN (Dense 16 → Dropout 0.3 → Dense 8 → Dense 1) |
| **Task** | Binary classification: no disease (0) vs disease (target 1 to 4) |
| **Output** | Probability from each model, averaged; Low below 30%, Medium 30% to 70%, High 70% or above |
| **Frameworks** | scikit-learn, TensorFlow/Keras |
| **Author** | Rizwan Ali |

## Intended use

- **Intended:** learning, demonstration, and research experiments on clinical risk prediction.
- **Not intended:** diagnosis, treatment decisions, or any use on real patients without independent clinical validation.

## Data

- `heart.csv`: 597 records, 13 features (`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`), target graded 0 to 4.
- 299 records contain missing values and are dropped, leaving 298 complete records (160 no disease, 138 disease).
- Split: 80% training, 20% test (`random_state=42`), features standardized with a scaler fitted on the training set only.

## Evaluation

Run `python evaluate.py` to produce accuracy, precision, recall, F1, ROC-AUC and confusion matrices for each model and the ensemble, saved in `results/`. The test set is small (about 60 patients), so treat any single result as a rough estimate.

Cross-validated results from `cross_validate.py` (15 folds, seed 42), on the 298 complete records:

| Model | Accuracy | Recall | ROC-AUC |
| --- | --- | --- | --- |
| Logistic Regression | 0.833 ± 0.040 | 0.790 ± 0.093 | 0.902 ± 0.046 |
| Random Forest | 0.822 ± 0.057 | 0.763 ± 0.085 | 0.903 ± 0.037 |

Median imputation on all 597 records gives similar averages with about half the variability between folds.

## Explainability

`explain.py` measures permutation importance on held-out folds. Both models rely most on the number of major vessels, chest pain type, thalassemia and sex. Explanations are global (whole dataset), not per patient.

## Limitations and risks

- Small dataset with half of the records removed for missing values, which may bias the remaining sample.
- The deployed models use a single split, with no calibration or hyperparameter search (cross-validated estimates are reported above).
- No analysis of performance across age or sex groups, and no external validation.
- The risk bands are fixed probability thresholds and have not been validated against clinical risk categories.
- A false negative (missing a patient with disease) is the more serious error in screening, so recall should be checked alongside accuracy.

## Ethical considerations

Health data is sensitive. The web app stores patient names and inputs in MySQL without encryption, consent handling or proper access control, so it is suitable only for local demonstration.
