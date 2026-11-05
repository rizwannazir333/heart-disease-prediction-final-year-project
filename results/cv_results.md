Mean ± standard deviation across folds. Random Forest and Logistic Regression: 5-fold stratified CV repeated 3 times (15 folds). ANN: 5-fold stratified CV (5 folds). Seed 42.

| Strategy | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Drop incomplete rows (n=298) | Logistic Regression | 0.833 ± 0.040 | 0.841 ± 0.031 | 0.790 ± 0.093 | 0.812 ± 0.055 | 0.902 ± 0.046 |
| Drop incomplete rows (n=298) | Random Forest | 0.822 ± 0.057 | 0.839 ± 0.068 | 0.763 ± 0.085 | 0.798 ± 0.068 | 0.903 ± 0.037 |
| Median imputation (n=597) | Logistic Regression | 0.839 ± 0.031 | 0.826 ± 0.047 | 0.771 ± 0.048 | 0.797 ± 0.039 | 0.897 ± 0.023 |
| Median imputation (n=597) | Random Forest | 0.829 ± 0.019 | 0.822 ± 0.036 | 0.747 ± 0.045 | 0.781 ± 0.026 | 0.895 ± 0.021 |
