Permutation importance: mean drop in ROC-AUC when a feature is shuffled in the held-out fold (5-fold stratified CV, 20 shuffles per fold, seed 42, n=298 complete records). Higher means the model relies on it more. Logistic Regression coefficients are on standardized features: positive raises the predicted risk, negative lowers it.

| Feature | Random Forest | Logistic Regression | LR coefficient |
| --- | --- | --- | --- |
| Major vessels (ca) | 0.055 ± 0.010 | 0.070 ± 0.018 | +1.10 |
| Chest pain type | 0.031 ± 0.018 | 0.013 ± 0.010 | +0.54 |
| Thalassemia (thal) | 0.027 ± 0.022 | 0.028 ± 0.016 | +0.65 |
| ST depression (oldpeak) | 0.017 ± 0.011 | 0.003 ± 0.003 | +0.30 |
| Sex | 0.015 ± 0.008 | 0.021 ± 0.011 | +0.58 |
| Age | 0.011 ± 0.003 | -0.001 ± 0.002 | -0.10 |
| Exercise-induced angina | 0.007 ± 0.003 | 0.008 ± 0.005 | +0.42 |
| Max heart rate | 0.004 ± 0.012 | 0.009 ± 0.007 | -0.46 |
| Resting blood pressure | 0.003 ± 0.002 | 0.007 ± 0.008 | +0.39 |
| Resting ECG | 0.002 ± 0.002 | 0.000 ± 0.005 | +0.24 |
| ST slope | 0.002 ± 0.007 | 0.004 ± 0.003 | +0.32 |
| Cholesterol | 0.001 ± 0.005 | -0.003 ± 0.007 | +0.24 |
| Fasting blood sugar | -0.001 ± 0.002 | 0.003 ± 0.009 | -0.32 |

Features in the top five for both models: Major vessels (ca), Chest pain type, Thalassemia (thal), Sex.
