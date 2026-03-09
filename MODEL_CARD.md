# Medical Insurance Pricing Model

## Model Overview

**Model Type:** Random Forest Regressor
**Task:** Regression
**Objective:** Predict medical insurance charges based on demographic and lifestyle features.

This model estimates expected healthcare insurance costs using structured tabular data.

---

# Dataset

**Dataset Size:** 1,338 records

### Features

| Feature  | Description           |
| -------- | --------------------- |
| age      | Age of the individual |
| sex      | Gender                |
| bmi      | Body Mass Index       |
| children | Number of dependents  |
| smoker   | Smoking status        |
| region   | Geographic region     |

### Target

| Target  | Description            |
| ------- | ---------------------- |
| charges | Medical insurance cost |

---

# Data Preprocessing

The following preprocessing steps are applied during training:

* Removal of duplicate records
* Validation of required columns
* Conversion of numeric variables
* One-hot encoding of categorical features
* Log transformation of the target variable (`charges`)
* Train/test split for model evaluation

These steps are implemented within a **scikit-learn pipeline** to ensure reproducibility.

---

# Model Architecture

The best performing model is a **Random Forest Regressor**.

Key configuration:

* `n_estimators = 400`
* `min_samples_split = 5`
* `min_samples_leaf = 2`
* `random_state = 42`

Tree-based models were selected because they capture **nonlinear relationships and feature interactions** present in insurance risk data.

---

# Model Performance

Performance was evaluated on a held-out test dataset.

| Metric | Score    |
| ------ | -------- |
| R²     | **0.87** |
| RMSE   | **0.34** |
| MAE    | **0.18** |

Random Forest achieved the best performance among the evaluated models.

---

# Feature Importance

The most influential predictors identified by the model include:

1. Smoking status
2. Age
3. Body Mass Index (BMI)

These variables have the strongest relationship with insurance costs.

---

# Intended Use

This model is intended for:

* Educational purposes
* Demonstrating end-to-end ML pipelines
* Insurance cost estimation experiments

It is **not intended for production insurance pricing or medical decision-making**.

---

# Limitations

* Small dataset size (1,338 records)
* No healthcare history or medical condition variables
* Limited geographic information
* Potential bias due to simplified demographic features

---

# Future Improvements

Potential improvements include:

* Hyperparameter optimisation
* Model explainability using SHAP
* Additional healthcare-related features
* Monitoring model drift in production environments
