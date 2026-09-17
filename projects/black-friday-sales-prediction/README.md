# Black Friday Sales Prediction

## Overview

This project analyses customer purchasing behaviour during Black Friday and compares multiple regression approaches to predict the `Purchase` amount.

The analysis uses the original Black Friday `train.csv` and `test.csv` supplied for the project. No synthetic records are created.

## Dataset

| File | Rows | Columns | Target |
|---|---:|---:|---|
| `train.csv` | 550,068 | 12 | `Purchase` |
| `test.csv` | 233,599 | 11 | Not provided |

Key fields include customer identifiers, demographics, city information, marital status and product categories. `Purchase` is the target variable in the training data.

The raw CSV files are intentionally excluded from this public repository. See `data/README.md` for the local data setup.

## Methodology

1. Load and inspect the original train and test datasets.
2. Check data types, missing values and basic purchase distributions.
3. Remove `User_ID` and `Product_ID` from the predictive feature set because they are identifiers.
4. Treat demographic and product-category fields as categorical variables.
5. Handle missing values through preprocessing pipelines so imputation is learned from the training data only.
6. Standardise numeric features and one-hot encode categorical features with unknown-category handling.
7. Validate required identifiers, target type, non-empty inputs and train/test feature compatibility before modelling.
8. Split the labelled training data 80/20 using `random_state=42`.
9. Compare `LinearRegression`, `Ridge` and `RandomForestRegressor` using the same preprocessing architecture.
10. Evaluate using R², MAE, MSE and RMSE.
11. Refit each candidate model on all labelled training data and generate predictions for the original test set.

## Validation results

The original validated baseline produced the following results on the 20% hold-out set:

| Metric | Result |
|---|---:|
| R² | **0.642016** |
| Adjusted R² | **0.641739** |
| MAE | **2,261.97** |
| MSE | **8,994,780.54** |
| RMSE | **2,999.13** |
| Encoded predictors | **85** |

The R² of approximately 0.642 means the baseline model explains about 64% of the variation in held-out purchase values under that feature representation. The model-comparison workflow now provides a reusable foundation for benchmarking stronger estimators without duplicating preprocessing logic.

## Repository contents

```text
black-friday-sales-prediction/
├── README.md
├── data/
│   └── README.md
├── notebooks/
│   └── black_friday_sales_prediction.ipynb
├── src/
│   ├── README.md
│   ├── train_model.py
│   └── model_comparison.py
├── tests/
│   └── test_model_comparison.py
├── outputs/
│   └── (generated locally; ignored by Git)
└── requirements.txt
```

## Reproducibility

Place the original files locally as:

```text
data/
├── train.csv
└── test.csv
```

For the baseline notebook/script workflow, run:

```bash
python src/train_model.py
```

For model comparison:

```bash
python src/model_comparison.py
```

Run the automated unit tests with:

```bash
pytest tests
```

The scripts write generated prediction/model-comparison outputs locally. These generated files are excluded from Git so the public repository remains lightweight.

## Quality checks

The model-comparison workflow includes unit tests for metric calculations, preprocessing, model fitting, unseen categories, input schema validation, target validation and train/test feature compatibility. GitHub Actions runs the test suite across supported Python versions.

## Future improvements

- Residual and assumption diagnostics
- Cross-validation and confidence intervals
- Lasso and elastic-net regularisation
- Feature importance and model interpretability
- Hyperparameter tuning with a reproducible validation strategy
- Experiment tracking and model versioning

## Disclaimer

The dataset is used for analysis and learning. Check the dataset's current licence and terms before redistributing the raw CSV files.
