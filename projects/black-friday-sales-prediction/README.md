# Black Friday Sales Prediction

## Overview

This project analyses customer purchasing behaviour during Black Friday and builds a **multivariable linear regression** model to predict the `Purchase` amount.

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
5. Preserve missing values in `Product_Category_2` and `Product_Category_3` using an explicit `Missing` category rather than deleting observations.
6. One-hot encode categorical variables with a reference category removed.
7. Split the labelled training data 80/20 using `random_state=42`.
8. Train `sklearn.linear_model.LinearRegression`.
9. Evaluate using R², adjusted R², MAE, MSE and RMSE.
10. Refit the model on all labelled training data and generate predictions for the original 233,599-row test set.

## Validation results

The validated baseline produced the following results on the 20% hold-out set:

| Metric | Result |
|---|---:|
| R² | **0.642016** |
| Adjusted R² | **0.641739** |
| MAE | **2,261.97** |
| MSE | **8,994,780.54** |
| RMSE | **2,999.13** |
| Encoded predictors | **85** |

The R² of approximately 0.642 means the model explains about 64% of the variation in held-out purchase values under this feature representation.

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
│   └── train_model.py
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

Then run the notebook or:

```bash
python src/train_model.py
```

The script writes `outputs/black_friday_test_predictions.csv` locally. The generated prediction file is excluded from Git so the public repository remains lightweight.

## Future improvements

- Residual and assumption diagnostics
- Cross-validation
- Regularised regression (Ridge/Lasso)
- Comparison with tree-based regression models
- Hyperparameter tuning and model comparison

## Disclaimer

The dataset is used for analysis and learning. Check the dataset's current licence and terms before redistributing the raw CSV files.
