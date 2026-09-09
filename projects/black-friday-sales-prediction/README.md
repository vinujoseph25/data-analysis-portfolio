# Black Friday Sales Prediction

## Overview

This project analyses customer purchasing behaviour during Black Friday and builds a multivariable linear regression model to predict the `Purchase` amount.

The project uses the original Kaggle Black Friday `train.csv` and `test.csv` datasets. No synthetic or fabricated records are used.

## Dataset

The dataset contains customer, demographic and product-category information. The training data contains the target column `Purchase`; the Kaggle test data does not.

Key fields include:

- `User_ID` and `Product_ID` — identifiers
- `Gender`, `Age`, `Occupation`
- `City_Category`, `Stay_In_Current_City_Years`, `Marital_Status`
- `Product_Category_1`, `Product_Category_2`, `Product_Category_3`
- `Purchase` — prediction target in the training data

The training dataset contains 550,068 rows and 12 columns.

## Approach

1. Load the original training and test datasets.
2. Inspect data types, missing values and distributions.
3. Apply consistent cleaning and feature engineering to both datasets.
4. Remove identifier fields from the predictive feature set.
5. Treat `Age` and `Stay_In_Current_City_Years` as ordered categories.
6. One-hot encode nominal categorical variables.
7. Preserve information about missing product sub-categories through a `Num_Categories` feature.
8. Split the labelled training data into model-training and validation subsets.
9. Fit a multivariable `LinearRegression` model.
10. Evaluate with R², adjusted R², MAE, MSE and RMSE.
11. Inspect model coefficients and statistical significance.
12. Apply the trained model to the Kaggle `test.csv` records to generate purchase predictions.

## Initial model result

Using the current cleaned feature design and an 80/20 split with `random_state=42`, the existing analysis produced approximately:

| Metric | Result |
|---|---:|
| R² | 0.6338 |
| Adjusted R² | 0.6337 |
| MAE | 2,292.73 |
| RMSE | 3,033.27 |

These figures are a baseline and should be reproduced by running the notebook rather than treated as immutable project results.

## Repository contents

- `notebooks/black_friday_sales_prediction.ipynb` — end-to-end analysis
- `data/README.md` — dataset acquisition and local file instructions
- `src/README.md` — notes on reusable source code
- `requirements.txt` — Python dependencies

## Reproducibility

Place the original Kaggle files in `data/` as:

```text
data/
├── train.csv
└── test.csv
```

Then run the notebook from the project directory.

## Disclaimer

The dataset is referenced for analysis and learning. Check the dataset's current Kaggle licence and terms before redistributing the raw CSV files.
