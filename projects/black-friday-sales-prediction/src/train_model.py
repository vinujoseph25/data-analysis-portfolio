"""Black Friday multivariable linear regression.

Place the original Kaggle train.csv and test.csv in ../data/ and run this script.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
TRAIN_PATH = DATA / "train.csv"
TEST_PATH = DATA / "test.csv"

CAT_COLS = [
    "Gender", "Age", "Occupation", "City_Category",
    "Stay_In_Current_City_Years", "Marital_Status",
    "Product_Category_1", "Product_Category_2", "Product_Category_3",
]


def prepare(df):
    out = df.drop(columns=["User_ID", "Product_ID"], errors="ignore").copy()
    for col in CAT_COLS:
        out[col] = out[col].astype("string").fillna("Missing")
    return out


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    X = prepare(train.drop(columns=["Purchase"]))
    y = train["Purchase"]
    X_test = prepare(test)

    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT_COLS)
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model.fit(X_train, y_train)
    valid_pred = model.predict(X_valid)

    r2 = r2_score(y_valid, valid_pred)
    n = len(y_valid)
    p = model.named_steps["preprocessor"].transform(X_train).shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    mae = mean_absolute_error(y_valid, valid_pred)
    mse = mean_squared_error(y_valid, valid_pred)
    rmse = np.sqrt(mse)

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Encoded predictors: {p}")
    print(f"R2: {r2:.6f}")
    print(f"Adjusted R2: {adjusted_r2:.6f}")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")

    # Refit on all labelled training observations before predicting Kaggle test.
    model.fit(X, y)
    predictions = model.predict(X_test)
    submission = test[["User_ID", "Product_ID"]].copy()
    submission["Predicted_Purchase"] = predictions
    OUTPUTS.mkdir(exist_ok=True)
    path = OUTPUTS / "black_friday_test_predictions.csv"
    submission.to_csv(path, index=False)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
