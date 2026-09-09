"""Compare regression models for the Black Friday purchase dataset.

Run from the project directory after placing train.csv and test.csv in data/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
RANDOM_STATE = 42


def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, test


def build_preprocessor(X):
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )


def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
    }


def main():
    train, test = load_data()

    target = "Purchase"
    drop_columns = ["User_ID", "Product_ID"]
    X = train.drop(columns=[target] + drop_columns)
    y = train[target]
    X_test = test.drop(columns=drop_columns)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=150,
            max_depth=18,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    results = []
    fitted_models = {}

    for name, estimator in models.items():
        pipeline = Pipeline(
            [("preprocessor", build_preprocessor(X_train)), ("model", estimator)]
        )
        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_valid)
        result = {"Model": name, **metrics(y_valid, prediction)}
        results.append(result)
        fitted_models[name] = pipeline

    results_df = pd.DataFrame(results).sort_values("RMSE")
    OUTPUT_DIR.mkdir(exist_ok=True)
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print(results_df.to_string(index=False))

    # Refit each model on all labelled data and create test predictions.
    prediction_columns = {"User_ID": test["User_ID"], "Product_ID": test["Product_ID"]}
    for name, estimator in models.items():
        pipeline = Pipeline(
            [("preprocessor", build_preprocessor(X)), ("model", estimator)]
        )
        pipeline.fit(X, y)
        prediction_columns[name] = pipeline.predict(X_test)

    pd.DataFrame(prediction_columns).to_csv(
        OUTPUT_DIR / "model_predictions.csv", index=False
    )


if __name__ == "__main__":
    main()
