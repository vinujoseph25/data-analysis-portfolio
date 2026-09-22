"""Compare regression models for the Black Friday purchase dataset.

Run from the project directory after placing train.csv and test.csv in data/.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
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
VALIDATION_SIZE = 0.20
TARGET = "Purchase"
ID_COLUMNS = ["User_ID", "Product_ID"]
REQUIRED_COLUMNS = ID_COLUMNS + [TARGET]


def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, test


def validate_input_schema(train, test):
    """Validate the minimum schema and feature compatibility for the workflow."""
    for name, frame in (("Training", train), ("Test", test)):
        if not frame.columns.is_unique:
            duplicates = frame.columns[frame.columns.duplicated()].tolist()
            raise ValueError(f"{name} data contains duplicate columns: {duplicates}")

    missing_train = [column for column in REQUIRED_COLUMNS if column not in train.columns]
    missing_test = [column for column in ID_COLUMNS if column not in test.columns]
    if missing_train:
        raise ValueError(f"Training data is missing required columns: {missing_train}")
    if missing_test:
        raise ValueError(f"Test data is missing required columns: {missing_test}")
    if train.empty:
        raise ValueError("Training data must contain at least one row")
    if test.empty:
        raise ValueError("Test data must contain at least one row")
    if not pd.api.types.is_numeric_dtype(train[TARGET]):
        raise ValueError(f"Training target '{TARGET}' must be numeric")
    if train[TARGET].isna().any():
        raise ValueError(f"Training target '{TARGET}' must not contain missing values")

    train_features = set(train.columns) - set(REQUIRED_COLUMNS)
    test_features = set(test.columns) - set(ID_COLUMNS)
    if train_features != test_features:
        missing_in_test = sorted(train_features - test_features)
        extra_in_test = sorted(test_features - train_features)
        raise ValueError(
            "Training and test features do not match: "
            f"missing_in_test={missing_in_test}, extra_in_test={extra_in_test}"
        )


def build_preprocessor(X):
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipeline, numeric), ("categorical", categorical_pipeline, categorical)],
        remainder="drop",
    )


def build_model_pipeline(X, estimator):
    """Create a preprocessing + estimator pipeline for consistent model training."""
    return Pipeline([("preprocessor", build_preprocessor(X)), ("model", estimator)])


def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"R2": r2_score(y_true, y_pred), "MAE": mean_absolute_error(y_true, y_pred), "MSE": mse, "RMSE": np.sqrt(mse)}


def add_baseline_lift(results_df):
    """Add model improvement metrics relative to the naive mean baseline."""
    baseline = results_df.loc[results_df["Model"] == "DummyMean"]
    if baseline.empty:
        raise ValueError("Model comparison requires a DummyMean baseline")
    baseline_rmse = baseline["RMSE"].iloc[0]
    baseline_r2 = baseline["R2"].iloc[0]
    results = results_df.copy()
    results["RMSE_Improvement_vs_Dummy"] = baseline_rmse - results["RMSE"]
    results["R2_Improvement_vs_Dummy"] = results["R2"] - baseline_r2
    return results


def select_best_model(results_df, exclude_models=("DummyMean",)):
    """Select the best predictive model by lowest validation RMSE."""
    candidates = results_df[~results_df["Model"].isin(exclude_models)]
    if candidates.empty:
        raise ValueError("No predictive models are available for selection")
    return candidates.sort_values(["RMSE", "MAE", "Model"], ascending=[True, True, True]).iloc[0]["Model"]


def build_run_metadata(results_df, best_model):
    """Build a machine-readable summary of the model-selection run."""
    selected = results_df.loc[results_df["Model"] == best_model].iloc[0]
    return {
        "target": TARGET,
        "selected_model": best_model,
        "candidate_models": results_df["Model"].tolist(),
        "baseline_model": "DummyMean",
        "random_state": RANDOM_STATE,
        "validation_size": VALIDATION_SIZE,
        "selected_metrics": {
            "R2": float(selected["R2"]),
            "MAE": float(selected["MAE"]),
            "MSE": float(selected["MSE"]),
            "RMSE": float(selected["RMSE"]),
        },
    }


def build_selected_prediction_frame(predictions_df, best_model):
    """Return stable identifiers plus the prediction from the selected model."""
    if best_model not in predictions_df.columns:
        raise ValueError(f"Selected model '{best_model}' is missing from predictions")
    required = ["User_ID", "Product_ID", best_model]
    missing = [column for column in required if column not in predictions_df.columns]
    if missing:
        raise ValueError(f"Prediction data is missing required columns: {missing}")
    selected = predictions_df[["User_ID", "Product_ID", best_model]].copy()
    return selected.rename(columns={best_model: "SelectedPrediction"})


def validate_prediction_output(test, predictions_df, best_model):
    """Validate prediction row count, identifiers, and selected predictions before export."""
    required = ["User_ID", "Product_ID", "SelectedModel", "SelectedPrediction"]
    missing = [column for column in required if column not in predictions_df.columns]
    if missing:
        raise ValueError(f"Prediction output is missing required columns: {missing}")
    if len(predictions_df) != len(test):
        raise ValueError(
            "Prediction row count does not match test data: "
            f"predictions={len(predictions_df)}, test={len(test)}"
        )
    if not predictions_df[["User_ID", "Product_ID"]].reset_index(drop=True).equals(
        test[["User_ID", "Product_ID"]].reset_index(drop=True)
    ):
        raise ValueError("Prediction identifiers do not match test identifiers")
    if predictions_df["SelectedModel"].nunique() != 1 or predictions_df["SelectedModel"].iloc[0] != best_model:
        raise ValueError("Prediction output contains an inconsistent selected model")
    if not pd.api.types.is_numeric_dtype(predictions_df["SelectedPrediction"]):
        raise ValueError("Selected predictions must be numeric")
    if not np.isfinite(predictions_df["SelectedPrediction"].to_numpy()).all():
        raise ValueError("Selected predictions must contain only finite values")


def main():
    train, test = load_data()
    validate_input_schema(train, test)
    X = train.drop(columns=REQUIRED_COLUMNS)
    y = train[TARGET]
    X_test = test.drop(columns=ID_COLUMNS)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE)

    models = {
        "DummyMean": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=150, max_depth=18, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE),
    }
    results = []
    for name, estimator in models.items():
        pipeline = build_model_pipeline(X_train, estimator)
        pipeline.fit(X_train, y_train)
        results.append({"Model": name, **metrics(y_valid, pipeline.predict(X_valid))})

    results_df = add_baseline_lift(pd.DataFrame(results).sort_values("RMSE"))
    best_model = select_best_model(results_df)
    results_df["Selected"] = results_df["Model"].eq(best_model)
    OUTPUT_DIR.mkdir(exist_ok=True)
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    with (OUTPUT_DIR / "model_run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(build_run_metadata(results_df, best_model), file, indent=2)
    print(results_df.to_string(index=False))
    print(f"Selected model: {best_model}")

    prediction_columns = {"User_ID": test["User_ID"], "Product_ID": test["Product_ID"]}
    for name, estimator in models.items():
        pipeline = build_model_pipeline(X, estimator)
        pipeline.fit(X, y)
        prediction_columns[name] = pipeline.predict(X_test)

    predictions_df = pd.DataFrame(prediction_columns)
    predictions_df["SelectedModel"] = best_model
    predictions_df["SelectedPrediction"] = predictions_df[best_model]
    validate_prediction_output(test, predictions_df, best_model)
    predictions_df.to_csv(OUTPUT_DIR / "model_predictions.csv", index=False)
    build_selected_prediction_frame(predictions_df, best_model).to_csv(
        OUTPUT_DIR / "selected_model_predictions.csv", index=False
    )


if __name__ == "__main__":
    main()
