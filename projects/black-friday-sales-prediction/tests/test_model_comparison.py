"""Unit tests for the reusable model-comparison helpers."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_comparison import (  # noqa: E402
    add_baseline_lift,
    build_model_pipeline,
    build_preprocessor,
    build_run_metadata,
    metrics,
    select_best_model,
    validate_input_schema,
)


def test_metrics_returns_expected_regression_values():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    result = metrics(y_true, y_pred)
    assert set(result) == {"R2", "MAE", "MSE", "RMSE"}
    assert result["MAE"] == np.mean([2.0, 2.0, 3.0])
    assert result["MSE"] == np.mean([4.0, 4.0, 9.0])
    assert result["RMSE"] == np.sqrt(result["MSE"])


def test_preprocessor_handles_numeric_and_categorical_missing_values():
    frame = pd.DataFrame({"Age": ["18-25", None, "26-35"], "Occupation": [1, 2, 1], "Purchase": [1000, 2000, 1500]})
    X = frame.drop(columns=["Purchase"])
    transformed = build_preprocessor(X).fit_transform(X)
    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > 0
    assert np.isfinite(transformed.toarray() if hasattr(transformed, "toarray") else transformed).all()


def test_model_pipeline_fits_and_predicts_with_mixed_features():
    frame = pd.DataFrame({"Age": ["18-25", "26-35", "36-45", "26-35"], "Occupation": [1, 2, 1, 3], "Purchase": [1000, 2000, 1500, 2200]})
    X, y = frame.drop(columns=["Purchase"]), frame["Purchase"]
    predictions = build_model_pipeline(X, Ridge(alpha=1.0)).fit(X, y).predict(X)
    assert predictions.shape == (len(X),)
    assert np.isfinite(predictions).all()


def test_model_pipeline_handles_unseen_categories_at_prediction_time():
    train = pd.DataFrame({"Age": ["18-25", "26-35", "36-45"], "City_Category": ["A", "B", "A"]})
    y = pd.Series([1000.0, 2000.0, 1500.0])
    new_data = pd.DataFrame({"Age": ["46-50"], "City_Category": ["C"]})
    predictions = build_model_pipeline(train, Ridge(alpha=1.0)).fit(train, y).predict(new_data)
    assert predictions.shape == (1,)
    assert np.isfinite(predictions).all()


def test_dummy_baseline_produces_constant_mean_prediction():
    train = pd.DataFrame({"Age": ["18-25", "26-35", "36-45"], "Occupation": [1, 2, 3]})
    y = pd.Series([1000.0, 2000.0, 3000.0])
    predictions = build_model_pipeline(train, DummyRegressor(strategy="mean")).fit(train, y).predict(train)
    assert np.allclose(predictions, y.mean())


def test_baseline_lift_calculates_expected_improvement():
    results = pd.DataFrame({"Model": ["DummyMean", "Ridge"], "R2": [0.0, 0.25], "RMSE": [100.0, 75.0]})
    enriched = add_baseline_lift(results)
    assert enriched.loc[enriched["Model"] == "Ridge", "RMSE_Improvement_vs_Dummy"].iloc[0] == 25.0
    assert enriched.loc[enriched["Model"] == "Ridge", "R2_Improvement_vs_Dummy"].iloc[0] == 0.25


def test_baseline_lift_requires_dummy_baseline():
    results = pd.DataFrame({"Model": ["Ridge"], "R2": [0.25], "RMSE": [75.0]})
    with pytest.raises(ValueError, match="DummyMean baseline"):
        add_baseline_lift(results)


def test_select_best_model_uses_validation_rmse_and_ignores_dummy():
    results = pd.DataFrame(
        {
            "Model": ["DummyMean", "Ridge", "RandomForest"],
            "R2": [0.0, 0.25, 0.30],
            "MAE": [100.0, 70.0, 65.0],
            "RMSE": [120.0, 80.0, 75.0],
        }
    )
    assert select_best_model(results) == "RandomForest"


def test_select_best_model_raises_when_only_baseline_exists():
    results = pd.DataFrame({"Model": ["DummyMean"], "R2": [0.0], "MAE": [100.0], "RMSE": [120.0]})
    with pytest.raises(ValueError, match="No predictive models"):
        select_best_model(results)


def test_build_run_metadata_contains_reproducibility_and_selected_metrics():
    results = pd.DataFrame(
        {"Model": ["Ridge"], "R2": [0.25], "MAE": [70.0], "MSE": [5625.0], "RMSE": [75.0]}
    )
    metadata = build_run_metadata(results, "Ridge")
    assert metadata["target"] == "Purchase"
    assert metadata["selected_model"] == "Ridge"
    assert metadata["random_state"] == 42
    assert metadata["validation_size"] == 0.20
    assert metadata["selected_metrics"]["RMSE"] == 75.0


def test_input_schema_accepts_required_columns():
    train = pd.DataFrame({"User_ID": [1], "Product_ID": ["P1"], "Purchase": [1000], "Age": ["18-25"]})
    validate_input_schema(train, train.drop(columns=["Purchase"]))


@pytest.mark.parametrize(
    "train_columns,test_columns,error_text",
    [(["User_ID", "Purchase"], ["User_ID", "Product_ID"], "Training"), (["User_ID", "Product_ID", "Purchase"], ["User_ID"], "Test")],
)
def test_input_schema_reports_missing_required_columns(train_columns, test_columns, error_text):
    train = pd.DataFrame({column: [1] for column in train_columns})
    test = pd.DataFrame({column: [1] for column in test_columns})
    with pytest.raises(ValueError, match=error_text):
        validate_input_schema(train, test)


def test_input_schema_rejects_empty_data():
    train = pd.DataFrame(columns=["User_ID", "Product_ID", "Purchase"])
    test = pd.DataFrame(columns=["User_ID", "Product_ID"])
    with pytest.raises(ValueError, match="at least one row"):
        validate_input_schema(train, test)


def test_input_schema_rejects_non_numeric_target():
    train = pd.DataFrame({"User_ID": [1], "Product_ID": ["P1"], "Purchase": ["not-a-number"], "Age": ["18-25"]})
    with pytest.raises(ValueError, match="must be numeric"):
        validate_input_schema(train, train.drop(columns=["Purchase"]))


def test_input_schema_rejects_missing_target_values():
    train = pd.DataFrame({"User_ID": [1, 2], "Product_ID": ["P1", "P2"], "Purchase": [1000.0, np.nan], "Age": ["18-25", "26-35"]})
    with pytest.raises(ValueError, match="must not contain missing values"):
        validate_input_schema(train, train.drop(columns=["Purchase"]))


def test_input_schema_rejects_mismatched_features():
    train = pd.DataFrame({"User_ID": [1], "Product_ID": ["P1"], "Purchase": [1000], "Age": ["18-25"], "Gender": ["F"]})
    test = pd.DataFrame({"User_ID": [2], "Product_ID": ["P2"], "Age": ["26-35"]})
    with pytest.raises(ValueError, match="features do not match"):
        validate_input_schema(train, test)


def test_input_schema_rejects_duplicate_columns():
    train = pd.DataFrame([[1, "P1", 1000, "18-25"]], columns=["User_ID", "Product_ID", "Purchase", "Age"])
    train.columns = ["User_ID", "Product_ID", "Purchase", "Age"]
    test = train.drop(columns=["Purchase"])
    train.insert(4, "Age_duplicate", train["Age"])
    train.columns = ["User_ID", "Product_ID", "Purchase", "Age", "Age"]
    with pytest.raises(ValueError, match="duplicate columns"):
        validate_input_schema(train, test)
