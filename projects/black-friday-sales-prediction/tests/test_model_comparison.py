"""Unit tests for the reusable model-comparison helpers."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_comparison import build_model_pipeline, build_preprocessor, metrics  # noqa: E402


def test_metrics_returns_expected_regression_values():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])

    result = metrics(y_true, y_pred)

    assert set(result) == {"R2", "MAE", "MSE", "RMSE"}
    assert result["MAE"] == np.mean([2.0, 2.0, 3.0])
    assert result["MSE"] == np.mean([4.0, 4.0, 9.0])
    assert result["RMSE"] == np.sqrt(result["MSE"])


def test_preprocessor_handles_numeric_and_categorical_missing_values():
    frame = pd.DataFrame(
        {
            "Age": ["18-25", None, "26-35"],
            "Occupation": [1, 2, 1],
            "Purchase": [1000, 2000, 1500],
        }
    )
    X = frame.drop(columns=["Purchase"])

    preprocessor = build_preprocessor(X)
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > 0
    assert np.isfinite(
        transformed.toarray() if hasattr(transformed, "toarray") else transformed
    ).all()


def test_model_pipeline_fits_and_predicts_with_mixed_features():
    frame = pd.DataFrame(
        {
            "Age": ["18-25", "26-35", "36-45", "26-35"],
            "Occupation": [1, 2, 1, 3],
            "Purchase": [1000, 2000, 1500, 2200],
        }
    )
    X = frame.drop(columns=["Purchase"])
    y = frame["Purchase"]

    pipeline = build_model_pipeline(X, Ridge(alpha=1.0))
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)

    assert predictions.shape == (len(X),)
    assert np.isfinite(predictions).all()


def test_model_pipeline_handles_unseen_categories_at_prediction_time():
    train = pd.DataFrame(
        {
            "Age": ["18-25", "26-35", "36-45"],
            "City_Category": ["A", "B", "A"],
        }
    )
    y = pd.Series([1000.0, 2000.0, 1500.0])
    new_data = pd.DataFrame(
        {
            "Age": ["46-50"],
            "City_Category": ["C"],
        }
    )

    pipeline = build_model_pipeline(train, Ridge(alpha=1.0))
    pipeline.fit(train, y)
    predictions = pipeline.predict(new_data)

    assert predictions.shape == (1,)
    assert np.isfinite(predictions).all()
