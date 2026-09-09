# Model Comparison

The baseline project establishes a multivariable linear regression model. This extension evaluates whether more flexible regression approaches improve predictive performance on the same hold-out split.

## Models

| Model | Why it is included |
| --- | --- |
| Linear Regression | Interpretable baseline and reference point |
| Ridge Regression | Tests whether L2 regularisation improves stability with many encoded predictors |
| Random Forest Regressor | Tests a non-linear ensemble against the linear assumptions |

All models use the same 80/20 validation split with `random_state=42` and the same feature exclusions. Categorical variables are one-hot encoded and numeric values are imputed and scaled within the pipeline to prevent preprocessing leakage.

## Run the comparison

From `projects/black-friday-sales-prediction/`:

```bash
python src/model_comparison.py
```

The script produces:

```text
outputs/
├── model_comparison.csv
└── model_predictions.csv
```

`model_comparison.csv` contains R², MAE, MSE and RMSE for the validation set. `model_predictions.csv` contains predictions from each fitted model for the original test dataset.

## Why this matters

The comparison moves the project beyond a single-model exercise toward a reproducible **model selection workflow**. The appropriate model should be selected from validation evidence, interpretability requirements, computational cost and business context rather than from algorithm complexity alone.

## Next analysis steps

- Add repeated cross-validation for more stable model estimates.
- Tune Ridge regularisation and Random Forest hyperparameters.
- Inspect residuals and error distribution by demographic/product segments.
- Add permutation or feature-importance analysis.
- Evaluate a gradient-boosting model as a further non-linear benchmark.
