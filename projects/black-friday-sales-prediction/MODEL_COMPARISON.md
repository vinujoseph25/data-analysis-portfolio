# Model Comparison

The baseline project establishes a multivariable linear regression model. This extension evaluates whether more flexible regression approaches improve predictive performance on the same hold-out split.

## Models

| Model | Why it is included |
| --- | --- |
| Dummy Regressor | Establishes a naive mean-prediction benchmark so model lift can be quantified |
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
├── model_predictions.csv
├── selected_model_predictions.csv
└── model_run_metadata.json
```

`model_comparison.csv` contains R², MAE, MSE and RMSE for the validation set, plus improvement metrics against the Dummy Regressor and a `Selected` flag for the best predictive model. The selection rule excludes `DummyMean` and chooses the lowest validation RMSE, using MAE and model name as deterministic tie-breakers.

`model_predictions.csv` contains predictions from every fitted candidate model for the original test dataset. It also records the selected model name and the corresponding `SelectedPrediction` value on each row.

`selected_model_predictions.csv` is a compact downstream artifact containing only `User_ID`, `Product_ID`, and `SelectedPrediction`. It is generated through a dedicated helper so consumers do not need to know the internal name of the selected estimator.

`model_run_metadata.json` records the selected model, candidate model registry, baseline model, validation configuration, random seed and selected-model metrics. This provides a small machine-readable audit trail for the run.

The `Dummy Regressor` provides a naive mean-prediction benchmark, making it possible to distinguish genuine predictive value from simply reproducing the target's central tendency.

## Why this matters

The comparison moves the project beyond a single-model exercise toward a reproducible **model selection workflow**. The appropriate model should be selected from validation evidence, interpretability requirements, computational cost and business context rather than from algorithm complexity alone. Keeping a naive baseline is an important part of that evaluation: a complex model should demonstrate measurable improvement over a simple reference strategy.

The automated selection step also makes the workflow easier to extend. New candidate regressors can be added to the model registry without hard-coding a preferred algorithm; the validation results determine which predictive model is selected.

Separating the selected prediction artifact from the full comparison output also creates a cleaner contract for downstream analytics, APIs or future deployment code. Consumers can use the stable `SelectedPrediction` field without depending on which model won a future run.

## Quality checks

The project includes unit tests covering preprocessing, mixed-feature model fitting, unseen categorical values, baseline benchmarking, schema validation, model selection, run metadata and the selected prediction output contract. GitHub Actions runs the test suite across supported Python versions. The tests verify the selection logic using small deterministic fixtures, while the full dataset remains reserved for the end-to-end modelling workflow.

## Next analysis steps

- Add repeated cross-validation for more stable model estimates.
- Tune Ridge regularisation and Random Forest hyperparameters.
- Inspect residuals and error distribution by demographic/product segments.
- Add permutation or feature-importance analysis.
- Evaluate a gradient-boosting model as a further non-linear benchmark.
