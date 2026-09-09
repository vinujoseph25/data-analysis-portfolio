# Vinu Joseph — Data & Machine Learning Portfolio

A curated portfolio of practical **data analysis, statistical modelling and machine learning** work, with an emphasis on reproducibility, evaluation and explaining why a model is appropriate for a problem.

## Featured project

### Black Friday Purchase Prediction

A supervised regression study using the Kaggle Black Friday dataset to understand customer purchasing behaviour and predict `Purchase` values.

**Current progression:**

```text
Data inspection
     ↓
Feature preparation
     ↓
Multivariable Linear Regression
     ↓
Validation & diagnostics
     ↓
Ridge / Random Forest comparison
     ↓
Model selection
```

| Dimension | Approach |
| --- | --- |
| Problem | Supervised regression |
| Target | `Purchase` |
| Baseline | Multivariable Linear Regression |
| Extensions | Ridge Regression, Random Forest |
| Evaluation | R², MAE, MSE, RMSE, adjusted R² |
| Preprocessing | Imputation + categorical encoding in pipelines |
| Tools | Python, pandas, NumPy, scikit-learn, matplotlib, statsmodels |

The original baseline achieved **R² ≈ 0.642** on the documented 20% hold-out set. See the project README for the full methodology and validation results.

### Model comparison

The project now includes a reproducible comparison pipeline covering linear, regularised and non-linear regression. The purpose is to demonstrate model selection rather than simply adding algorithms: each model is evaluated on the same validation split and preprocessing is kept inside the modelling pipeline.

See [`MODEL_COMPARISON.md`](projects/black-friday-sales-prediction/MODEL_COMPARISON.md).

## Portfolio roadmap

The repository is being expanded around progressively stronger data-science capabilities:

1. **Exploratory data analysis** — distributions, missingness, relationships and segment analysis.
2. **Statistical modelling** — interpretable regression and diagnostics.
3. **Model comparison** — baseline, regularised and non-linear models.
4. **Feature engineering** — transforming raw business data into useful predictors.
5. **Model evaluation** — robust validation, error analysis and model selection.
6. **Explainability** — feature importance and business-facing interpretation.
7. **Applied ML projects** — classification, forecasting and customer analytics as the portfolio grows.

## Repository structure

```text
projects/
└── black-friday-sales-prediction/
    ├── README.md
    ├── MODEL_COMPARISON.md
    ├── data/README.md
    ├── notebooks/
    │   └── black_friday_sales_prediction.ipynb
    ├── src/
    │   ├── README.md
    │   ├── train_model.py
    │   └── model_comparison.py
    ├── outputs/              # Generated locally; ignored by Git
    └── requirements.txt
```

## Reproducibility

Raw Kaggle CSV files are intentionally not committed. Place the original `train.csv` and `test.csv` files under the project's `data/` directory and follow the project documentation to reproduce the analysis.

Generated predictions and model-comparison outputs remain local so the repository stays lightweight and source-controlled code remains the focus.

## Engineering standards

The data portfolio follows the same engineering principles as my software portfolio:

- reproducible scripts and notebooks
- explicit preprocessing steps
- no target leakage through preprocessing
- fixed validation seeds where appropriate
- documented evaluation metrics
- separation of raw data, source code and generated outputs
- clear distinction between exploratory analysis and validated results

## Author

**Vinu Joseph**

Data Science · Machine Learning · Statistical Analysis · Software Engineering
