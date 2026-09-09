# Vinu Joseph — Data & Machine Learning Portfolio

> Data Science · Machine Learning · Statistical Analysis · Software Engineering

A curated portfolio of practical **data analysis, statistical modelling and machine learning** work, with an emphasis on reproducibility, rigorous evaluation and communicating results clearly.

## Featured project

### Black Friday Purchase Prediction

A supervised regression study using the Kaggle Black Friday dataset to analyse customer purchasing behaviour and predict `Purchase` values.

The project starts with an interpretable **multivariable linear regression baseline** and extends the analysis through regularised and non-linear models to demonstrate model selection rather than simply adding algorithms.

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
| Evaluation | R², adjusted R², MAE, MSE, RMSE |
| Preprocessing | Imputation + categorical encoding in pipelines |
| Tooling | Python, pandas, NumPy, scikit-learn, matplotlib, statsmodels |

The documented baseline achieved **R² ≈ 0.642** on the project's 20% hold-out evaluation set. Detailed methodology and comparison results are maintained with the project.

See [`MODEL_COMPARISON.md`](projects/black-friday-sales-prediction/MODEL_COMPARISON.md).

## What this portfolio demonstrates

- Exploratory data analysis and data quality assessment
- Feature preparation and reproducible preprocessing
- Interpretable statistical modelling
- Multivariable regression
- Regularisation and non-linear model comparison
- Appropriate model evaluation and error analysis
- Separation of training, validation and generated outputs
- Reproducible Python workflows
- Translating model results into understandable conclusions

## Portfolio roadmap

The portfolio is being expanded toward progressively stronger applied-data capabilities:

1. **Exploratory analysis** — distributions, missingness, relationships and segmentation.
2. **Statistical modelling** — interpretable models and diagnostics.
3. **Model comparison** — baseline, regularised and non-linear approaches.
4. **Feature engineering** — transforming business data into useful predictors.
5. **Model evaluation** — validation, error analysis and model selection.
6. **Explainability** — feature importance and business-facing interpretation.
7. **Applied ML** — classification, forecasting and customer analytics.

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

Raw Kaggle CSV files are intentionally not committed. Place the original `train.csv` and `test.csv` files in the documented project data location and follow the project instructions to reproduce the analysis.

Generated predictions and model-comparison outputs remain local so that source-controlled code, methodology and documentation remain the focus.

## Engineering standards

The data portfolio follows the same engineering principles as my software work:

- reproducible scripts and notebooks
- explicit preprocessing steps
- prevention of target leakage
- consistent validation methodology
- documented metrics and assumptions
- separation of raw data, source code and generated outputs
- clear distinction between exploration and validated results

## Portfolio context

This repository represents the **Data & Machine Learning** pillar of my broader engineering portfolio. It complements the Enterprise React Platform and Software Engineering Portfolio by demonstrating the ability to work across software systems, data analysis and applied machine learning.

## Author

**Vinu Joseph**

Data Science · Machine Learning · Statistical Analysis · Software Engineering
