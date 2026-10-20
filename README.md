# California Housing Price Prediction

## Overview
An end-to-end simple machine learning project that predicts median house values in California using the California Housing dataset. The project covers exploratory data analysis, feature engineering, model comparison, hyperparameter tuning, and evaluation.

## Features
- **EDA:** Visualises house value distribution, income relationships, and geographic patterns.
- **Feature Engineering:** Creates `bedrooms_per_room` and `rooms_per_person`.
- **Model Comparison:** Compares Ridge Regression, Random Forest, and Gradient Boosting against a baseline using five-fold cross-validation.
- **Hyperparameter Tuning:** Uses `RandomizedSearchCV` to optimise the best candidate model.
- **Evaluation:** Measures performance using RMSE, MAE, and R² on held-out test data.
- **Prediction:** Provides a reusable function for predicting median house values.

## Technologies
- Python
- pandas and NumPy
- scikit-learn
- Matplotlib and Seaborn

## Getting Started

Requires Python 3.11 or newer. Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline:

```bash
python house_price_predictor.py
```

A full run can take 10 minutes or more, mostly during hyperparameter tuning. The dataset may be downloaded automatically on the first run. Generated visualisations and evaluation results are saved in the `outputs/` directory next to the script.

## Outputs

| File | Contents |
|---|---|
| `target_distribution.png` | Distribution of median house values |
| `income_vs_house_value.png` | Median income vs median house value |
| `geographic_house_values.png` | House values plotted by latitude and longitude |
| `model_comparison.png` | Cross-validation RMSE for each candidate model |
| `residual_analysis.png` | Residuals vs predictions and residual distribution |
| `feature_importance.png` | Feature importances (or Ridge coefficients) of the tuned model |
| `results.json` | Dataset summary, model comparison, tuning details, test metrics and a sample prediction |

## Results
The script compares candidate models using cross-validation, tunes the best candidate, and evaluates it on unseen test data.

| Model | 5-fold CV RMSE |
|---|---|
| Random Forest | 0.511 |
| Gradient Boosting | 0.512 |
| Ridge | 0.682 |
| Baseline (mean) | 1.156 |

Random Forest was selected and tuned. On the held-out test set:

| Metric | Tuned Random Forest | Baseline |
|---|---|---|
| RMSE | 0.503 (~$50,300) | 1.145 (~$114,500) |
| MAE | 0.329 (~$32,900) | – |
| R² | 0.807 | – |

Median income is by far the most important feature. See `outputs/results.json` for the full results.

## Limitations
The model predicts median housing values for California census block groups, not current market prices for individual properties.

## Future Improvements

- **Expand hyperparameter tuning.** Random Forest and Gradient Boosting achieved nearly identical cross-validation RMSE scores (0.511 and 0.512, respectively). Testing more configurations and tuning both models could identify further improvements. `HistGradientBoostingRegressor` is another candidate worth evaluating.
- **Improve results reporting.** Add percentage improvement over the baseline alongside the absolute RMSE reduction (in $100,000 units) in `results.json` to make model performance easier to interpret.