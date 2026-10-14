"""End-to-end regression pipeline for the California Housing dataset.

Run:
    python house_price_predictor.py

Workflow:
    1. Explore the dataset.
    2. Engineer features.
    3. Compare regression models using cross-validation.
    4. Tune the best model.
    5. Evaluate on a held-out test set.
    6. Demonstrate predictions on new observations.

Plots and numerical results are saved to outputs/.
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


# Configuration

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5
N_SEARCH_ITER = 15
TARGET = "MedHouseVal"
TARGET_UNIT_USD = 100_000
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

ENGINEERED_FEATURES = [
    "bedrooms_per_room",
    "rooms_per_person",
]

ALL_FEATURES = FEATURES + ENGINEERED_FEATURES


# Step 1: Load and explore the data

def load_data():
    """Load California Housing; the first run may download the dataset."""
    return fetch_california_housing(as_frame=True).frame


def explore_data(df):
    """Display dataset statistics and return a summary for the results file."""
    target = df[TARGET]

    correlations = (
        df.corr(numeric_only=True)[TARGET]
        .drop(TARGET)
        .sort_values(ascending=False)
    )

    print("\n" + "=" * 65)
    print("STEP 1: DATASET EXPLORATION")
    print("=" * 65)

    print(f"\nDataset shape: {df.shape}")
    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nDescriptive statistics:")
    print(df.describe().round(3).to_string())

    print("\nMissing values:")
    print(df.isna().sum().to_string())

    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    print("\nFeature correlations with the target:")
    print(correlations.round(3).to_string())

    print(f"\nTarget mean: {target.mean():.3f}")
    print(f"Target median: {target.median():.3f}")
    print(f"Target maximum: {target.max():.3f}")

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_mean": float(target.mean()),
        "target_median": float(target.median()),
        "target_max": float(target.max()),
        "correlations_with_target": correlations.round(4).to_dict(),
    }


def save_plot(fig, filename):
    """Save a plot and release its memory."""
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=120, bbox_inches="tight")
    plt.close(fig)


def create_eda_plots(df):
    """Create three visualisations for exploratory data analysis."""

    # Target distribution
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(df[TARGET], bins=50, ax=ax)

    ax.set_title("Distribution of Median House Values")
    ax.set_xlabel("Median house value ($100,000s)")
    ax.set_ylabel("Number of block groups")

    save_plot(fig, "target_distribution.png")

    # Income versus house value
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        df["MedInc"],
        df[TARGET],
        s=10,
        alpha=0.2,
    )

    ax.set_title("Median Income vs Median House Value")
    ax.set_xlabel("Median household income (dataset units)")
    ax.set_ylabel("Median house value ($100,000s)")

    save_plot(fig, "income_vs_house_value.png")

    # Geographic distribution
    fig, ax = plt.subplots(figsize=(8, 6))

    points = ax.scatter(
        df["Longitude"],
        df["Latitude"],
        c=df[TARGET],
        cmap="viridis",
        s=6,
        alpha=0.6,
        edgecolors="none",
    )

    fig.colorbar(
        points,
        ax=ax,
        label="Median house value ($100,000s)",
    )

    ax.set_title("Median House Values by Geographic Location")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim")

    save_plot(fig, "geographic_house_values.png")


# Step 2: Feature engineering

def add_engineered_features(X):
    """Add two ratios while handling zero denominators safely."""
    X = pd.DataFrame(X, columns=FEATURES).copy()

    X["bedrooms_per_room"] = np.divide(
        X["AveBedrms"].to_numpy(dtype=float),
        X["AveRooms"].to_numpy(dtype=float),
        out=np.full(len(X), np.nan),
        where=X["AveRooms"].to_numpy(dtype=float) > 0,
    )

    X["rooms_per_person"] = np.divide(
        X["AveRooms"].to_numpy(dtype=float),
        X["AveOccup"].to_numpy(dtype=float),
        out=np.full(len(X), np.nan),
        where=X["AveOccup"].to_numpy(dtype=float) > 0,
    )

    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    return X


def feature_engineering_step():
    """Make feature engineering part of the model pipeline."""
    return FunctionTransformer(
        add_engineered_features,
        feature_names_out=lambda transformer, names: ALL_FEATURES,
    )


# Step 3: Build and compare regression models

def build_pipeline(estimator, scale=False):
    """Combine feature engineering, imputation and model training."""
    steps = [
        ("features", feature_engineering_step()),
        ("impute", SimpleImputer(strategy="median")),
    ]

    if scale:
        steps.append(("scale", StandardScaler()))

    steps.append(("model", estimator))

    return Pipeline(steps)


def candidate_models():
    """Return the baseline and three regression models."""
    return {
        "Dummy (mean)": build_pipeline(
            DummyRegressor(strategy="mean")
        ),
        "Ridge": build_pipeline(
            Ridge(),
            scale=True,
        ),
        "Random Forest": build_pipeline(
            RandomForestRegressor(
                n_estimators=150,
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
        ),
        "Gradient Boosting": build_pipeline(
            GradientBoostingRegressor(
                n_estimators=150,
                random_state=RANDOM_STATE,
            )
        ),
    }


def cross_validated_rmse(model, X, y, cv):
    """Calculate RMSE for each cross-validation fold."""
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    return -scores


def compare_models(X_train, y_train, cv):
    """Compare models using the same cross-validation folds."""
    results = {}

    for name, model in candidate_models().items():
        start = time.perf_counter()

        scores = cross_validated_rmse(
            model,
            X_train,
            y_train,
            cv,
        )

        results[name] = {
            "cv_rmse_mean": float(scores.mean()),
            "cv_rmse_std": float(scores.std()),
            "runtime_seconds": round(
                time.perf_counter() - start, 2
            ),
        }

    comparison = pd.DataFrame(results).T
    comparison = comparison.sort_values("cv_rmse_mean")

    print("\n" + "=" * 65)
    print("STEP 3: MODEL COMPARISON")
    print("=" * 65)
    print(comparison.round(4).to_string())
    print("\nLower RMSE indicates better predictive performance.")

    return comparison


def plot_model_comparison(comparison):
    """Plot cross-validation RMSE for the candidate models."""
    ordered = comparison.sort_values(
        "cv_rmse_mean",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.barh(
        ordered.index,
        ordered["cv_rmse_mean"],
        xerr=ordered["cv_rmse_std"],
        color="#3478b8",
        error_kw={"capsize": 3},
    )

    ax.set_title("Five-Fold Cross-Validation RMSE")
    ax.set_xlabel("RMSE ($100,000s)")

    save_plot(fig, "model_comparison.png")


# Step 4: Hyperparameter tuning

def search_space(model_name):
    """Define the hyperparameters to test for each model."""
    if model_name == "Random Forest":
        return {
            "model__n_estimators": [100, 150, 250, 350],
            "model__max_depth": [None, 10, 20, 30],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": [0.5, 0.75, 1.0],
        }

    if model_name == "Gradient Boosting":
        return {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__min_samples_leaf": [1, 5, 10],
        }

    if model_name == "Ridge":
        return {
            "model__alpha": np.logspace(-3, 3, 30),
        }

    raise ValueError(f"No search space defined for {model_name!r}")


def tune_model(model_name, X_train, y_train, cv):
    """Find the best hyperparameters using randomized search."""
    search = RandomizedSearchCV(
        estimator=candidate_models()[model_name],
        param_distributions=search_space(model_name),
        n_iter=N_SEARCH_ITER,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )

    start = time.perf_counter()
    search.fit(X_train, y_train)
    runtime = time.perf_counter() - start

    details = {
        "model": model_name,
        "best_params": search.best_params_,
        "best_cv_rmse": float(-search.best_score_),
        "configurations_tested": len(search.cv_results_["params"]),
        "runtime_seconds": round(runtime, 2),
    }

    print("\n" + "=" * 65)
    print("STEP 4: HYPERPARAMETER TUNING")
    print("=" * 65)
    print(f"Selected model: {model_name}")
    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV RMSE: {-search.best_score_:.4f}")
    print(f"Configurations tested: {len(search.cv_results_['params'])}")
    print(f"Runtime: {runtime:.1f} seconds")

    return search.best_estimator_, details


# Step 5: Final evaluation

def evaluate_model(model, baseline, X_test, y_test):
    """Evaluate the tuned model on data held out from training."""
    predictions = model.predict(X_test)
    baseline_predictions = baseline.predict(X_test)

    metrics = {
        "test_rmse": float(
            np.sqrt(mean_squared_error(y_test, predictions))
        ),
        "test_mae": float(
            mean_absolute_error(y_test, predictions)
        ),
        "test_r2": float(
            r2_score(y_test, predictions)
        ),
        "baseline_test_rmse": float(
            np.sqrt(
                mean_squared_error(y_test, baseline_predictions)
            )
        ),
    }

    metrics["rmse_improvement_vs_baseline"] = float(
        metrics["baseline_test_rmse"] - metrics["test_rmse"]
    )

    print("\n" + "=" * 65)
    print("STEP 5: FINAL TEST EVALUATION")
    print("=" * 65)

    print(
        f"Test RMSE: {metrics['test_rmse']:.4f} "
        f"(approximately ${metrics['test_rmse'] * TARGET_UNIT_USD:,.0f})"
    )

    print(
        f"Test MAE:  {metrics['test_mae']:.4f} "
        f"(approximately ${metrics['test_mae'] * TARGET_UNIT_USD:,.0f})"
    )

    print(f"Test R²:   {metrics['test_r2']:.4f}")

    print(
        f"Baseline RMSE: {metrics['baseline_test_rmse']:.4f}"
    )

    print(
        "RMSE improvement over baseline: "
        f"{metrics['rmse_improvement_vs_baseline']:.4f}"
    )

    return predictions, metrics


def plot_residuals(y_test, predictions):
    """Plot prediction errors to help identify systematic patterns."""
    residuals = np.asarray(y_test) - predictions

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].scatter(
        predictions,
        residuals,
        s=10,
        alpha=0.3,
    )
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_title("Residuals vs Predicted Values")
    axes[0].set_xlabel("Predicted value ($100,000s)")
    axes[0].set_ylabel("Actual - predicted ($100,000s)")

    sns.histplot(residuals, bins=40, ax=axes[1])
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_title("Distribution of Residuals")
    axes[1].set_xlabel("Residual ($100,000s)")

    save_plot(fig, "residual_analysis.png")


def plot_feature_importance(model):
    """Plot tree feature importances or absolute Ridge coefficients."""
    estimator = model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        label = "Feature importance"
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_)
        label = "Absolute coefficient (standardised features)"
    else:
        print("No feature importance measure is available.")
        return None

    importance = pd.Series(
        values,
        index=ALL_FEATURES,
    ).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))

    importance.plot(
        kind="barh",
        ax=ax,
        color="#3478b8",
    )

    ax.set_title("Feature Importance")
    ax.set_xlabel(label)

    save_plot(fig, "feature_importance.png")

    return importance.sort_values(
        ascending=False
    ).round(4).to_dict()


# Step 6: Prediction interface

def validate_inputs(features):
    """Check that all required prediction inputs are valid."""
    missing = [
        name for name in FEATURES
        if name not in features
    ]

    unexpected = sorted(
        set(features) - set(FEATURES)
    )

    if missing:
        raise ValueError(
            f"Missing required inputs: {', '.join(missing)}"
        )

    if unexpected:
        raise ValueError(
            f"Unexpected inputs: {', '.join(unexpected)}"
        )

    clean = {}

    for name in FEATURES:
        if isinstance(features[name], (bool, np.bool_)):
            raise ValueError(f"{name} must be numeric, not boolean.")

        try:
            value = float(features[name])
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be numeric.") from None

        if not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")

        clean[name] = value

    for name in ("MedInc", "AveRooms", "AveOccup"):
        if clean[name] <= 0:
            raise ValueError(f"{name} must be greater than zero.")

    for name in ("HouseAge", "AveBedrms", "Population"):
        if clean[name] < 0:
            raise ValueError(f"{name} cannot be negative.")

    if not 32.0 <= clean["Latitude"] <= 42.5:
        raise ValueError("Latitude must be within the approximate range of California.")

    if not -124.5 <= clean["Longitude"] <= -114.0:
        raise ValueError("Longitude must be within the approximate range of California.")

    return clean


def predict_house_value(model, **features):
    """Estimate a block group's median house value, not an individual home's price."""
    clean = validate_inputs(features)

    row = pd.DataFrame(
        [clean],
        columns=FEATURES,
    )

    prediction = float(model.predict(row)[0])

    return {
        "prediction": prediction,
        "prediction_usd": prediction * TARGET_UNIT_USD,
    }


def demonstrate_prediction(model, X_test, y_test):
    """Demonstrate prediction using one held-out observation."""
    example = X_test.iloc[0].to_dict()

    result = predict_house_value(
        model,
        **example,
    )

    actual = float(y_test.iloc[0])

    print("\n" + "=" * 65)
    print("STEP 6: PREDICTION DEMONSTRATION")
    print("=" * 65)

    print(
        f"Predicted value: {result['prediction']:.3f} "
        f"(approximately ${result['prediction_usd']:,.0f})"
    )

    print(
        f"Actual value:    {actual:.3f} "
        f"(approximately ${actual * TARGET_UNIT_USD:,.0f})"
    )

    print(
        "The prediction estimates the median value of a census "
        "block group's housing, not an individual property's market price."
    )

    return {
        "inputs": {
            name: float(value)
            for name, value in example.items()
        },
        "prediction": result["prediction"],
        "prediction_usd": result["prediction_usd"],
        "actual": actual,
    }


# Main workflow

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 120)

    df = load_data()
    overview = explore_data(df)
    create_eda_plots(df)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print("\nTrain/test split")
    print(f"Training rows: {len(X_train):,}")
    print(f"Held-out test rows: {len(X_test):,}")

    cv = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    comparison = compare_models(
        X_train,
        y_train,
        cv,
    )

    plot_model_comparison(comparison)

    best_name = (
        comparison
        .drop(index="Dummy (mean)")["cv_rmse_mean"]
        .idxmin()
    )

    print(f"\nBest candidate before tuning: {best_name}")

    tuned_model, tuning = tune_model(
        best_name,
        X_train,
        y_train,
        cv,
    )

    baseline = candidate_models()["Dummy (mean)"]
    baseline.fit(X_train, y_train)

    predictions, metrics = evaluate_model(
        tuned_model,
        baseline,
        X_test,
        y_test,
    )

    plot_residuals(
        y_test,
        predictions,
    )

    importances = plot_feature_importance(tuned_model)

    if importances:
        print("\nFeature importance, highest first:")
        print(pd.Series(importances).to_string())

    demo = demonstrate_prediction(
        tuned_model,
        X_test,
        y_test,
    )

    results = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": N_SPLITS,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "overview": overview,
        "model_comparison": comparison.round(4).to_dict(orient="index"),
        "selected_model": best_name,
        "tuning": tuning,
        "test_metrics": metrics,
        "feature_importance": importances,
        "prediction_demo": demo,
    }

    with (OUTPUT_DIR / "results.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            default=lambda value: (
                value.item()
                if isinstance(value, np.generic)
                else value
            ),
        )

    print(f"\nPlots and results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()