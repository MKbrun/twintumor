from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATASET_PATH = Path("/Users/phillipovera/Downloads/series/ml_longitudinal_dataset.csv")
TEST_SIZE = 0.2
RANDOM_STATE = 42

TARGET_COLUMN = "target_next_log_volume"
GROUP_COLUMN = "subject_id"

FEATURE_COLUMNS = [
    "trajectory_type",
    "target_fu_index",
    "baseline_volume",
    "fu1_volume",
    "fu2_volume",
    "fu3_volume",
    "fu4_volume",
    "num_known_followups",
    "last_observed_volume",
    "previous_observed_volume",
    "min_previous_volume",
    "max_previous_volume",
    "mean_previous_volume",
    "change_last_vs_baseline",
    "change_last_vs_previous",
]


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load the ML dataset and check that required columns exist."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN, GROUP_COLUMN, "target_next_volume"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in dataset: {missing}")

    return df


def split_by_subject(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset so each subject stays fully in train or test."""
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(splitter.split(df, groups=df[GROUP_COLUMN]))
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    return train_df, test_df


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing for numeric and categorical features."""
    numeric_features = [col for col in FEATURE_COLUMNS if col != "trajectory_type"]
    categorical_features = ["trajectory_type"]

    # Fill missing numeric values and scale them
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Fill missing categorical values and one-hot encode them
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def evaluate_predictions(
    model_name: str,
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
) -> dict[str, float]:
    """Evaluate predictions in original volume space."""
    # Convert log-transformed values back to normal tumour volumes
    y_true_volume = np.expm1(y_true_log)
    y_pred_volume = np.expm1(y_pred_log)

    # Avoid tiny negative values after inverse transform
    y_pred_volume = np.clip(y_pred_volume, a_min=0, a_max=None)

    mae = mean_absolute_error(y_true_volume, y_pred_volume)
    rmse = np.sqrt(mean_squared_error(y_true_volume, y_pred_volume))
    r2 = r2_score(y_true_volume, y_pred_volume)

    return {
        "model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def train_and_evaluate(
    model_name: str,
    regressor,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[Pipeline, dict[str, float], pd.DataFrame]:
    """Train one regressor, evaluate it, and return predictions."""
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    # Build one pipeline with preprocessing + model
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", regressor),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred_log = pipeline.predict(X_test)

    metrics = evaluate_predictions(
        model_name=model_name,
        y_true_log=y_test.to_numpy(),
        y_pred_log=y_pred_log,
    )

    # Save useful information for later inspection
    predictions_df = test_df[
        [GROUP_COLUMN, "trajectory_type", "target_fu_index", "target_next_volume"]
    ].copy()
    predictions_df["true_next_log_volume"] = y_test.to_numpy()
    predictions_df["pred_next_log_volume"] = y_pred_log
    predictions_df["pred_next_volume"] = np.clip(np.expm1(y_pred_log), a_min=0, a_max=None)
    predictions_df["absolute_error"] = np.abs(
        predictions_df["target_next_volume"] - predictions_df["pred_next_volume"]
    )
    predictions_df["model"] = model_name

    return pipeline, metrics, predictions_df


def main() -> None:
    """Train the regression models and save metrics and predictions."""
    df = load_dataset(DATASET_PATH)

    print("Loaded dataset:")
    print(df.shape)
    print()

    # Split by subject to avoid leakage between train and test
    train_df, test_df = split_by_subject(df)

    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Train subjects: {train_df[GROUP_COLUMN].nunique()}")
    print(f"Test subjects: {test_df[GROUP_COLUMN].nunique()}")
    print()

    # Start with one simple linear model and one stronger non-linear model
    linear_model = LinearRegression()
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    all_metrics = []
    all_predictions = []

    for model_name, regressor in [
        ("LinearRegression", linear_model),
        ("RandomForestRegressor", rf_model),
    ]:
        print(f"Training {model_name}...")

        _, metrics, predictions_df = train_and_evaluate(
            model_name=model_name,
            regressor=regressor,
            train_df=train_df,
            test_df=test_df,
        )

        all_metrics.append(metrics)
        all_predictions.append(predictions_df)

        print(f"{model_name} results:")
        print(f"  MAE:  {metrics['MAE']:.3f}")
        print(f"  RMSE: {metrics['RMSE']:.3f}")
        print(f"  R2:   {metrics['R2']:.3f}")
        print()

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    print("Summary of model performance:")
    print(metrics_df)
    print()

    print("Example predictions:")
    print(
        predictions_df[
            [
                "model",
                "subject_id",
                "trajectory_type",
                "target_fu_index",
                "target_next_volume",
                "pred_next_volume",
                "absolute_error",
            ]
        ].head(20)
    )

    # Save outputs so they can be reused in later analysis
    metrics_output = Path("regression_metrics.csv")
    preds_output = Path("regression_predictions.csv")

    metrics_df.to_csv(metrics_output, index=False)
    predictions_df.to_csv(preds_output, index=False)

    print()
    print(f"Saved metrics to: {metrics_output.resolve()}")
    print(f"Saved predictions to: {preds_output.resolve()}")


if __name__ == "__main__":
    main()