from __future__ import annotations

from pathlib import Path

import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATASET_PATH = Path("/Users/phillipovera/Downloads/series/ml_longitudinal_dataset.csv")
PREDICTIONS_PATH = Path("/Users/phillipovera/Downloads/series/regression_predictions.csv")

OUTPUT_MERGED_PATH = Path("response_class_predictions.csv")
OUTPUT_CONFUSION_PATH = Path("response_confusion_matrix.csv")

CLASS_ORDER = ["CR", "PR", "SD", "PD"]


def classify_response(volume: float, baseline_volume: float, min_previous_volume: float) -> str:
    """Assign CR, PR, SD, or PD based on the project response rules."""
    if pd.isna(volume) or pd.isna(baseline_volume) or pd.isna(min_previous_volume):
        return "UNKNOWN"

    # Complete remission means no remaining tumour volume
    if volume == 0:
        return "CR"

    # Partial remission means at least 50% reduction from baseline
    if baseline_volume > 0:
        relative_change_vs_baseline = (volume - baseline_volume) / baseline_volume
        if relative_change_vs_baseline <= -0.50:
            return "PR"

    # Progressive disease means at least 25% increase from the smallest previous volume
    if min_previous_volume > 0:
        relative_change_vs_min_prev = (volume - min_previous_volume) / min_previous_volume
        if relative_change_vs_min_prev >= 0.25:
            return "PD"

    # Everything else is stable disease
    return "SD"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the ML dataset and the saved regression predictions."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Predictions file not found: {PREDICTIONS_PATH}")

    dataset_df = pd.read_csv(DATASET_PATH)
    preds_df = pd.read_csv(PREDICTIONS_PATH)

    return dataset_df, preds_df


def merge_data(dataset_df: pd.DataFrame, preds_df: pd.DataFrame) -> pd.DataFrame:
    """Merge regression predictions with baseline and history features needed for class assignment."""
    merge_columns = [
        "subject_id",
        "trajectory_type",
        "target_fu_index",
        "baseline_volume",
        "min_previous_volume",
        "target_next_volume",
    ]

    missing_dataset_cols = [col for col in merge_columns if col not in dataset_df.columns]
    if missing_dataset_cols:
        raise KeyError(f"Dataset is missing columns: {missing_dataset_cols}")

    required_pred_cols = [
        "model",
        "subject_id",
        "trajectory_type",
        "target_fu_index",
        "target_next_volume",
        "pred_next_volume",
    ]
    missing_pred_cols = [col for col in required_pred_cols if col not in preds_df.columns]
    if missing_pred_cols:
        raise KeyError(f"Predictions file is missing columns: {missing_pred_cols}")

    base_df = dataset_df[merge_columns].copy()

    # Match each prediction row with the baseline and previous-volume information
    merged = preds_df.merge(
        base_df,
        on=["subject_id", "trajectory_type", "target_fu_index", "target_next_volume"],
        how="left",
        validate="many_to_one",
    )

    if merged["baseline_volume"].isna().any() or merged["min_previous_volume"].isna().any():
        raise ValueError("Some prediction rows could not be matched back to dataset history.")

    return merged


def add_response_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Add true and predicted response classes to the merged DataFrame."""
    df = df.copy()

    # Compute the true class from the real next volume
    df["true_response_class"] = df.apply(
        lambda row: classify_response(
            volume=row["target_next_volume"],
            baseline_volume=row["baseline_volume"],
            min_previous_volume=row["min_previous_volume"],
        ),
        axis=1,
    )

    # Compute the predicted class from the model-predicted next volume
    df["pred_response_class"] = df.apply(
        lambda row: classify_response(
            volume=row["pred_next_volume"],
            baseline_volume=row["baseline_volume"],
            min_previous_volume=row["min_previous_volume"],
        ),
        axis=1,
    )

    return df


def evaluate_model(df: pd.DataFrame, model_name: str) -> None:
    """Print accuracy, confusion matrix, and classification report for one model."""
    model_df = df[df["model"] == model_name].copy()

    y_true = model_df["true_response_class"]
    y_pred = model_df["pred_response_class"]

    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    report = classification_report(y_true, y_pred, labels=CLASS_ORDER, zero_division=0)

    print(f"\n===== {model_name} =====")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER))

    print("\nClassification report:")
    print(report)


def save_outputs(df: pd.DataFrame) -> None:
    """Save merged class predictions and the Random Forest confusion matrix."""
    df.to_csv(OUTPUT_MERGED_PATH, index=False)

    # Save one confusion matrix file for the main model
    rf_df = df[df["model"] == "RandomForestRegressor"].copy()
    if not rf_df.empty:
        cm = confusion_matrix(
            rf_df["true_response_class"],
            rf_df["pred_response_class"],
            labels=CLASS_ORDER,
        )
        cm_df = pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER)
        cm_df.to_csv(OUTPUT_CONFUSION_PATH)

    print(f"\nSaved merged class predictions to: {OUTPUT_MERGED_PATH.resolve()}")
    print(f"Saved Random Forest confusion matrix to: {OUTPUT_CONFUSION_PATH.resolve()}")


def main() -> None:
    """Evaluate response classes derived from predicted next tumour volumes."""
    dataset_df, preds_df = load_inputs()
    merged_df = merge_data(dataset_df, preds_df)
    merged_df = add_response_classes(merged_df)

    print("Merged data shape:", merged_df.shape)

    print("\nClass distribution (true):")
    print(merged_df["true_response_class"].value_counts())

    print("\nClass distribution by model:")
    for model_name in merged_df["model"].unique():
        model_df = merged_df[merged_df["model"] == model_name]
        print(f"\n{model_name}:")
        print(model_df["pred_response_class"].value_counts())

    for model_name in merged_df["model"].unique():
        evaluate_model(merged_df, model_name)

    print("\nExample rows:")
    print(
        merged_df[
            [
                "model",
                "subject_id",
                "trajectory_type",
                "target_fu_index",
                "baseline_volume",
                "min_previous_volume",
                "target_next_volume",
                "pred_next_volume",
                "true_response_class",
                "pred_response_class",
            ]
        ].head(20)
    )

    save_outputs(merged_df)


if __name__ == "__main__":
    main()