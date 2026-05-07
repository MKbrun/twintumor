"""Tests for the multi-model held-out evaluator."""

import pandas as pd

from src.data.loaders import load_demo_dataset
from src.models.multi_model_eval import evaluate_models


def test_summary_includes_all_models():
    df = load_demo_dataset()
    summary, per_subject = evaluate_models(df, test_size=0.2, random_state=42)

    expected_models = {
        "Exponential", "Gompertz",
        "Linear Regression",
        "Random Forest (depth=6)",
        "Gradient Boosting",
        "MLP (64,32)",
    }
    assert expected_models.issubset(set(summary["Model"]))


def test_summary_has_per_timepoint_rows():
    df = load_demo_dataset()
    summary, _ = evaluate_models(df, test_size=0.2, random_state=42)
    timepoints = set(summary["Timepoint"].unique())
    assert {"FU3", "FU4", "FU5", "FU3-FU5 mean"} == timepoints


def test_per_subject_has_predictions_for_every_model():
    df = load_demo_dataset()
    _, per_subject = evaluate_models(df, test_size=0.2, random_state=42)
    assert not per_subject.empty
    for prefix in ["Linear Regression", "Random Forest (depth=6)",
                   "Gradient Boosting", "MLP (64,32)", "Exponential", "Gompertz"]:
        for fu in ["FU3", "FU4", "FU5"]:
            assert f"{prefix}_{fu}" in per_subject.columns


def test_test_split_size_respected():
    df = load_demo_dataset()
    _, per_subject = evaluate_models(df, test_size=0.3, random_state=0)
    expected = int(round(len(df) * 2 * 0.3))   # 2 scenarios per patient
    assert abs(len(per_subject) - expected) <= 1


def test_empty_df_returns_empty():
    summary, per_subject = evaluate_models(pd.DataFrame())
    assert summary.empty and per_subject.empty


def test_test_split_is_stratified_by_scenario():
    """The held-out test set should keep the cohort's 50/50 progression /
    remission balance even on small splits."""
    df = load_demo_dataset()
    _, per_subject = evaluate_models(df, test_size=0.2, random_state=0)
    counts = per_subject["scenario"].value_counts()
    # Either exactly equal or off by at most 1 (depending on rounding)
    assert abs(counts.get("progression", 0) - counts.get("remission", 0)) <= 1
