"""Tests for the continually-trained Random Forest predictor."""

import pandas as pd
import pytest

from src.models.ml_predictor import (
    DEFAULT_INCREMENTAL_TREES,
    INITIAL_TREES,
    get_or_train,
    predict_future,
    predict_future_with_uncertainty,
    reset_model,
    train_fresh_from,
    train_incremental,
)
from src.data.paths import DEMO_DATASET_CSV


@pytest.fixture
def tmp_model_path(tmp_path):
    return tmp_path / "rf.joblib"


@pytest.fixture
def tmp_csv(tmp_path):
    """Tiny synthetic 4-patient cohort, baseline-normalisable trajectories."""
    rows = []
    for i in range(4):
        b = 100.0 + i * 50
        rows.append({
            "subject": f"P{i:03d}",
            "baseline": b,
            "progression_FU1": b * 1.05, "progression_FU2": b * 1.10,
            "progression_FU3": b * 1.15, "progression_FU4": b * 1.18, "progression_FU5": b * 1.20,
            "remission_FU1": b * 0.95, "remission_FU2": b * 0.90,
            "remission_FU3": b * 0.85, "remission_FU4": b * 0.82, "remission_FU5": b * 0.80,
        })
    p = tmp_path / "cohort.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_initial_train_falls_back_to_demo(tmp_model_path):
    model, log, was_trained = get_or_train(tmp_model_path, DEMO_DATASET_CSV)
    assert was_trained
    assert len(log) == 1
    assert log[0].trees_after == INITIAL_TREES
    assert "Bundled demo" in log[0].source_label


def test_second_call_loads_persisted(tmp_model_path):
    get_or_train(tmp_model_path, DEMO_DATASET_CSV)
    _, _, was_trained = get_or_train(tmp_model_path, DEMO_DATASET_CSV)
    assert was_trained is False


def test_incremental_grows_forest(tmp_model_path, tmp_csv):
    # 1) Initial train from demo
    model, log, _ = get_or_train(tmp_model_path, DEMO_DATASET_CSV)
    trees_before = len(model.estimators_)
    # 2) Add 50 trees from the small extra CSV
    model2, log2 = train_incremental(
        csv_path=tmp_csv,
        source_label="unit-test cohort",
        model_path=tmp_model_path,
        n_new_trees=50,
    )
    assert len(model2.estimators_) == trees_before + 50
    assert len(log2) == 2
    assert log2[1].trees_after - log2[1].trees_before == 50
    assert log2[1].source_label == "unit-test cohort"
    assert log2[1].n_samples == 8  # 4 patients × 2 scenarios


def test_incremental_creates_initial_if_missing(tmp_model_path, tmp_csv):
    """When no model exists, train_incremental cold-starts on the same
    csv_path it's adding from (no demo fallback) so each source's model
    is calibrated to that source from the very first round."""
    assert not tmp_model_path.exists()
    model, log = train_incremental(
        csv_path=tmp_csv,
        source_label="cold-start cohort",
        model_path=tmp_model_path,
        n_new_trees=30,
    )
    # First round = cold-start on csv_path; second = +30 trees from csv_path
    assert len(log) == 2
    assert "cold-start cohort" in log[0].source_label
    assert log[1].source_label == "cold-start cohort"
    assert len(model.estimators_) == log[0].trees_after + 30


def test_reset_model_deletes_file(tmp_model_path):
    get_or_train(tmp_model_path, DEMO_DATASET_CSV)
    assert tmp_model_path.exists()
    assert reset_model(tmp_model_path) is True
    assert not tmp_model_path.exists()
    assert reset_model(tmp_model_path) is False  # idempotent


def test_predict_future_returns_keys_in_baseline_units(tmp_model_path):
    model, _, _ = get_or_train(tmp_model_path, DEMO_DATASET_CSV)
    out = predict_future(model, baseline=1000.0, fu1=950.0, fu2=920.0)
    assert set(out) == {"FU3", "FU4", "FU5"}
    # Predictions should be within a sane envelope of the baseline
    for v in out.values():
        assert 0.0 < v < 5000.0


def test_train_fresh_replaces_existing_model(tmp_model_path, tmp_csv):
    """train_fresh_from should produce a single-round log on the new CSV,
    even if a model with previous training history was present."""
    # Pre-existing model with some history
    get_or_train(tmp_model_path, DEMO_DATASET_CSV)
    train_incremental(
        csv_path=tmp_csv, source_label="prior round",
        model_path=tmp_model_path, n_new_trees=30,
    )
    # Now wipe and train fresh
    model, log = train_fresh_from(
        csv_path=tmp_csv, source_label="fresh-from-cohort",
        model_path=tmp_model_path,
    )
    # Should have a single round, only those samples, only INITIAL_TREES trees
    assert len(log) == 1
    assert "fresh-from-cohort" in log[0].source_label
    assert log[0].trees_before == 0
    assert log[0].trees_after == INITIAL_TREES
    assert len(model.estimators_) == INITIAL_TREES
    assert log[0].n_samples == 8  # 4 patients × 2 scenarios


def test_predict_uncertainty_shapes(tmp_model_path):
    model, _, _ = get_or_train(tmp_model_path, DEMO_DATASET_CSV)
    mean, std = predict_future_with_uncertainty(model, 1000.0, 950.0, 920.0)
    assert mean.shape == (3,)
    assert std.shape == (3,)
    assert (std >= 0).all()
