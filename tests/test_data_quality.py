"""Tests for flat-trajectory detection (data-quality helpers)."""

import pandas as pd

from src.data.loaders import flat_trajectory_summary, is_trajectory_flat


def test_is_trajectory_flat_detects_constant():
    assert is_trajectory_flat([100.0] * 6) is True
    assert is_trajectory_flat([789.0, 789.0, 789.0, 789.0, 789.0, 789.0]) is True


def test_is_trajectory_flat_rejects_varying():
    assert is_trajectory_flat([100.0, 100.0, 100.0, 100.0, 100.0, 101.0]) is False
    assert is_trajectory_flat([45.5, 37.6, 38.9, 39.5, 41.9, 43.7]) is False


def test_is_trajectory_flat_handles_floating_point_noise():
    base = 789.0
    eps = base * 1e-10  # well below default rel_tol
    assert is_trajectory_flat([base + i * eps for i in range(6)]) is True


def test_is_trajectory_flat_empty():
    assert is_trajectory_flat([]) is True


def test_flat_trajectory_summary_counts():
    rows = [
        # Mets_001: progression flat, remission varies
        {"subject": "Mets_001", "baseline": 100.0,
         "progression_FU1": 100, "progression_FU2": 100, "progression_FU3": 100,
         "progression_FU4": 100, "progression_FU5": 100,
         "remission_FU1": 95, "remission_FU2": 90, "remission_FU3": 85,
         "remission_FU4": 80, "remission_FU5": 75},
        # Mets_002: both vary
        {"subject": "Mets_002", "baseline": 200.0,
         "progression_FU1": 210, "progression_FU2": 220, "progression_FU3": 230,
         "progression_FU4": 240, "progression_FU5": 250,
         "remission_FU1": 195, "remission_FU2": 190, "remission_FU3": 185,
         "remission_FU4": 180, "remission_FU5": 175},
        # Mets_003: both flat
        {"subject": "Mets_003", "baseline": 300.0,
         "progression_FU1": 300, "progression_FU2": 300, "progression_FU3": 300,
         "progression_FU4": 300, "progression_FU5": 300,
         "remission_FU1": 300, "remission_FU2": 300, "remission_FU3": 300,
         "remission_FU4": 300, "remission_FU5": 300},
    ]
    df = pd.DataFrame(rows)
    summary = flat_trajectory_summary(df)
    assert summary["n_total"] == 3
    assert summary["flat_counts"]["progression"] == 2
    assert summary["flat_counts"]["remission"] == 1
    assert "Mets_001" in summary["flat_subjects"]["progression"]
    assert "Mets_003" in summary["flat_subjects"]["progression"]
    assert "Mets_003" in summary["flat_subjects"]["remission"]
    assert "Mets_001" not in summary["flat_subjects"]["remission"]
