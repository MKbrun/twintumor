"""Unit tests for TreatmentAgent classification logic."""

import pytest

from src.agent.treatment_agent import (
    STATUS_COMPLETE_REMISSION,
    STATUS_PARTIAL_REMISSION,
    STATUS_PROGRESSION,
    STATUS_PROVISIONAL_PROGRESSION,
    STATUS_PSEUDOPROGRESSION,
    STATUS_STABLE,
    TreatmentAgent,
)


def test_complete_remission():
    a = TreatmentAgent(initial_volume=100.0)
    assert a.evaluate(0.0)["status"] == STATUS_COMPLETE_REMISSION


def test_partial_remission_at_50_percent_drop():
    a = TreatmentAgent(initial_volume=100.0)
    assert a.evaluate(50.0)["status"] == STATUS_PARTIAL_REMISSION


def test_stable_disease_small_change():
    a = TreatmentAgent(initial_volume=100.0)
    assert a.evaluate(110.0)["status"] == STATUS_STABLE  # +10% < 25%


def test_progression_25_percent_above_smallest():
    a = TreatmentAgent(initial_volume=100.0)
    a.evaluate(80.0)            # smallest -> 80
    out = a.evaluate(101.0)     # +26% vs smallest -> Progression
    assert out["status"] == STATUS_PROGRESSION


def test_baseline_must_be_positive():
    with pytest.raises(ValueError):
        TreatmentAgent(initial_volume=0.0)


def test_negative_volume_rejected():
    a = TreatmentAgent(initial_volume=100.0)
    with pytest.raises(ValueError):
        a.evaluate(-1.0)


# ----- Step 3: pseudoprogression -----

def test_pseudoprogression_resolves_when_next_scan_drops():
    a = TreatmentAgent(initial_volume=100.0, enable_pseudoprogression=True, grace_window=2)
    out1 = a.evaluate(140.0)   # +40% inside grace window -> Provisional
    out2 = a.evaluate(105.0)   # falls back -> retroactively Pseudoprogression
    assert out1["status"] == STATUS_PROVISIONAL_PROGRESSION
    final = a.final_statuses()
    assert final[1] == STATUS_PSEUDOPROGRESSION
    assert out2["status"] in (STATUS_STABLE, STATUS_PARTIAL_REMISSION)


def test_pseudoprogression_confirms_when_next_scan_grows_more():
    a = TreatmentAgent(initial_volume=100.0, enable_pseudoprogression=True, grace_window=2)
    a.evaluate(140.0)            # provisional
    a.evaluate(200.0)            # confirms
    final = a.final_statuses()
    assert final[1] == STATUS_PROGRESSION
    assert final[2] == STATUS_PROGRESSION


def test_progression_outside_grace_window_is_immediate():
    a = TreatmentAgent(initial_volume=100.0, enable_pseudoprogression=True, grace_window=1)
    a.evaluate(80.0)             # FU1 within grace, no progression
    out2 = a.evaluate(120.0)     # FU2 outside grace, +50% vs smallest -> direct PD
    assert out2["status"] == STATUS_PROGRESSION
