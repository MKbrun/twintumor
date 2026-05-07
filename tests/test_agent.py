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


# ----- Step 3: pseudoprogression (PDF flag-based, no time window) -----

def test_pseudoprogression_resolves_when_next_scan_drops():
    a = TreatmentAgent(initial_volume=100.0, enable_pseudoprogression=True)
    out1 = a.evaluate(140.0)   # first +40% growth -> Provisional flag
    out2 = a.evaluate(105.0)   # smaller than last -> retroactively Pseudoprogression
    assert out1["status"] == STATUS_PROVISIONAL_PROGRESSION
    assert a.final_statuses()[1] == STATUS_PSEUDOPROGRESSION


def test_pseudoprogression_confirms_when_next_scan_grows_more():
    a = TreatmentAgent(initial_volume=100.0, enable_pseudoprogression=True)
    a.evaluate(140.0)            # provisional
    a.evaluate(200.0)            # confirms
    final = a.final_statuses()
    assert final[1] == STATUS_PROGRESSION
    assert final[2] == STATUS_PROGRESSION


def test_pseudoprogression_can_fire_at_any_timepoint():
    """The PDF's Step 3 says 'first time' growth — not 'first FU'. The flag
    must be available at any point, not just the first scan after baseline."""
    a = TreatmentAgent(initial_volume=100.0, enable_pseudoprogression=True)
    a.evaluate(95.0)   # FU1 stable
    a.evaluate(90.0)   # FU2 stable, smallest=90
    out3 = a.evaluate(125.0)  # FU3 first growth event >= 25% vs smallest -> Provisional
    assert out3["status"] == STATUS_PROVISIONAL_PROGRESSION


def test_pseudoprogression_can_re_fire_after_resolution():
    """After a pseudoprogression resolves, a later first-time growth should
    flag again (each event is independent)."""
    a = TreatmentAgent(initial_volume=100.0, enable_pseudoprogression=True)
    a.evaluate(140.0)  # provisional
    a.evaluate(80.0)   # resolves -> previous = Pseudoprogression
    out3 = a.evaluate(120.0)  # +50% vs smallest=80 -> new flag
    assert out3["status"] == STATUS_PROVISIONAL_PROGRESSION


def test_pseudoprogression_off_by_default():
    a = TreatmentAgent(initial_volume=100.0)  # enable_pseudoprogression=False
    out = a.evaluate(140.0)
    assert out["status"] == STATUS_PROGRESSION  # immediate, no flag
