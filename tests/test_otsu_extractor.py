"""Tests for the AIMI-layout Otsu signal extractor."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from src.pipelines.otsu_signal_extractor import (
    detect_root_layout,
    extract_patient_row,
    is_aimi_layout,
    otsu_threshold,
    signal_percent_in_roi,
)


# ----- Otsu math -----

def test_otsu_bimodal():
    rng = np.random.default_rng(42)
    low = rng.normal(0.2, 0.05, 1000)
    high = rng.normal(0.9, 0.05, 1000)
    th = otsu_threshold(np.concatenate([low, high]))
    assert 0.3 < th < 0.8   # threshold sits between the modes


def test_otsu_constant():
    th = otsu_threshold(np.full(100, 0.5))
    assert th == pytest.approx(0.5)


def test_otsu_empty():
    assert otsu_threshold(np.array([])) == 0.0


def test_signal_percent_basic():
    t1 = np.zeros((4, 4, 2))
    t1[:, :, 0] = 1.0  # bright slice
    t1[:, :, 1] = 0.0  # dark slice
    roi = np.ones_like(t1, dtype=bool)
    out = signal_percent_in_roi(t1, roi)
    assert out["n_roi"] == 32
    # Otsu separates the two modes; bright voxels (16) should pass
    assert out["n_above"] == 16
    assert out["percent"] == pytest.approx(50.0)


def test_signal_percent_shape_mismatch_raises():
    with pytest.raises(ValueError):
        signal_percent_in_roi(np.zeros((4, 4, 2)), np.zeros((4, 4, 3), dtype=bool))


# ----- AIMI layout detection -----

def _write(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), str(path))


@pytest.fixture
def aimi_patient(tmp_path) -> Path:
    p = tmp_path / "Mets_999"
    seg = np.zeros((6, 6, 6), dtype=np.uint8)
    seg[2:5, 2:5, 2:5] = 1
    _write(p / "baseline" / "seg.nii", seg)
    _write(p / "baseline" / "t1_gd.nii", np.where(seg, 0.9, 0.1))
    for scenario in ("progression", "remission"):
        for fu in range(1, 6):
            t1 = np.where(seg, 0.85, 0.1) + 0.0  # vary slightly per fu? not necessary
            _write(p / scenario / f"FU{fu}" / "t1_gd.nii", t1)
    return p


@pytest.fixture
def legacy_patient(tmp_path) -> Path:
    p = tmp_path / "Mets_legacy"
    seg = np.zeros((6, 6, 6), dtype=np.uint8); seg[2:5, 2:5, 2:5] = 1
    _write(p / "baseline" / "seg.nii", seg)
    _write(p / "baseline" / "t1_gd.nii", np.where(seg, 0.9, 0.1))
    for scenario in ("progression", "remission"):
        for fu in range(1, 6):
            _write(p / scenario / f"FU{fu}" / "seg.nii", seg)
            _write(p / scenario / f"FU{fu}" / "t1_gd.nii", np.where(seg, 0.85, 0.1))
    return p


def test_is_aimi_layout_true(aimi_patient):
    assert is_aimi_layout(aimi_patient) is True


def test_is_aimi_layout_false_for_legacy(legacy_patient):
    assert is_aimi_layout(legacy_patient) is False


def test_detect_root_layout_aimi(tmp_path, aimi_patient):
    # aimi_patient already lives under tmp_path
    assert detect_root_layout(tmp_path) == "aimi"


def test_detect_root_layout_legacy(tmp_path, legacy_patient):
    assert detect_root_layout(tmp_path) == "legacy"


def test_detect_root_layout_unknown(tmp_path):
    assert detect_root_layout(tmp_path / "does-not-exist") == "unknown"


def test_extract_patient_row_aimi(aimi_patient):
    row = extract_patient_row(aimi_patient)
    assert row is not None
    assert row["subject"] == "Mets_999"
    assert "baseline" in row
    for scen in ("progression", "remission"):
        for fu in range(1, 6):
            assert f"{scen}_FU{fu}" in row
    # Reasonable signal range
    assert 0 <= row["baseline"] <= 100


def test_extract_patient_row_modes_are_distinct(aimi_patient):
    """The two normalisation modes should be selectable and may differ."""
    pt = extract_patient_row(aimi_patient, mode="per_timepoint")
    ba = extract_patient_row(aimi_patient, mode="baseline_anchored")
    assert pt is not None and ba is not None
    # Baseline value reads the same in both modes
    assert pt["baseline"] == pytest.approx(ba["baseline"])
    # Both modes produce all required columns
    for scen in ("progression", "remission"):
        for fu in range(1, 6):
            col = f"{scen}_FU{fu}"
            assert col in pt and col in ba


def test_extract_patient_row_invalid_mode(aimi_patient):
    with pytest.raises(ValueError):
        extract_patient_row(aimi_patient, mode="not-a-mode")


def test_signal_percent_with_fixed_threshold_and_mean_matching():
    """Mean-matching should rescale FU intensities to baseline's mean before
    applying the fixed threshold."""
    roi = np.ones((4, 4, 2), dtype=bool)
    base = np.full((4, 4, 2), 1.0)        # baseline mean = 1.0
    fu = np.full((4, 4, 2), 2.0)          # FU mean = 2.0  → rescaled to 1.0
    th = 0.5                              # baseline threshold

    # Without normalisation, all FU voxels are above 0.5 → 100%
    out_unscaled = signal_percent_in_roi(fu, roi, fixed_threshold=th)
    assert out_unscaled["percent"] == 100.0

    # With mean-matching to baseline (target_mean=1.0), FU rescaled to 1.0;
    # 1.0 > 0.5, so still all above → 100%
    out_scaled = signal_percent_in_roi(fu, roi, fixed_threshold=th, target_mean=1.0)
    assert out_scaled["percent"] == 100.0

    # Now FU has mean 0.4 → rescaled to 1.0 each; all above 0.5 → 100%
    fu_low = np.full((4, 4, 2), 0.4)
    out_norm = signal_percent_in_roi(fu_low, roi, fixed_threshold=th, target_mean=1.0)
    assert out_norm["percent"] == 100.0
    # Without normalisation, all 0.4 < 0.5 → 0%
    out_raw = signal_percent_in_roi(fu_low, roi, fixed_threshold=th)
    assert out_raw["percent"] == 0.0
