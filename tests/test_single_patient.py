"""Tests for single-patient inference helpers (no real MRI required)."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from src.pipelines.single_patient import (
    ScanVolumes,
    discover_followups,
    discover_scenarios,
    forecast_for_patient,
    read_patient_volumes,
)


def _make_seg(path: Path, n_voxels: int) -> None:
    """Write a minimal 4×4×4 NIfTI mask with `n_voxels` foreground voxels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    arr.flat[:n_voxels] = 1
    img = nib.Nifti1Image(arr, affine=np.eye(4))
    nib.save(img, str(path))


@pytest.fixture
def fake_patient(tmp_path) -> Path:
    p = tmp_path / "Mets_999"
    _make_seg(p / "baseline" / "seg.nii", 10)
    for i, n in enumerate([12, 14, 17, 21, 26], start=1):
        _make_seg(p / "progression" / f"FU{i}" / "seg.nii", n)
    for i, n in enumerate([9, 8, 7, 6, 5], start=1):
        _make_seg(p / "remission" / f"FU{i}" / "seg.nii", n)
    return p


def test_discover_scenarios(fake_patient):
    assert discover_scenarios(fake_patient) == ["progression", "remission"]


def test_discover_followups_sorted(fake_patient):
    fu = discover_followups(fake_patient / "progression")
    assert [f.name for f in fu] == ["FU1", "FU2", "FU3", "FU4", "FU5"]


def test_read_patient_volumes_shape(fake_patient):
    scan = read_patient_volumes(fake_patient, "progression")
    assert scan.scenario == "progression"
    # 1mm^3 per voxel => volume == voxel count
    assert scan.baseline == pytest.approx(10.0)
    assert scan.timepoints == {"FU1": 12.0, "FU2": 14.0, "FU3": 17.0, "FU4": 21.0, "FU5": 26.0}
    assert scan.trajectory == [10.0, 12.0, 14.0, 17.0, 21.0, 26.0]


def test_read_patient_invalid_scenario(fake_patient):
    with pytest.raises(ValueError):
        read_patient_volumes(fake_patient, "junk")


def test_read_patient_missing_baseline(tmp_path):
    p = tmp_path / "Mets_xx"
    _make_seg(p / "progression" / "FU1" / "seg.nii", 5)
    with pytest.raises(FileNotFoundError):
        read_patient_volumes(p, "progression")


def test_forecast_for_patient_returns_expected_keys(fake_patient):
    from src.data.paths import DEFAULT_ML_MODEL_PATH, DEMO_DATASET_CSV
    from src.models.ml_predictor import get_or_train, reset_model

    reset_model(DEFAULT_ML_MODEL_PATH)
    model, _, _ = get_or_train(DEFAULT_ML_MODEL_PATH, DEMO_DATASET_CSV)

    scan = read_patient_volumes(fake_patient, "progression")
    out = forecast_for_patient(scan, model)
    assert set(out) == {"observed", "actual_future", "forecasts", "ml_mean", "ml_std"}
    assert len(out["observed"]) == 3
    assert out["actual_future"].shape == (3,)
    assert out["ml_mean"].shape == (3,)
    assert out["ml_std"].shape == (3,)
