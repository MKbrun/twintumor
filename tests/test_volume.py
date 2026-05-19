"""Unit tests for tumor volume calculation."""

import numpy as np
import pytest

from src.metrics.volume import compute_tumor_volume


def test_simple_voxel_count():
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[:2, :2, :2] = 1   # 8 tumor voxels
    out = compute_tumor_volume(mask, voxel_dims=(1.0, 1.0, 1.0))
    assert out["voxel_count"] == 8
    assert out["volume_mm3"] == pytest.approx(8.0)
    assert out["volume_cm3"] == pytest.approx(0.008)


def test_anisotropic_voxels():
    mask = np.ones((2, 2, 2), dtype=np.uint8)  # 8 voxels
    out = compute_tumor_volume(mask, voxel_dims=(1.0, 2.0, 3.0))
    assert out["voxel_volume_mm3"] == pytest.approx(6.0)
    assert out["volume_mm3"] == pytest.approx(48.0)


def test_threshold_excludes_low_intensities():
    mask = np.array([[[0.4, 0.6]]] , dtype=float)  # one below, one above
    out = compute_tumor_volume(mask, voxel_dims=(1.0, 1.0, 1.0), threshold=0.5)
    assert out["voxel_count"] == 1


def test_rejects_non_3d_input():
    with pytest.raises(ValueError):
        compute_tumor_volume(np.zeros((4, 4)), voxel_dims=(1.0, 1.0, 1.0))


def test_rejects_nonpositive_voxel_dims():
    with pytest.raises(ValueError):
        compute_tumor_volume(np.zeros((2, 2, 2), dtype=np.uint8), voxel_dims=(1.0, 0.0, 1.0))
