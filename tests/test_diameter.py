"""Tests for SPD-style diameter metric (Step 1, RANO PDF)."""

import numpy as np
import pytest

from src.metrics.diameter import compute_spd


def test_empty_mask_returns_zeros():
    out = compute_spd(np.zeros((4, 4, 4), dtype=np.uint8), (1.0, 1.0, 1.0))
    assert out["spd_mm2"] == 0.0
    assert out["longest_diameter_mm"] == 0.0
    assert out["perpendicular_mm"] == 0.0
    assert out["slice_index"] == -1


def test_simple_square_isotropic():
    """4-pixel × 3-pixel rectangle in slice z=2 with 1mm isotropic voxels."""
    mask = np.zeros((10, 10, 5), dtype=np.uint8)
    mask[3:7, 4:7, 2] = 1   # 4 rows, 3 cols
    out = compute_spd(mask, (1.0, 1.0, 1.0))
    assert out["slice_index"] == 2
    assert out["longest_diameter_mm"] == pytest.approx(4.0)
    assert out["perpendicular_mm"] == pytest.approx(3.0)
    assert out["spd_mm2"] == pytest.approx(12.0)


def test_anisotropic_voxels_are_used():
    """Same shape but voxels are 2mm in x and 0.5mm in y → SPD scales accordingly."""
    mask = np.zeros((10, 10, 3), dtype=np.uint8)
    mask[2:6, 1:5, 1] = 1   # 4 rows × 4 cols
    out = compute_spd(mask, (2.0, 0.5, 1.0))
    # rows side: 4 × 2.0 = 8.0 mm; cols side: 4 × 0.5 = 2.0 mm
    assert out["longest_diameter_mm"] == pytest.approx(8.0)
    assert out["perpendicular_mm"] == pytest.approx(2.0)
    assert out["spd_mm2"] == pytest.approx(16.0)


def test_picks_slice_with_largest_area():
    mask = np.zeros((10, 10, 3), dtype=np.uint8)
    mask[0:2, 0:2, 0] = 1   # 4 pixels
    mask[0:5, 0:5, 1] = 1   # 25 pixels  ← largest
    mask[0:3, 0:3, 2] = 1   # 9 pixels
    out = compute_spd(mask, (1.0, 1.0, 1.0))
    assert out["slice_index"] == 1
    assert out["n_tumor_pixels_in_slice"] == 25


def test_rejects_non_3d_input():
    with pytest.raises(ValueError):
        compute_spd(np.zeros((4, 4)), (1.0, 1.0, 1.0))


def test_rejects_nonpositive_voxel_dims():
    with pytest.raises(ValueError):
        compute_spd(np.zeros((2, 2, 2), dtype=np.uint8), (1.0, 0.0, 1.0))
