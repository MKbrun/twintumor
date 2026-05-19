"""Tests for the 2D growth-heatmap helpers."""

import numpy as np
import pytest

from src.viz.growth_heatmap import (
    GrowthLayer,
    find_largest_tumor_slice,
    grow_mask_2d_to_area,
    predicted_growth_layers_2d,
    stacked_timeline_heatmap,
    synthetic_disk_mask,
)


def test_find_largest_tumor_slice_picks_correct_slice():
    mask = np.zeros((10, 10, 4), dtype=np.uint8)
    mask[2:5, 2:5, 0] = 1   # 9
    mask[1:8, 1:8, 1] = 1   # 49 ← largest
    mask[3:6, 3:6, 2] = 1   # 9
    assert find_largest_tumor_slice(mask) == 1


def test_find_largest_tumor_slice_empty_mask_raises():
    with pytest.raises(ValueError):
        find_largest_tumor_slice(np.zeros((4, 4, 4)))


def test_grow_mask_2d_to_area_grows_outward():
    m = synthetic_disk_mask(side=80, radius=10)   # ≈ π·100 ≈ 314 px
    grown = grow_mask_2d_to_area(m, target_area_px=int(m.sum() * 1.5))
    assert grown.sum() >= int(m.sum() * 1.5) - 5
    assert grown.sum() <= int(m.sum() * 1.5) + 5
    # Original disk is fully contained in the grown disk
    assert (grown[m]).all()


def test_grow_mask_2d_to_area_shrinks_inward():
    m = synthetic_disk_mask(side=80, radius=15)
    target = m.sum() // 2
    shrunk = grow_mask_2d_to_area(m, target_area_px=target)
    assert abs(int(shrunk.sum()) - target) <= 2
    # Shrunk mask is a subset of the original
    assert ((m | shrunk) == m).all()


def test_grow_mask_to_zero_returns_empty():
    m = synthetic_disk_mask(side=40, radius=8)
    out = grow_mask_2d_to_area(m, target_area_px=0)
    assert out.sum() == 0


def test_predicted_growth_layers_isotropic_scaling():
    seg = np.zeros((40, 40, 5), dtype=np.uint8)
    seg[10:20, 10:20, 2] = 1   # 100 px in slice 2

    slice_idx, current, layers = predicted_growth_layers_2d(
        seg_3d=seg, voxel_dims=(1.0, 1.0, 1.0),
        current_volume_mm3=1000.0,
        predicted_volumes_mm3=[1000.0, 8000.0, 27.0],   # ratio 1, 8, 0.027
        timepoint_labels=["FU3", "FU4", "FU5"],
    )
    assert slice_idx == 2
    # FU3: ratio 1 → no change
    assert layers[0].mask_2d.sum() == current.sum()
    # FU4: ratio 8 → area scale 8^(2/3) = 4 → 100 → 400
    assert abs(int(layers[1].mask_2d.sum()) - 400) <= 5
    # FU5: ratio 0.027 → area scale 0.027^(2/3) ≈ 0.09 → 100 → 9
    assert abs(int(layers[2].mask_2d.sum()) - 9) <= 3


def test_stacked_timeline_heatmap_encoding():
    seg = np.zeros((20, 20, 3), dtype=np.uint8)
    seg[5:10, 5:10, 1] = 1   # 25 px
    _, current, layers = predicted_growth_layers_2d(
        seg_3d=seg, voxel_dims=(1.0, 1.0, 1.0),
        current_volume_mm3=100.0,
        predicted_volumes_mm3=[100.0, 800.0, 800.0],
        timepoint_labels=["FU3", "FU4", "FU5"],
    )
    heatmap = stacked_timeline_heatmap(current, layers)

    # Pixels already tumor encoded as 0
    assert (heatmap[current] == 0).all()
    # Pixels never tumor in any layer encoded as NaN
    largest_layer = layers[-1].mask_2d
    assert np.isnan(heatmap[~largest_layer]).all()
    # Each non-NaN pixel encodes the earliest layer (>= 0)
    nonnan = ~np.isnan(heatmap)
    assert (heatmap[nonnan] >= 0).all()


def test_predicted_growth_layers_label_count_must_match():
    seg = np.zeros((10, 10, 3), dtype=np.uint8); seg[2:5, 2:5, 1] = 1
    with pytest.raises(ValueError):
        predicted_growth_layers_2d(
            seg, (1, 1, 1), 100.0,
            predicted_volumes_mm3=[100, 200],
            timepoint_labels=["FU3", "FU4", "FU5"],
        )
