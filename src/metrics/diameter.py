"""
Step-1 diameter metrics from a 3D segmentation mask.

The PDF specifies SPD (Sum of Products of perpendicular Diameters) as a
RANO feature alongside volume. This module computes a tractable
approximation:

    1. For each axial slice, find the tumor pixels.
    2. Pick the slice with the largest tumor area.
    3. In that slice compute a longest-diameter / perpendicular pair via
       the bounding box of the tumor's pixels (in physical mm using the
       voxel spacing).
    4. SPD = longest_mm * perpendicular_mm.

This is an approximation — a full RANO SPD uses the longest in-plane
diameter (Feret-style) and a perpendicular through that diameter. The
bounding-box version is a reasonable, dependency-light proxy that the
report can clearly document as such.

Step 2 in `treatment_agent.py` follows the PDF *pseudocode* (volume-based)
rather than the PDF *table* (SPD-based), because seg.nii is a binary mask
and SPD adds approximation error. SPD here is exposed for completeness and
for future Step-2 variants.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def compute_spd(
    mask_data: np.ndarray,
    voxel_dims: Tuple[float, float, float],
    threshold: float = 0.0,
) -> Dict[str, float]:
    """
    Compute an SPD (Sum of Products of perpendicular Diameters) approximation
    plus the longest in-plane diameter, in physical millimetres.

    Parameters
    ----------
    mask_data : 3D numpy array (axial slices stacked along the last axis)
    voxel_dims: (dx, dy, dz) in mm
    threshold : voxel value above which a voxel is considered tumor

    Returns
    -------
    dict with keys:
        slice_index            : the axial slice with the largest tumor area
        longest_diameter_mm    : longer side of that slice's tumor bounding box
        perpendicular_mm       : shorter side of that slice's tumor bounding box
        spd_mm2                : longest * perpendicular
        n_tumor_pixels_in_slice
    """
    if mask_data.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {mask_data.shape}")
    if len(voxel_dims) != 3 or any(d <= 0 for d in voxel_dims):
        raise ValueError(f"Voxel dims must be 3 positive values, got {voxel_dims}")

    dx, dy, dz = voxel_dims
    binary = mask_data > threshold

    if not binary.any():
        return {
            "slice_index": -1,
            "longest_diameter_mm": 0.0,
            "perpendicular_mm": 0.0,
            "spd_mm2": 0.0,
            "n_tumor_pixels_in_slice": 0,
        }

    # Pick the axial slice with the biggest tumor area.
    areas = binary.sum(axis=(0, 1))  # one number per slice along last axis
    z = int(np.argmax(areas))
    slice_mask = binary[:, :, z]
    if not slice_mask.any():
        return {
            "slice_index": z,
            "longest_diameter_mm": 0.0,
            "perpendicular_mm": 0.0,
            "spd_mm2": 0.0,
            "n_tumor_pixels_in_slice": 0,
        }

    rows = np.any(slice_mask, axis=1)
    cols = np.any(slice_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    side_a_mm = float((rmax - rmin + 1) * dx)
    side_b_mm = float((cmax - cmin + 1) * dy)
    longest = max(side_a_mm, side_b_mm)
    perpendicular = min(side_a_mm, side_b_mm)

    return {
        "slice_index": z,
        "longest_diameter_mm": longest,
        "perpendicular_mm": perpendicular,
        "spd_mm2": longest * perpendicular,
        "n_tumor_pixels_in_slice": int(slice_mask.sum()),
    }
