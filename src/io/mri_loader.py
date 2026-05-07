"""Load full 3D MRI volumes (t1_gd, flair, …) — sibling of nifti_loader."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np


def load_mri_volume(path: str | Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Load a 3D MRI volume and return (data, voxel_dims_in_mm)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MRI file not found: {path}")
    if not (str(path).endswith(".nii") or str(path).endswith(".nii.gz")):
        raise ValueError(f"Unsupported file format: {path}")

    img = nib.load(str(path))
    data = np.squeeze(img.get_fdata())
    if data.ndim != 3:
        raise ValueError(f"Expected 3D MRI volume, got shape {data.shape}")

    voxel_dims = tuple(float(v) for v in img.header.get_zooms()[:3])
    return data, voxel_dims
