from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np


def load_nifti(path: str | Path) -> Tuple[np.ndarray, tuple[float, float, float]]:
    """
    Load a NIfTI file and return image data + voxel spacing.

    Returns:
        data: 3D numpy array
        voxel_dims: (dx, dy, dz) in mm
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")

    nii = nib.load(str(path))
    data = nii.get_fdata()
    header = nii.header

    voxel_dims = header.get_zooms()[:3]

    return data, voxel_dims