from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np



# Load a tumor segmentation mask from a NIfTI file (.nii / .nii.gz)
# and return both the mask data and voxel spacing.
def load_mask(path: str | Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    path = Path(path)

    # Make sure the file exists
    if not path.exists():
        raise FileNotFoundError(f"Mask file not found: {path}")

    # Accept .nii and .nii.gz files
    if not (str(path).endswith(".nii") or str(path).endswith(".nii.gz")):
        raise ValueError(f"Unsupported file format: {path}")

    # Load image data and remove singleton dimensions if present
    img = nib.load(str(path))
    mask_data = np.squeeze(img.get_fdata())

    # The pipeline expects a 3D mask
    if mask_data.ndim != 3:
        raise ValueError(
            f"Expected a 3D mask, but got shape {mask_data.shape} from {path}"
        )

    # Read voxel spacing in mm from the NIfTI header
    voxel_dims = img.header.get_zooms()[:3]
    voxel_dims = tuple(float(v) for v in voxel_dims)

    return mask_data, voxel_dims