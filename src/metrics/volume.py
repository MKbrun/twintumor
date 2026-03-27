from typing import Dict, Tuple

import numpy as np


# Compute tumor volume from a 3D segmentation mask.
#
# - Count tumor voxels (mask values > threshold)
# - Compute voxel volume from voxel spacing (dx * dy * dz)
# - Compute total volume in mm^3 and cm^3
def compute_tumor_volume(
    mask_data: np.ndarray,
    voxel_dims: Tuple[float, float, float],
    threshold: float = 0.0,
) -> Dict[str, float]:
    # Basic shape checks
    if mask_data.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {mask_data.shape}")

    if len(voxel_dims) != 3:
        raise ValueError(f"Expected 3 voxel dimensions, got {voxel_dims}")

    dx, dy, dz = voxel_dims

    # Voxel spacing must be positive
    if dx <= 0 or dy <= 0 or dz <= 0:
        raise ValueError(f"Voxel dimensions must be positive, got {voxel_dims}")

    # Tumor voxels are values above the threshold (binary masks typically use 0/1)
    voxel_count = int(np.sum(mask_data > threshold))

    # Convert voxel spacing to a voxel volume (mm^3)
    voxel_volume_mm3 = float(dx * dy * dz)

    # Total tumor volume
    volume_mm3 = float(voxel_count * voxel_volume_mm3)

    return {
    "voxel_count": voxel_count,
    "voxel_volume_mm3": voxel_volume_mm3,
    "volume_mm3": volume_mm3,
}