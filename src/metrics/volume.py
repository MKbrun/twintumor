from typing import Any

import numpy as np


def compute_tumor_volume(
    mask_data: np.ndarray,
    voxel_dims: tuple[float, float, float],
    tumor_labels: tuple[int, ...] = (1,)
) -> dict[str, Any]:
    """
    Compute tumor volume from a segmentation mask.

    Args:
        mask_data: 3D segmentation array
        voxel_dims: voxel spacing in mm, e.g. (1.0, 1.0, 1.0)
        tumor_labels: labels that count as tumor

    Returns:
        {
            "voxel_count": int,
            "voxel_volume_mm3": float,
            "volume_mm3": float,
            "volume_cm3": float
        }
    """
    if mask_data.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {mask_data.shape}")

    tumor_mask = np.isin(mask_data, tumor_labels)
    voxel_count = int(np.count_nonzero(tumor_mask))

    voxel_volume_mm3 = float(voxel_dims[0] * voxel_dims[1] * voxel_dims[2])
    volume_mm3 = float(voxel_count * voxel_volume_mm3)
    volume_cm3 = float(volume_mm3 / 1000.0)

    return {
        "voxel_count": voxel_count,
        "voxel_volume_mm3": voxel_volume_mm3,
        "volume_mm3": volume_mm3,
        "volume_cm3": volume_cm3,
    }