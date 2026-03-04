"""
Compute tumor volume from a segmentation mask.

Should implement:
- compute_tumor_volume(mask_data, voxel_dims)

Steps:
1. Count tumor voxels in mask
2. Compute voxel volume (dx * dy * dz)
3. Multiply voxel_count * voxel_volume

Returns:
- voxel_count
- volume_mm3
- volume_cm3
"""