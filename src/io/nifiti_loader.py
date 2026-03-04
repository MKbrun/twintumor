"""
Load tumor segmentation masks from NIfTI files (.nii / .nii.gz).

Should implement:
- load_mask(path) -> (mask_data, voxel_dims)

Returns:
- mask_data: 3D numpy array
- voxel_dims: (dx, dy, dz) voxel spacing in mm

Used as the first step in the pipeline before volume calculation.
"""