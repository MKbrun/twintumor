from src.io.nifti_loader import load_nifti
from src.metrics.volume import compute_tumor_volume

MASK_PATH = "data/raw/Mets_010/baseline/seg.nii"

mask_data, voxel_dims = load_nifti(MASK_PATH)
result = compute_tumor_volume(mask_data, voxel_dims)

print(result)