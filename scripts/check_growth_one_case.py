from pathlib import Path
import nibabel as nib
import numpy as np

# Brukt til å sjekke datasett 1

CASE = Path("Mets_256")  # Change to the case you want to check

paths = [
    CASE / "progression" / "FU1" / "seg.nii",
    CASE / "progression" / "FU5" / "seg.nii",
]

for p in paths:
    img = nib.load(str(p))
    data = img.get_fdata()
    vox = int(np.count_nonzero(data > 0))
    dx, dy, dz = img.header.get_zooms()[:3]
    vol_ml = vox * float(dx*dy*dz) / 1000.0
    print(p, "| voxels>0:", vox, "| volume_ml:", vol_ml, "| max:", float(data.max()))
