"""
Compute per-timepoint signal masks (Otsu within the baseline ROI) for one
patient in the AIMI layout. These are the *actual* observed signal masks
at each follow-up — used by the heatmap tab as the "ground truth" panel
alongside the framework's predicted isotropic-dilation overlay.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.io.mri_loader import load_mri_volume
from src.io.nifti_loader import load_mask
from src.pipelines.otsu_signal_extractor import otsu_threshold


@dataclass
class FuSignalMask:
    label: str                 # "FU3", "FU4", "FU5", …
    t1_gd_3d: np.ndarray       # the FU's t1_gd volume
    signal_mask_3d: np.ndarray # bool, same shape as roi
    threshold: float           # Otsu threshold used
    n_above: int
    n_roi: int

    @property
    def percent(self) -> float:
        return 100.0 * self.n_above / max(self.n_roi, 1)


def compute_fu_signal_masks(
    patient_dir: str | Path,
    scenario: str,
    timepoints: List[str],
    mode: str = "baseline_anchored",
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float], Dict[str, FuSignalMask]]:
    """
    For every requested timepoint, load `<patient>/<scenario>/<TP>/t1_gd.nii`,
    apply Otsu inside the baseline ROI defined by `<patient>/baseline/seg.nii`,
    and return the resulting signal mask.

    Returns
    -------
    roi_3d        : bool 3D mask from baseline/seg.nii
    baseline_t1_gd: float 3D image
    voxel_dims    : (dx, dy, dz) in mm
    masks         : dict label -> FuSignalMask
    """
    patient_dir = Path(patient_dir)
    if scenario not in ("progression", "remission"):
        raise ValueError(f"scenario must be progression|remission, got {scenario!r}")

    seg_path = patient_dir / "baseline" / "seg.nii"
    if not seg_path.exists():
        raise FileNotFoundError(f"Missing baseline seg: {seg_path}")
    seg_3d, voxel_dims = load_mask(seg_path)
    roi_3d = seg_3d > 0

    base_t1_path = patient_dir / "baseline" / "t1_gd.nii"
    if not base_t1_path.exists():
        raise FileNotFoundError(f"Missing baseline t1_gd: {base_t1_path}")
    baseline_t1, _ = load_mri_volume(base_t1_path)
    if baseline_t1.shape != roi_3d.shape:
        raise ValueError(
            f"Baseline t1_gd shape {baseline_t1.shape} does not match seg {roi_3d.shape}"
        )

    # Pre-compute baseline-derived constants for the anchored mode.
    base_inside = baseline_t1[roi_3d].astype(float)
    baseline_threshold = otsu_threshold(base_inside)
    baseline_mean = float(base_inside.mean())

    masks: Dict[str, FuSignalMask] = {}
    for tp in timepoints:
        t1_path = patient_dir / scenario / tp / "t1_gd.nii"
        if not t1_path.exists():
            continue
        t1_data, _ = load_mri_volume(t1_path)
        if t1_data.shape != roi_3d.shape:
            continue
        inside = t1_data[roi_3d].astype(float)
        if inside.size == 0:
            continue

        if mode == "baseline_anchored":
            cur_mean = float(inside.mean())
            scale = (baseline_mean / cur_mean) if cur_mean > 0 else 1.0
            th = baseline_threshold
            scaled_volume = t1_data.astype(float) * scale
            signal_3d = (scaled_volume > th) & roi_3d
        else:  # per_timepoint
            th = otsu_threshold(inside)
            signal_3d = (t1_data > th) & roi_3d

        n_above = int(signal_3d.sum())
        masks[tp] = FuSignalMask(
            label=tp,
            t1_gd_3d=t1_data,
            signal_mask_3d=signal_3d,
            threshold=th,
            n_above=n_above,
            n_roi=int(roi_3d.sum()),
        )
    return roi_3d, baseline_t1, voxel_dims, masks
