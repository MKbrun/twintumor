"""
AIMI-layout MRI ingestion — Otsu thresholding of `t1_gd.nii` inside the
baseline `seg.nii` ROI, applied per-timepoint.

Why
---
AIMI's source layout (E:/mri_series/series) provides:
    Mets_xxx/
        baseline/{seg, t1_gd, t1_pre, flair}.nii    — seg defines the ROI
        progression/FU{1..5}/{t1_gd, t1_pre, flair}.nii    — NO seg.nii
        remission/FU{1..5}/{t1_gd, t1_pre, flair}.nii      — NO seg.nii

Per AIMI's e-mail, the per-timepoint "tumor signal" value is the percentage
of voxels inside the baseline ROI whose `t1_gd.nii` intensity at that
timepoint is above an Otsu threshold computed from those same in-ROI
intensities. Each timepoint computes its own threshold so they are all
processed by the same rule ("consistent signal-to-signal comparison").

Output schema matches `data/processed/tumor_volumes.csv`:

    subject, baseline,
    progression_FU1..5,
    remission_FU1..5

so the rest of the framework (loader, ML predictor, visualisation) sees
exactly the same column names regardless of the ingestion path used.

Note on AIMI parity
-------------------
Our reimplementation will not exactly reproduce `consistent_tumor_analysis.csv`
because the email does not specify the intensity-normalisation step or the
exact histogram-binning. Empirically we get ~60% where the reference CSV
says 45% for Mets_005 baseline. The methodology is the same family; the
absolute scale may differ. Documented in the README.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Optional

import nibabel as nib
import numpy as np
import pandas as pd

from src.data.paths import RAW_DATA_DIR, TUMOR_VOLUMES_CSV


SCENARIOS = ("progression", "remission")
NUM_FOLLOWUPS = 5
ProgressCallback = Callable[[int, int, str], None]


# ----------------------------------------------------------------- Otsu

def otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
    """
    Otsu's method on a 1-D intensity array. Returns the threshold value.

    Uses bin centres as candidate thresholds and selects the one that
    maximises between-class variance σ_b² = ω(μ_T·ω - μ)² / (ω(1-ω)).
    Works on negative or non-integer intensities (no need for 0..255).
    """
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    vmin, vmax = float(v.min()), float(v.max())
    if vmax == vmin:
        return vmin
    hist, edges = np.histogram(v, bins=n_bins, range=(vmin, vmax))
    total = hist.sum()
    if total == 0:
        return vmin
    p = hist / total
    centres = 0.5 * (edges[:-1] + edges[1:])
    omega = np.cumsum(p)
    mu = np.cumsum(p * centres)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    safe = denom > 0
    sigma_b2 = np.zeros_like(denom)
    sigma_b2[safe] = (mu_t * omega[safe] - mu[safe]) ** 2 / denom[safe]
    return float(centres[int(np.argmax(sigma_b2))])


# ----------------------------------------------------------------- per-timepoint signal

def signal_percent_in_roi(
    t1_gd: np.ndarray,
    roi: np.ndarray,
    fixed_threshold: Optional[float] = None,
    target_mean: Optional[float] = None,
) -> Dict[str, float]:
    """
    Apply a threshold inside the ROI on the given t1_gd volume.

    Two modes:
      * `fixed_threshold=None` (per-timepoint Otsu) — recompute Otsu on this
        volume's in-ROI intensities and use that.
      * `fixed_threshold=<value>` — use the given threshold (typically derived
        from baseline). When `target_mean` is also provided, rescale this
        volume's in-ROI intensities so their mean matches `target_mean`
        before thresholding (baseline-anchored mean matching).
    """
    if t1_gd.shape != roi.shape:
        raise ValueError(
            f"t1_gd shape {t1_gd.shape} does not match ROI shape {roi.shape}; "
            "the framework assumes aligned baseline-space images."
        )
    inside = t1_gd[roi].astype(float)
    n_roi = int(inside.size)
    if n_roi == 0:
        return {"n_above": 0, "n_roi": 0, "percent": 0.0, "threshold": 0.0}

    if fixed_threshold is None:
        th = otsu_threshold(inside)
    else:
        th = float(fixed_threshold)
        if target_mean is not None:
            cur_mean = float(inside.mean())
            if cur_mean > 0:
                inside = inside * (float(target_mean) / cur_mean)

    n_above = int(np.count_nonzero(inside > th))
    return {
        "n_above": n_above,
        "n_roi": n_roi,
        "percent": 100.0 * n_above / n_roi,
        "threshold": th,
    }


# ----------------------------------------------------------------- per-patient row

def _load(path: Path) -> np.ndarray:
    return np.squeeze(nib.load(str(path)).get_fdata())


def extract_patient_row(
    patient_dir: Path,
    mode: str = "baseline_anchored",
) -> Optional[Dict[str, float]]:
    """
    Build one wide-CSV row for a patient in AIMI layout.

    Parameters
    ----------
    patient_dir : Path
    mode : {"baseline_anchored", "per_timepoint"}
        - **baseline_anchored** (default): compute Otsu *once* on baseline's
          in-ROI intensities; at every FU rescale in-ROI intensities so their
          mean matches baseline's, then apply the baseline threshold. Removes
          per-FU intensity drift; trajectory shape becomes biologically
          interpretable.
        - **per_timepoint**: recompute Otsu independently at every FU. Most
          literal reading of "Otsu within the segmentation ROI"; sensitive to
          scanner intensity drift.

    Returns None when the folder is incomplete (caller may skip).
    """
    if mode not in ("baseline_anchored", "per_timepoint"):
        raise ValueError(f"mode must be baseline_anchored|per_timepoint, got {mode!r}")

    base_seg_path = patient_dir / "baseline" / "seg.nii"
    base_t1_path  = patient_dir / "baseline" / "t1_gd.nii"
    if not base_seg_path.exists() or not base_t1_path.exists():
        return None

    seg = _load(base_seg_path)
    if seg.ndim != 3:
        return None
    roi = seg > 0
    if not roi.any():
        return None

    base_t1 = _load(base_t1_path)
    base_inside = base_t1[roi].astype(float)
    base_threshold = otsu_threshold(base_inside)
    base_mean_in_roi = float(base_inside.mean())

    base_stats = signal_percent_in_roi(base_t1, roi)  # baseline reads same in either mode

    row: Dict[str, float] = {
        "subject": patient_dir.name,
        "baseline": float(base_stats["percent"]),
    }

    for scenario in SCENARIOS:
        for fu in range(1, NUM_FOLLOWUPS + 1):
            fu_t1 = patient_dir / scenario / f"FU{fu}" / "t1_gd.nii"
            if not fu_t1.exists():
                return None
            t1 = _load(fu_t1)
            if mode == "baseline_anchored":
                stats = signal_percent_in_roi(
                    t1, roi,
                    fixed_threshold=base_threshold,
                    target_mean=base_mean_in_roi,
                )
            else:
                stats = signal_percent_in_roi(t1, roi)
            row[f"{scenario}_FU{fu}"] = float(stats["percent"])

    return row


# ----------------------------------------------------------------- layout detection

def is_aimi_layout(patient_dir: Path) -> bool:
    """
    A patient folder is "AIMI layout" if `baseline/seg.nii` is present but
    no FU folder contains a `seg.nii`. The legacy layout (E:/series) has a
    placeholder `seg.nii` in every FU folder.
    """
    if not (patient_dir / "baseline" / "seg.nii").exists():
        return False
    for scenario in SCENARIOS:
        scen_dir = patient_dir / scenario
        if not scen_dir.is_dir():
            continue
        for fu_dir in scen_dir.iterdir():
            if fu_dir.is_dir() and (fu_dir / "seg.nii").exists():
                return False
    return True


def detect_root_layout(data_root: Path) -> str:
    """
    Inspect a few representative patients under `data_root` and return
    'aimi', 'legacy', or 'unknown'.
    """
    if not data_root.exists():
        return "unknown"
    candidates = sorted(
        p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("Mets_")
    )
    sampled = [p for p in candidates if (p / "baseline" / "seg.nii").exists()][:5]
    if not sampled:
        return "unknown"
    aimi_votes = sum(1 for p in sampled if is_aimi_layout(p))
    if aimi_votes == len(sampled):
        return "aimi"
    if aimi_votes == 0:
        return "legacy"
    # Mixed — be conservative and call it legacy so the user is warned.
    return "legacy"


# ----------------------------------------------------------------- batch

def build_aimi_dataframe(
    data_root: Path,
    verbose: bool = True,
    progress: Optional[ProgressCallback] = None,
    mode: str = "baseline_anchored",
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    patients = sorted(
        p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("Mets_")
    )
    total = len(patients)
    for i, patient_dir in enumerate(patients, start=1):
        try:
            row = extract_patient_row(patient_dir, mode=mode)
        except Exception as exc:
            if verbose:
                print(f"[ERR ] {patient_dir.name}: {exc}")
            row = None
        if row is None:
            if verbose:
                print(f"[SKIP] {patient_dir.name}: incomplete folder layout")
        else:
            rows.append(row)
            if verbose:
                print(f"[OK  ] {patient_dir.name}: baseline={row['baseline']:.2f}%")
        if progress is not None:
            progress(i, total, patient_dir.name)
    return pd.DataFrame(rows)


def build_and_save_aimi(
    data_root: str | Path,
    output: str | Path = TUMOR_VOLUMES_CSV,
    progress: Optional[ProgressCallback] = None,
    verbose: bool = True,
) -> Path:
    data_root = Path(data_root)
    output = Path(output)
    df = build_aimi_dataframe(data_root, verbose=verbose, progress=progress)
    if df.empty:
        raise RuntimeError(f"No patients with complete AIMI layout under {data_root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return output


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a wide-format signal CSV from the AIMI MRI layout "
                    "(t1_gd Otsu within baseline ROI, per timepoint)."
    )
    p.add_argument("--data-root", type=Path, default=RAW_DATA_DIR,
                   help=f"Folder containing Mets_* patients in AIMI layout "
                        f"(default: {RAW_DATA_DIR})")
    p.add_argument("--output", type=Path, default=TUMOR_VOLUMES_CSV,
                   help=f"Output CSV (default: {TUMOR_VOLUMES_CSV})")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out = build_and_save_aimi(args.data_root, args.output)
    print(f"\nSaved {out}")
