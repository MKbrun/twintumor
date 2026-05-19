"""
Build the canonical wide-format tumor-volume dataset by walking a folder of
patient series and computing volumes from each timepoint's seg.nii.

Expected folder layout (under --data-root):

    <data-root>/
        Mets_005/
            baseline/seg.nii
            progression/FU1/seg.nii
            progression/FU2/seg.nii
            ...
            progression/FU5/seg.nii
            remission/FU1/seg.nii
            ...
            remission/FU5/seg.nii
        Mets_010/
            ...

Output (default: data/processed/tumor_volumes.csv):
    subject, baseline,
    progression_FU1..FU5,
    remission_FU1..FU5

All numeric values are tumor volume in mm^3.

Usage
-----
    python -m src.pipelines.build_volumes_csv --data-root E:/series
    python -m src.pipelines.build_volumes_csv --data-root data/raw --output data/processed/tumor_volumes.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

from src.data.paths import RAW_DATA_DIR, TUMOR_VOLUMES_CSV
from src.io.nifti_loader import load_mask
from src.metrics.volume import compute_tumor_volume


SCENARIOS = ("progression", "remission")
NUM_FOLLOWUPS = 5


def _volume_mm3(seg_path: Path) -> float:
    mask, voxel_dims = load_mask(seg_path)
    return compute_tumor_volume(mask, voxel_dims)["volume_mm3"]


def discover_patients(data_root: Path) -> List[Path]:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    return sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("Mets_"))


def process_patient(patient_dir: Path) -> Optional[Dict[str, float]]:
    """Return a row dict for a patient, or None if the layout is incomplete."""
    baseline_seg = patient_dir / "baseline" / "seg.nii"
    if not baseline_seg.exists():
        return None

    row: Dict[str, float] = {
        "subject": patient_dir.name,
        "baseline": _volume_mm3(baseline_seg),
    }

    for scenario in SCENARIOS:
        for i in range(1, NUM_FOLLOWUPS + 1):
            seg_path = patient_dir / scenario / f"FU{i}" / "seg.nii"
            if not seg_path.exists():
                return None
            row[f"{scenario}_FU{i}"] = _volume_mm3(seg_path)

    return row


ProgressCallback = Callable[[int, int, str], None]
"""(index, total, patient_id) — called once per patient. Use for UI progress bars."""


def build_dataframe(
    data_root: Path,
    verbose: bool = True,
    progress: Optional[ProgressCallback] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    patients = discover_patients(data_root)
    total = len(patients)
    for i, patient_dir in enumerate(patients, start=1):
        try:
            row = process_patient(patient_dir)
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
                print(f"[OK  ] {patient_dir.name}: baseline={row['baseline']:.1f} mm^3")
        if progress is not None:
            progress(i, total, patient_dir.name)
    return pd.DataFrame(rows)


def build_and_save(
    data_root: str | Path,
    output: str | Path = TUMOR_VOLUMES_CSV,
    progress: Optional[ProgressCallback] = None,
    verbose: bool = True,
) -> Path:
    """
    Auto-detect whether `data_root` is in the legacy layout (per-FU `seg.nii`)
    or in AIMI's layout (no FU `seg.nii`, t1_gd at every timepoint, Otsu in
    baseline ROI). Route to the appropriate extractor either way.
    """
    from src.pipelines.otsu_signal_extractor import (
        build_aimi_dataframe,
        detect_root_layout,
    )
    data_root = Path(data_root)
    output = Path(output)
    layout = detect_root_layout(data_root)
    if verbose:
        print(f"[layout] detected: {layout}")

    if layout == "aimi":
        df = build_aimi_dataframe(data_root, verbose=verbose, progress=progress)
    elif layout == "legacy":
        df = build_dataframe(data_root, verbose=verbose, progress=progress)
    else:
        raise RuntimeError(
            f"Could not detect a usable layout under {data_root}. "
            "Expected Mets_* folders containing baseline/seg.nii."
        )

    if df.empty:
        raise RuntimeError(f"No patients with complete layout found under {data_root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return output


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build tumor_volumes.csv from a folder of Mets_* patient series.")
    p.add_argument("--data-root", type=Path, default=RAW_DATA_DIR,
                   help=f"Folder containing Mets_* subject folders (default: {RAW_DATA_DIR})")
    p.add_argument("--output", type=Path, default=TUMOR_VOLUMES_CSV,
                   help=f"Output CSV path (default: {TUMOR_VOLUMES_CSV})")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out = build_and_save(args.data_root, args.output)
    print(f"\nSaved {out}")
