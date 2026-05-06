from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd


DATASET_ROOT = Path("/Users/phillipovera/Downloads/series")
CSV_PATH = DATASET_ROOT / "tumor_volumes_all_subjects_v3.csv"
OUTPUT_PATH = Path("ml_longitudinal_dataset.csv")

SUBJECT_ID_COLUMN_CANDIDATES = ["subject"]

PROGRESSION_COLUMNS = [
    "progression_FU1",
    "progression_FU2",
    "progression_FU3",
    "progression_FU4",
    "progression_FU5",
]

REMISSION_COLUMNS = [
    "remission_FU1",
    "remission_FU2",
    "remission_FU3",
    "remission_FU4",
    "remission_FU5",
]


def find_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first matching column name from a list of candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"None of these columns were found in the CSV: {candidates}")


def parse_numeric(value: Any) -> float:
    """Convert a value to float, or return NaN if conversion fails."""
    if pd.isna(value):
        return np.nan

    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def safe_relative_change(current: float, reference: float) -> float:
    """Compute (current - reference) / reference safely."""
    if pd.isna(current) or pd.isna(reference) or reference == 0:
        return np.nan
    return (current - reference) / reference


def load_baseline_volume_from_seg(seg_path: Path) -> float:
    """Load a baseline segmentation and compute tumour volume in mm^3."""
    if not seg_path.exists():
        raise FileNotFoundError(f"Missing segmentation file: {seg_path}")

    # Load the NIfTI segmentation and read voxel values
    img = nib.load(str(seg_path))
    data = img.get_fdata()

    # Count all non-zero voxels as tumour voxels
    voxel_count = np.count_nonzero(data)

    # Compute voxel volume from image spacing
    zooms = img.header.get_zooms()[:3]
    voxel_volume_mm3 = float(np.prod(zooms)) if len(zooms) == 3 else 1.0

    # Total volume = number of tumour voxels * volume per voxel
    return float(voxel_count * voxel_volume_mm3)


def get_subject_row(
    df: pd.DataFrame,
    subject_id_col: str,
    subject_id: str,
) -> pd.Series | None:
    """Return the CSV row for one subject, or None if no match is found."""
    matches = df[df[subject_id_col].astype(str) == str(subject_id)]
    if matches.empty:
        return None
    return matches.iloc[0]


def extract_named_volumes(row: pd.Series, columns: list[str]) -> list[float]:
    """Extract a list of follow-up volumes from selected CSV columns."""
    return [parse_numeric(row[col]) for col in columns]


def make_regression_rows(
    subject_id: str,
    trajectory_type: str,
    baseline_volume: float,
    fu_volumes: list[float],
) -> list[dict[str, Any]]:
    """Create one ML row per next-followup prediction step."""
    rows: list[dict[str, Any]] = []

    # Put baseline first, then FU1-FU5 after it
    all_volumes = [baseline_volume] + fu_volumes

    # Example:
    # baseline -> predict FU1
    # baseline + FU1 -> predict FU2
    # baseline + FU1 + FU2 -> predict FU3
    for target_idx in range(1, len(all_volumes)):
        target_next_volume = all_volumes[target_idx]

        # Skip rows where the target follow-up is missing
        if pd.isna(target_next_volume):
            continue

        # Known history up to the point just before the target
        known = all_volumes[:target_idx]
        baseline = known[0]
        known_followups = known[1:]

        last_observed = known[-1]
        previous_observed = known[-2] if len(known) >= 2 else baseline

        # Build one training row with raw values + summary features
        row = {
            "subject_id": subject_id,
            "trajectory_type": trajectory_type,
            "target_fu_index": target_idx,
            "baseline_volume": baseline,
            "fu1_volume": known[1] if len(known) > 1 else np.nan,
            "fu2_volume": known[2] if len(known) > 2 else np.nan,
            "fu3_volume": known[3] if len(known) > 3 else np.nan,
            "fu4_volume": known[4] if len(known) > 4 else np.nan,
            "num_known_followups": len(known_followups),
            "last_observed_volume": last_observed,
            "previous_observed_volume": previous_observed,
            "min_previous_volume": float(np.nanmin(known)),
            "max_previous_volume": float(np.nanmax(known)),
            "mean_previous_volume": float(np.nanmean(known)),
            "change_last_vs_baseline": safe_relative_change(last_observed, baseline),
            "change_last_vs_previous": safe_relative_change(last_observed, previous_observed),
            "target_next_volume": target_next_volume,
            "target_next_log_volume": float(np.log1p(target_next_volume)),
        }

        rows.append(row)

    return rows


def build_dataset(dataset_root: Path, csv_path: Path) -> pd.DataFrame:
    """Build the full ML dataset from baseline segmentations and synthetic trajectories."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Read the CSV file with generated progression/remission trajectories
    df_csv = pd.read_csv(csv_path)
    subject_id_col = find_first_existing_column(df_csv, SUBJECT_ID_COLUMN_CANDIDATES)

    # Make sure all expected follow-up columns exist
    missing_required_columns = [
        col for col in PROGRESSION_COLUMNS + REMISSION_COLUMNS if col not in df_csv.columns
    ]
    if missing_required_columns:
        raise KeyError(
            f"CSV is missing required columns: {missing_required_columns}\n"
            f"Available columns: {df_csv.columns.tolist()}"
        )

    rows: list[dict[str, Any]] = []
    skipped_subjects: list[str] = []

    # Only use actual subject folders like Mets_005, Mets_014, etc.
    subject_dirs = sorted(
        [p for p in dataset_root.iterdir() if p.is_dir() and p.name.startswith("Mets_")]
    )

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        seg_path = subject_dir / "baseline" / "seg.nii"

        # Load baseline tumour volume from the segmentation
        try:
            baseline_volume = load_baseline_volume_from_seg(seg_path)
        except Exception as e:
            skipped_subjects.append(f"{subject_id} (baseline error: {e})")
            continue

        # Find the matching subject row in the CSV
        subject_row = get_subject_row(df_csv, subject_id_col, subject_id)
        if subject_row is None:
            skipped_subjects.append(f"{subject_id} (missing in CSV)")
            continue

        # Read the two generated trajectories for this subject
        progression_volumes = extract_named_volumes(subject_row, PROGRESSION_COLUMNS)
        remission_volumes = extract_named_volumes(subject_row, REMISSION_COLUMNS)

        # Build ML rows for the progression trajectory
        progression_rows = make_regression_rows(
            subject_id=subject_id,
            trajectory_type="progression",
            baseline_volume=baseline_volume,
            fu_volumes=progression_volumes,
        )

        # Build ML rows for the remission trajectory
        remission_rows = make_regression_rows(
            subject_id=subject_id,
            trajectory_type="remission",
            baseline_volume=baseline_volume,
            fu_volumes=remission_volumes,
        )

        rows.extend(progression_rows)
        rows.extend(remission_rows)

    dataset = pd.DataFrame(rows)

    if dataset.empty:
        print("No rows were created.")
        return dataset

    # Print a quick summary so it is easy to sanity-check the output
    print(f"Built dataset with {len(dataset)} rows.")
    print(f"Unique generated subjects used: {dataset['subject_id'].nunique()}")

    print("\nTrajectory counts:")
    print(dataset["trajectory_type"].value_counts())

    print("\nTarget FU index counts:")
    print(dataset["target_fu_index"].value_counts().sort_index())

    if skipped_subjects:
        print("\nSkipped subjects:")
        for item in skipped_subjects[:20]:
            print(" -", item)
        if len(skipped_subjects) > 20:
            print(f" - ... and {len(skipped_subjects) - 20} more")

    return dataset


def main() -> None:
    """Build the dataset, save it to CSV, and print a short summary."""
    dataset = build_dataset(DATASET_ROOT, CSV_PATH)

    if dataset.empty:
        print("Dataset is empty. Check paths, subject IDs, and segmentation files.")
        return

    # Save the final ML dataset
    dataset.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved ML dataset to: {OUTPUT_PATH.resolve()}")

    print("\nColumns:")
    print(dataset.columns.tolist())

    print("\nFirst 10 rows:")
    print(dataset.head(10))

    print("\nBasic summary:")
    print(dataset.describe(include="all"))


if __name__ == "__main__":
    main()