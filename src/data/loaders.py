"""Wide-format trajectory loading and helpers.

Two CSV schemas are supported transparently:

  * Bundled demo dataset (synthetic, in percent of MRI signal):
        subject, baseline_percent, progression_FU1..5, remission_FU1..5

  * MRI-derived volume dataset (built by build_volumes_csv.py, in mm^3):
        subject, baseline,         progression_FU1..5, remission_FU1..5

After loading, both schemas are normalised to use a single `baseline` column
so downstream code never has to branch on the source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd

from src.data.paths import DEMO_DATASET_CSV, TUMOR_VOLUMES_CSV


# Six timepoints: baseline + 5 follow-ups.
TIME_LABELS: List[str] = ["Baseline", "FU1", "FU2", "FU3", "FU4", "FU5"]
OBSERVED_IDX: List[int] = [0, 1, 2]   # input window
FUTURE_IDX:   List[int] = [3, 4, 5]   # forecast window


SCENARIOS = ("progression", "remission")


def _required_columns() -> set[str]:
    cols = {"subject"}
    for scen in SCENARIOS:
        for i in range(1, 6):
            cols.add(f"{scen}_FU{i}")
    return cols


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Rename `baseline_percent` -> `baseline` if present so the rest of the
    framework sees a single column name."""
    if "baseline" not in df.columns and "baseline_percent" in df.columns:
        df = df.rename(columns={"baseline_percent": "baseline"})
    return df


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load any of the supported wide-format CSVs and normalise the schema."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    df = _normalise(pd.read_csv(csv_path))
    missing = (_required_columns() | {"baseline"}) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    return df


def load_demo_dataset() -> pd.DataFrame:
    """Load the bundled synthetic demo dataset (always available)."""
    return load_dataset(DEMO_DATASET_CSV)


def load_volume_dataset(csv_path: str | Path = TUMOR_VOLUMES_CSV) -> pd.DataFrame:
    """Load the MRI-derived volume dataset (must be built first)."""
    return load_dataset(csv_path)


def get_trajectory(row: pd.Series, scenario: str) -> List[float]:
    """Return the 6-point trajectory (Baseline, FU1..FU5) for one row + scenario."""
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}, got {scenario!r}")
    return [
        float(row["baseline"]),
        float(row[f"{scenario}_FU1"]),
        float(row[f"{scenario}_FU2"]),
        float(row[f"{scenario}_FU3"]),
        float(row[f"{scenario}_FU4"]),
        float(row[f"{scenario}_FU5"]),
    ]


def trajectory_matrix(df: pd.DataFrame, scenario: str) -> Iterable[List[float]]:
    for _, row in df.iterrows():
        yield get_trajectory(row, scenario)


def is_trajectory_flat(trajectory: List[float], rel_tol: float = 1e-6) -> bool:
    """Return True if every timepoint in the trajectory is essentially equal."""
    if not trajectory:
        return True
    arr = [float(v) for v in trajectory]
    spread = max(arr) - min(arr)
    scale = max(abs(max(arr)), abs(min(arr)), 1.0)
    return spread <= rel_tol * scale


def flat_trajectory_summary(df: pd.DataFrame) -> dict:
    """Count how many (subject, scenario) trajectories in a wide-format
    cohort CSV are completely flat. Useful for surfacing source-data
    quality issues (e.g. duplicated FU segmentations)."""
    counts = {"progression": 0, "remission": 0}
    flat_subjects = {"progression": [], "remission": []}
    for _, row in df.iterrows():
        for scen in SCENARIOS:
            traj = get_trajectory(row, scen)
            if is_trajectory_flat(traj):
                counts[scen] += 1
                flat_subjects[scen].append(str(row["subject"]))
    return {
        "n_total": len(df),
        "flat_counts": counts,
        "flat_subjects": flat_subjects,
    }
