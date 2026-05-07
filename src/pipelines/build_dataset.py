"""
Run the per-patient RANO pipeline across an entire data root and emit a
long-form CSV of (patient, scenario, timepoint, volume, status, ...).

This complements `build_volumes_csv.py`:
    - build_volumes_csv.py  -> wide CSV used for ML / forecasting
    - build_dataset.py      -> long CSV with RANO statuses per timepoint
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.paths import PROCESSED_DIR, RAW_DATA_DIR
from src.pipelines.longitudinal_run import run_patient_pipeline


DEFAULT_OUTPUT = PROCESSED_DIR / "all_patient_trajectories.csv"


def discover_patient_dirs(data_root: str | Path) -> list[Path]:
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    return sorted(p for p in data_root.iterdir()
                  if p.is_dir() and p.name.startswith("Mets_"))


def build_dataset(
    data_root: str | Path,
    enable_pseudoprogression: bool = False,
) -> pd.DataFrame:
    dfs = []
    for patient_dir in discover_patient_dirs(data_root):
        try:
            patient_df = run_patient_pipeline(
                patient_dir,
                enable_pseudoprogression=enable_pseudoprogression,
            )
            dfs.append(patient_df)
            print(f"[OK  ] {patient_dir.name}")
        except Exception as exc:
            print(f"[WARN] Skipped {patient_dir.name}: {exc}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the RANO pipeline on every patient under a data root.")
    p.add_argument("--data-root", type=Path, default=RAW_DATA_DIR)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--pseudoprogression", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    df = build_dataset(args.data_root, enable_pseudoprogression=args.pseudoprogression)
    if df.empty:
        print("No patient data processed.")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\nSaved {args.output}")
