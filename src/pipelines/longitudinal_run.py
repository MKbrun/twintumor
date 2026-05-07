"""
Per-patient RANO pipeline:
    seg.nii at every timepoint -> volume_mm3 -> TreatmentAgent label

Reads volumes directly from each timepoint's seg.nii, so it does not require
a pre-computed tumor_volume.csv. Works on any folder shaped like:

    <patient_dir>/
        baseline/seg.nii
        progression/FU{1..N}/seg.nii
        remission/FU{1..N}/seg.nii
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.agent.treatment_agent import TreatmentAgent
from src.data.paths import RAW_DATA_DIR
from src.io.nifti_loader import load_mask
from src.metrics.volume import compute_tumor_volume


SCENARIOS = ("progression", "remission")


def _volume_info(seg_path: Path) -> Dict[str, float]:
    mask, voxel_dims = load_mask(seg_path)
    return compute_tumor_volume(mask, voxel_dims)


def _discover_followups(scenario_dir: Path) -> List[Path]:
    """Return FU folders sorted by their numeric suffix."""
    fu_dirs = [d for d in scenario_dir.iterdir()
               if d.is_dir() and re.fullmatch(r"FU\d+", d.name)]
    return sorted(fu_dirs, key=lambda d: int(d.name[2:]))


def _run_branch(
    patient_id: str,
    scenario: str,
    baseline_volume: float,
    baseline_voxel_count: int,
    followup_dirs: List[Path],
    enable_pseudoprogression: bool,
) -> List[Dict[str, float | str]]:
    agent = TreatmentAgent(
        initial_volume=baseline_volume,
        enable_pseudoprogression=enable_pseudoprogression,
    )

    results: List[Dict[str, float | str]] = [{
        "patient_id": patient_id,
        "scenario": scenario,
        "timepoint": "Baseline",
        "volume_mm3": baseline_volume,
        "voxel_count": baseline_voxel_count,
        "status": "Baseline",
        "raw_status": "Baseline",
        "percent_change_vs_baseline": 0.0,
        "percent_change_vs_smallest": 0.0,
    }]

    for fu_dir in followup_dirs:
        info = _volume_info(fu_dir / "seg.nii")
        evaluation = agent.evaluate(info["volume_mm3"])
        results.append({
            "patient_id": patient_id,
            "scenario": scenario,
            "timepoint": fu_dir.name,
            "volume_mm3": info["volume_mm3"],
            "voxel_count": info["voxel_count"],
            "status": evaluation["status"],
            "raw_status": evaluation["raw_status"],
            "percent_change_vs_baseline": evaluation["percent_change_vs_baseline"],
            "percent_change_vs_smallest": evaluation["percent_change_vs_smallest"],
        })

    # Patch finalised statuses (pseudoprogression may have rewritten earlier ones)
    final = agent.final_statuses()
    for i, status in enumerate(final):
        results[i]["status"] = status

    return results


def run_patient_pipeline(
    patient_dir: str | Path,
    enable_pseudoprogression: bool = False,
) -> pd.DataFrame:
    patient_dir = Path(patient_dir)
    patient_id = patient_dir.name

    baseline_seg = patient_dir / "baseline" / "seg.nii"
    if not baseline_seg.exists():
        raise FileNotFoundError(f"Missing baseline seg: {baseline_seg}")
    baseline = _volume_info(baseline_seg)

    all_rows: List[Dict[str, float | str]] = []
    for scenario in SCENARIOS:
        scen_dir = patient_dir / scenario
        if not scen_dir.exists():
            continue
        fu_dirs = _discover_followups(scen_dir)
        if not fu_dirs:
            continue
        all_rows.extend(_run_branch(
            patient_id=patient_id,
            scenario=scenario,
            baseline_volume=baseline["volume_mm3"],
            baseline_voxel_count=baseline["voxel_count"],
            followup_dirs=fu_dirs,
            enable_pseudoprogression=enable_pseudoprogression,
        ))

    return pd.DataFrame(all_rows)


def print_results(df: pd.DataFrame) -> None:
    if df.empty:
        print("No results to display.")
        return
    for scenario, scen_df in df.groupby("scenario"):
        print(f"\nScenario: {scenario}")
        print("-" * 64)
        for _, row in scen_df.iterrows():
            tp = row["timepoint"]
            vol = row["volume_mm3"]
            status = row["status"]
            if tp == "Baseline":
                print(f"  {tp:9} {vol:10.2f} mm^3   ({status})")
            else:
                pcb = row["percent_change_vs_baseline"]
                pcs = row["percent_change_vs_smallest"]
                print(f"  {tp:9} {vol:10.2f} mm^3 -> {status:24}  "
                      f"vs baseline: {pcb:+7.2f}%, vs smallest: {pcs:+7.2f}%")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the RANO pipeline on one patient folder.")
    p.add_argument("--patient-dir", type=Path, required=False,
                   help="Patient folder (containing baseline/ and progression/ remission/).")
    p.add_argument("--data-root", type=Path, default=RAW_DATA_DIR,
                   help=f"If --patient-dir is omitted, pick the first Mets_* folder under this root "
                        f"(default: {RAW_DATA_DIR}).")
    p.add_argument("--pseudoprogression", action="store_true",
                   help="Enable RANO Step-3 pseudoprogression handling.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.patient_dir is None:
        candidates = sorted(p for p in args.data_root.iterdir()
                            if p.is_dir() and p.name.startswith("Mets_"))
        if not candidates:
            raise SystemExit(f"No Mets_* folders found under {args.data_root}")
        patient_dir = candidates[0]
    else:
        patient_dir = args.patient_dir

    df = run_patient_pipeline(patient_dir, enable_pseudoprogression=args.pseudoprogression)
    print_results(df)
