from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.agent.treatment_agent import TreatmentAgent
from src.io.nifti_loader import load_mask
from src.io.tumor_volume_csv import load_tumor_volume_csv
from src.metrics.volume import compute_tumor_volume


def compute_baseline_volume_from_mask(baseline_seg_path: str | Path) -> Dict[str, float]:
    mask_data, voxel_dims = load_mask(baseline_seg_path)
    return compute_tumor_volume(mask_data, voxel_dims)


def load_followup_volumes_from_patient_csv(
    csv_path: str | Path,
) -> List[Dict[str, float | str]]:
    df = load_tumor_volume_csv(csv_path)

    required_columns = {"FU", "tumor_volume"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns {required_columns}, got {set(df.columns)}"
        )

    followups = []
    for _, row in df.iterrows():
        timepoint = str(row["FU"])
        volume = row["tumor_volume"]

        if pd.isna(volume):
            raise ValueError(f"Missing tumor_volume for {timepoint} in {csv_path}")

        followups.append({
            "timepoint": timepoint,
            "volume": float(volume),
        })

    def sort_key(item: Dict[str, float | str]) -> int:
        tp = str(item["timepoint"]).upper()
        if tp.startswith("FU"):
            try:
                return int(tp.replace("FU", ""))
            except ValueError:
                return 999
        return 999

    followups = sorted(followups, key=sort_key)
    return followups


def run_longitudinal_pipeline_for_branch(
    patient_id: str,
    scenario: str,
    baseline_volume: float,
    baseline_voxel_count: int,
    followups: List[Dict[str, float | str]],
) -> List[Dict[str, float | str]]:
    agent = TreatmentAgent(initial_volume=baseline_volume)

    results = [
        {
            "patient_id": patient_id,
            "scenario": scenario,
            "timepoint": "Baseline",
            "volume_mm3": baseline_volume,
            "voxel_count": baseline_voxel_count,
            "status": "Baseline",
        }
    ]

    for item in followups:
        evaluation = agent.evaluate(item["volume"])

        results.append({
            "patient_id": patient_id,
            "scenario": scenario,
            "timepoint": item["timepoint"],
            "volume_mm3": item["volume"],
            "status": evaluation["status"],
            "percent_change_vs_baseline": evaluation["percent_change_vs_baseline"],
            "percent_change_vs_smallest": evaluation["percent_change_vs_smallest"],
            "smallest_volume_before_update": evaluation["smallest_volume_before_update"],
        })

    return results


def run_patient_pipeline(patient_dir: str | Path) -> pd.DataFrame:
    patient_dir = Path(patient_dir)
    patient_id = patient_dir.name

    baseline_seg_path = patient_dir / "baseline" / "seg.nii"
    progression_csv_path = patient_dir / "progression" / "tumor_volume.csv"
    remission_csv_path = patient_dir / "remission" / "tumor_volume.csv"

    if not baseline_seg_path.exists():
        raise FileNotFoundError(f"Missing baseline seg file: {baseline_seg_path}")

    baseline_info = compute_baseline_volume_from_mask(baseline_seg_path)
    baseline_volume_mm3 = baseline_info["volume_mm3"]
    baseline_voxel_count = baseline_info["voxel_count"]

    progression_followups = load_followup_volumes_from_patient_csv(progression_csv_path)
    remission_followups = load_followup_volumes_from_patient_csv(remission_csv_path)

    progression_results = run_longitudinal_pipeline_for_branch(
        patient_id=patient_id,
        scenario="progression",
        baseline_volume=baseline_volume_mm3,
        baseline_voxel_count=baseline_voxel_count,
        followups=progression_followups,
    )

    remission_results = run_longitudinal_pipeline_for_branch(
        patient_id=patient_id,
        scenario="remission",
        baseline_volume=baseline_volume_mm3,
        baseline_voxel_count=baseline_voxel_count,
        followups=remission_followups,
    )

    all_results = progression_results + remission_results
    return pd.DataFrame(all_results)


def print_results(df: pd.DataFrame) -> None:
    if df.empty:
        print("No results to display.")
        return

    for scenario, scenario_df in df.groupby("scenario"):
        print(f"\nScenario: {scenario}")
        print("-" * 60)

        for _, row in scenario_df.iterrows():
            if row["timepoint"] == "Baseline":
                print(
                    f"{row['timepoint']}: "
                    f"{row['volume_mm3']:.2f} mm^3 "
                    f"(voxel_count={row['voxel_count']}, status={row['status']})"
                )
            else:
                print(
                    f"{row['timepoint']}: "
                    f"{row['volume_mm3']:.2f} mm^3 -> {row['status']} "
                    f"(vs baseline: {row['percent_change_vs_baseline']:.2f}%, "
                    f"vs smallest: {row['percent_change_vs_smallest']:.2f}%)"
                )


if __name__ == "__main__":
    PATIENT_DIR = Path("data/raw/Mets_010")

    try:
        results_df = run_patient_pipeline(PATIENT_DIR)
        print(results_df)
        print_results(results_df)
    except Exception as e:
        print(f"Pipeline error: {e}")