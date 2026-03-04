from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.agent.treatment_agent import TreatmentAgent
from src.io.nifiti_loader import load_mask
from src.metrics.volume import compute_tumor_volume


# Load baseline segmentation and compute tumor volume from the mask.
def compute_baseline_volume_from_mask(baseline_seg_path: str | Path) -> Dict[str, float]:
    mask_data, voxel_dims = load_mask(baseline_seg_path)
    return compute_tumor_volume(mask_data, voxel_dims)


# Load FU1-FU5 follow-up volumes for one subject and one scenario from CSV.
#
# Expected scenarios:
# - progression
# - remission

def load_followup_volumes_from_csv(
    csv_path: str | Path,
    subject: str,
    scenario: str,
) -> List[Dict[str, float | str]]:
    csv_path = Path(csv_path)

    if scenario not in {"progression", "remission"}:
        raise ValueError("scenario must be 'progression' or 'remission'")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "subject" not in df.columns:
        raise ValueError("CSV must contain column 'subject'")

    row = df[df["subject"] == subject]

    if row.empty:
        raise ValueError(f"Subject {subject} not found in CSV")

    row = row.iloc[0]
    followups = []

    for timepoint in ["FU1", "FU2", "FU3", "FU4", "FU5"]:
        col = f"{scenario}_{timepoint}"

        if col not in df.columns:
            raise ValueError(f"CSV missing column: {col}")

        volume = row[col]

        if pd.isna(volume):
            raise ValueError(f"Missing value for {subject} {col}")

        followups.append({
            "timepoint": timepoint,
            "volume": float(volume),
        })

    return followups


# Run the longitudinal workflow for one subject and one scenario.
#
# Workflow:
# - compute baseline volume from baseline/seg.nii
# - load FU1-FU5 volumes from CSV
# - evaluate each follow-up with the TreatmentAgent

def run_longitudinal_pipeline(
    baseline_seg_path: str | Path,
    csv_path: str | Path,
    subject: str,
    scenario: str,
) -> List[Dict[str, float | str]]:
    baseline_info = compute_baseline_volume_from_mask(baseline_seg_path)
    baseline_volume_mm3 = baseline_info["volume_mm3"]

    followups = load_followup_volumes_from_csv(csv_path, subject, scenario)
    agent = TreatmentAgent(initial_volume=baseline_volume_mm3)

    results = [
        {
            "subject": subject,
            "scenario": scenario,
            "timepoint": "Baseline",
            "volume_mm3": baseline_volume_mm3,
            "voxel_count": baseline_info["voxel_count"],
            "status": "Baseline",
        }
    ]

    for item in followups:
        evaluation = agent.evaluate(item["volume"])

        results.append({
            "subject": subject,
            "scenario": scenario,
            "timepoint": item["timepoint"],
            "volume_mm3": item["volume"],
            "status": evaluation["status"],
            "percent_change_vs_baseline": evaluation["percent_change_vs_baseline"],
            "percent_change_vs_smallest": evaluation["percent_change_vs_smallest"],
            "smallest_volume_before_update": evaluation["smallest_volume_before_update"],
        })

    return results


# Print one subject/scenario trajectory in a readable format.
def print_results(results: List[Dict[str, float | str]]) -> None:
    if not results:
        print("No results to display.")
        return

    subject = results[0]["subject"]
    scenario = results[0]["scenario"]

    print(f"\nSubject: {subject}")
    print(f"Scenario: {scenario}")
    print("-" * 50)

    for item in results:
        if item["timepoint"] == "Baseline":
            print(
                f"{item['timepoint']}: "
                f"{item['volume_mm3']:.2f} mm^3 "
                f"(voxel_count={item['voxel_count']}, status={item['status']})"
            )
        else:
            print(
                f"{item['timepoint']}: "
                f"{item['volume_mm3']:.2f} mm^3 -> {item['status']} "
                f"(vs baseline: {item['percent_change_vs_baseline']:.2f}%, "
                f"vs smallest: {item['percent_change_vs_smallest']:.2f}%)"
            )


if __name__ == "__main__":

    # Update this to the folder that contains:
    # - Mets_XXX subject folders
    # - tumor_volumes_all_subjects_v3.csv
    #
    # 
    # DATASET_ROOT = Path("/Users/phillipovera/Downloads/series")

    DATASET_ROOT = Path("/path/to/your/series")
    

    # Change to the case you want to check
    subject = "Mets_005"
    scenario = "progression"  # or "remission"

    baseline_seg_path = DATASET_ROOT / subject / "baseline" / "seg.nii"
    csv_path = DATASET_ROOT / "tumor_volumes_all_subjects_v3.csv"

    try:
        results = run_longitudinal_pipeline(
            baseline_seg_path=baseline_seg_path,
            csv_path=csv_path,
            subject=subject,
            scenario=scenario,
        )
        print_results(results)
    except Exception as e:
        print(f"Pipeline error: {e}")