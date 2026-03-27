from pathlib import Path
import pandas as pd

from src.pipelines.longitudinal_run import run_patient_pipeline


def discover_patient_dirs(data_root: str | Path) -> list[Path]:
    data_root = Path(data_root)

    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    patient_dirs = [
        p for p in data_root.iterdir()
        if p.is_dir() and p.name.startswith("Mets_")
    ]

    return sorted(patient_dirs)


def build_dataset(data_root: str | Path) -> pd.DataFrame:
    patient_dirs = discover_patient_dirs(data_root)

    all_dfs = []

    for patient_dir in patient_dirs:
        try:
            patient_df = run_patient_pipeline(patient_dir)
            all_dfs.append(patient_df)
            print(f"[OK] Processed {patient_dir.name}")
        except Exception as e:
            print(f"[WARN] Skipped {patient_dir.name}: {e}")

    if not all_dfs:
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df


if __name__ == "__main__":
    data_root = Path("data/raw")
    output_path = Path("data/processed/all_patient_trajectories.csv")

    df = build_dataset(data_root)

    if df.empty:
        print("No patient data processed.")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved dataset to: {output_path}")
        print(df.head())