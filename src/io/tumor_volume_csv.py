from pathlib import Path
import pandas as pd


def load_tumor_volume_csv(path: str | Path) -> pd.DataFrame:
    """
    Load a tumor_volume.csv file and return it as a DataFrame.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"tumor_volume.csv not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"tumor_volume.csv is empty: {path}")

    return df