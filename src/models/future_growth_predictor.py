from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor


CSV_PATH = Path("data/analysis/consistent_tumor_analysis.csv")


def load_data(csv_path: str | Path = CSV_PATH) -> pd.DataFrame:
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    return pd.read_csv(csv_path)


def train_future_predictor(csv_path: str | Path = CSV_PATH):
    df = load_data(csv_path)

    samples = []

    for _, row in df.iterrows():
        baseline = float(row["baseline_percent"])

        for scenario in ["progression", "remission"]:
            samples.append({
                "subject": row["subject"],
                "scenario": scenario,
                "baseline": baseline,
                "FU1": float(row[f"{scenario}_FU1"]),
                "FU2": float(row[f"{scenario}_FU2"]),
                "target_FU3": float(row[f"{scenario}_FU3"]),
                "target_FU4": float(row[f"{scenario}_FU4"]),
                "target_FU5": float(row[f"{scenario}_FU5"]),
            })

    samples_df = pd.DataFrame(samples)

    X = samples_df[["baseline", "FU1", "FU2"]]
    y = samples_df[["target_FU3", "target_FU4", "target_FU5"]]

    model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )
    )
    model.fit(X, y)

    return model, samples_df


def predict_future_for_case(model, baseline: float, fu1: float, fu2: float) -> dict:
    X_new = pd.DataFrame([{
        "baseline": baseline,
        "FU1": fu1,
        "FU2": fu2,
    }])

    pred = model.predict(X_new)[0]

    return {
        "FU3": float(pred[0]),
        "FU4": float(pred[1]),
        "FU5": float(pred[2]),
    }