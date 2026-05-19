"""
Multi-model comparison on a held-out test split.

Brings back the model zoo from the original `ml_model_prediction_test.py`
(Linear Regression, Random Forest, Gradient Boosting, MLP) so the thesis
can compare an exponential baseline, a Gompertz growth curve and several
ML models head-to-head on the same patients.

Differences from the original:

  * Features are **baseline-normalised** (FU/baseline ratios), so a model
    trained on the demo CSV (percent values) and one trained on the MRI
    CSV (mm^3) live in the same feature space. Predictions are returned
    in the patient's original units by multiplying back by `baseline`.
  * Models are scored on a **held-out test split** (default 20%) instead
    of the same data they were trained on, so reported MAEs are honest
    generalisation error — not the in-sample numbers `app.py` reports
    for the persisted continual-training Random Forest.
  * The MLP runs through a `Pipeline` with `StandardScaler` so the scaler
    fit happens only on the training half (no test-set leakage).

These models are evaluated only — they are *not* persisted, and they
do not interact with the continually-trained Random Forest in
`ml_predictor.py`. They exist for the comparison tab and for the report.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.loaders import get_trajectory
from src.models.exponential_model import fit_exponential, predict_exponential
from src.models.gompertz_model import fit_gompertz, predict_gompertz


FEATURE_COLS = ["FU1_rel", "FU2_rel"]
TARGET_COLS  = ["FU3_rel", "FU4_rel", "FU5_rel"]


# ---------------------------------------------------------------- model zoo

def _build_models() -> Dict[str, object]:
    """Return a fresh, untrained estimator for each ML model in the zoo."""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest (depth=6)": RandomForestRegressor(
            n_estimators=200, max_depth=6, random_state=42
        ),
        "Gradient Boosting": MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
            )
        ),
        "MLP (64,32)": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(64, 32), activation="relu",
                solver="adam", max_iter=2000, random_state=42,
            )),
        ]),
    }


# ---------------------------------------------------------------- sample build

def _build_relative_samples(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (subject, scenario) with all values normalised by baseline."""
    rows: List[dict] = []
    for _, r in df.iterrows():
        baseline = float(r["baseline"])
        if baseline <= 0:
            continue
        for scenario in ("progression", "remission"):
            rows.append({
                "subject":  r["subject"],
                "scenario": scenario,
                "baseline": baseline,
                "FU1_rel":  float(r[f"{scenario}_FU1"]) / baseline,
                "FU2_rel":  float(r[f"{scenario}_FU2"]) / baseline,
                "FU3_rel":  float(r[f"{scenario}_FU3"]) / baseline,
                "FU4_rel":  float(r[f"{scenario}_FU4"]) / baseline,
                "FU5_rel":  float(r[f"{scenario}_FU5"]) / baseline,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- growth-curve fits

def _per_row_growth_predictions(
    samples: pd.DataFrame,
    df: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """Fit an Exponential and a Gompertz curve per (subject, scenario), then
    return predicted FU3..FU5 in original units. NaN where the fit fails."""
    df_indexed = df.set_index("subject")
    n = len(samples)
    exp_preds = np.full((n, 3), np.nan)
    gom_preds = np.full((n, 3), np.nan)

    for i, (_, s) in enumerate(samples.iterrows()):
        baseline = float(df_indexed.loc[s["subject"], "baseline"])
        traj = get_trajectory(df_indexed.loc[s["subject"]], s["scenario"])
        observed = np.array(traj[:3], dtype=float)
        times = np.array([0, 1, 2])
        all_times = np.arange(6)

        try:
            k, log_v0 = fit_exponential(times, observed)
            full = predict_exponential(all_times, k, log_v0)
            exp_preds[i] = full[3:]
        except Exception:
            pass

        try:
            v_inf, b, c = fit_gompertz(times, observed)
            full = predict_gompertz(all_times, v_inf, b, c)
            gom_preds[i] = full[3:]
        except Exception:
            pass

    return {"Exponential": exp_preds, "Gompertz": gom_preds}


# ---------------------------------------------------------------- main API

def evaluate_models(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train Linear / RF / GB / MLP on a (1 - test_size) split of the cohort and
    score them on the held-out remainder, alongside per-patient Exponential
    and Gompertz fits.

    Returns
    -------
    summary : DataFrame  — one row per (model, target FU), MAE in original units.
    per_subject : DataFrame — long-form actual + predicted values for the test split.
    """
    samples = _build_relative_samples(df)
    if samples.empty:
        return pd.DataFrame(), pd.DataFrame()

    X_all = samples[FEATURE_COLS].to_numpy()
    Y_all = samples[TARGET_COLS].to_numpy()
    base_all = samples["baseline"].to_numpy().reshape(-1, 1)

    # Stratify by scenario so train and test each contain a 50/50 mix of
    # progression and remission rows (the cohort itself is 50/50 by
    # construction; this stops the random split from drifting unbalanced).
    indices = np.arange(len(samples))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=samples["scenario"].values,
    )
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    Y_train, Y_test = Y_all[train_idx], Y_all[test_idx]
    base_test = base_all[test_idx]
    test_samples = samples.iloc[test_idx].reset_index(drop=True)

    # ML models on the test split
    ml_predictions: Dict[str, np.ndarray] = {}
    for name, est in _build_models().items():
        est.fit(X_train, Y_train)
        rel_pred = est.predict(X_test)
        ml_predictions[name] = rel_pred * base_test  # back to original units

    # Per-row growth-curve fits (no train/test split — each row fits independently)
    growth_predictions = _per_row_growth_predictions(test_samples, df)

    # Actual values in original units
    actual = Y_test * base_test

    # ---------- summary ----------
    fu_labels = ["FU3", "FU4", "FU5"]
    summary_rows: List[dict] = []
    for name, pred in {**growth_predictions, **ml_predictions}.items():
        for j, fu in enumerate(fu_labels):
            err = np.abs(actual[:, j] - pred[:, j])
            summary_rows.append({
                "Model": name,
                "Timepoint": fu,
                "MAE": float(np.nanmean(err)),
                "n_test": int(np.sum(~np.isnan(err))),
            })
        all_err = np.abs(actual - pred)
        summary_rows.append({
            "Model": name,
            "Timepoint": "FU3-FU5 mean",
            "MAE": float(np.nanmean(all_err)),
            "n_test": int(np.sum(~np.isnan(all_err.mean(axis=1)))),
        })
    summary = pd.DataFrame(summary_rows)

    # ---------- per-subject ----------
    per_rows: List[dict] = []
    for i in range(len(test_samples)):
        rec = {
            "subject":  test_samples.loc[i, "subject"],
            "scenario": test_samples.loc[i, "scenario"],
            "baseline": float(base_test[i, 0]),
            "actual_FU3": float(actual[i, 0]),
            "actual_FU4": float(actual[i, 1]),
            "actual_FU5": float(actual[i, 2]),
        }
        for name, pred in {**growth_predictions, **ml_predictions}.items():
            for j, fu in enumerate(fu_labels):
                rec[f"{name}_{fu}"] = float(pred[i, j])
        per_rows.append(rec)
    per_subject = pd.DataFrame(per_rows)

    return summary, per_subject
