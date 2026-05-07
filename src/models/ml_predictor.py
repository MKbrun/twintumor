"""
Continually-trained Random-Forest future-volume predictor.

Design
------
The framework keeps **one** Random Forest at `data/models/rf_predictor.joblib`
that is grown across many training rounds. Every round adds N new trees
(via sklearn's `warm_start=True`) trained on that round's data only. Old
trees from earlier rounds are kept untouched, so:

    - Past knowledge persists without retaining past patient rows.
    - A new training round only needs the data that round contributes.
    - Cloning the repo gives a user the model with everything it has
      learned so far; they can then train more on their own data without
      ever seeing earlier users' raw data.

Unit-agnostic features
----------------------
The demo CSV is in "percent of MRI signal" while MRI-derived CSVs are in
mm^3. To make a single growing model coherent across both, the predictor
trains on baseline-normalised ratios:

    features = (FU1/baseline, FU2/baseline)
    targets  = (FU3/baseline, FU4/baseline, FU5/baseline)

At inference the predicted ratios are multiplied by the patient's own
baseline to return values in whichever unit that patient was given.

API
---
    get_or_train(model_path, demo_csv) -> (model, log, was_trained)
        Load the persisted model. If it does not exist, train a fresh one
        from the bundled demo CSV.

    train_incremental(model_path, csv_path, source_label, n_new_trees)
        Load the existing model and add `n_new_trees` trees fitted on the
        new CSV. If no model exists yet, train an initial one first.

    reset_model(model_path)
        Delete the persisted model entirely (next call to get_or_train
        will retrain from the demo CSV).

    predict_future(model, baseline, fu1, fu2) -> dict[FU3, FU4, FU5]
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.data.loaders import load_dataset
from src.data.paths import DEFAULT_ML_MODEL_PATH, DEMO_DATASET_CSV


# Bump when the persisted bundle layout changes so old pickles auto-rebuild.
MODEL_SCHEMA_VERSION = 3

FEATURE_COLS = ["FU1_rel", "FU2_rel"]
TARGET_COLS  = ["FU3_rel", "FU4_rel", "FU5_rel"]

INITIAL_TREES = 200
DEFAULT_INCREMENTAL_TREES = 100


# ----------------------------------------------------------------- training log

@dataclass
class TrainingRound:
    timestamp: str
    source_label: str
    n_samples: int
    trees_before: int
    trees_after: int


# ----------------------------------------------------------------- preprocessing

def _build_relative_samples(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (subject, scenario) with all values normalised by baseline."""
    rows: list[dict] = []
    for _, r in df.iterrows():
        baseline = float(r["baseline"])
        if baseline <= 0:
            continue
        for scenario in ("progression", "remission"):
            try:
                fu1 = float(r[f"{scenario}_FU1"]) / baseline
                fu2 = float(r[f"{scenario}_FU2"]) / baseline
                t3  = float(r[f"{scenario}_FU3"]) / baseline
                t4  = float(r[f"{scenario}_FU4"]) / baseline
                t5  = float(r[f"{scenario}_FU5"]) / baseline
            except (KeyError, ValueError):
                continue
            rows.append({
                "subject": r["subject"], "scenario": scenario,
                "baseline": baseline,
                "FU1_rel": fu1, "FU2_rel": fu2,
                "FU3_rel": t3,  "FU4_rel": t4,  "FU5_rel": t5,
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- bundle I/O

def _save_bundle(model: RandomForestRegressor,
                 training_log: list[TrainingRound],
                 model_path: Path,
                 lopo_mae: Optional[float] = None) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "schema_version": MODEL_SCHEMA_VERSION,
        "feature_cols": FEATURE_COLS,
        "target_cols":  TARGET_COLS,
        "model": model,
        "training_log": [vars(r) for r in training_log],
        "lopo_mae": lopo_mae,
    }, model_path)


def _compute_rf_lopo_mae(samples: pd.DataFrame, n_estimators: int = 100,
                         random_state: int = 42) -> float:
    """
    Quick patient-level LOPO-CV on a fresh Random Forest of the same family
    we ship for live prediction. Returns the cohort-mean MAE in original
    (per-patient baseline-multiplied) units. Cached in the model bundle so
    the UI can show "expected MAE on a new patient" without recomputing.
    """
    from sklearn.ensemble import RandomForestRegressor
    if samples.empty or "subject" not in samples.columns:
        return float("nan")
    subjects = samples["subject"].unique()
    if len(subjects) < 3:
        return float("nan")
    errors: list[float] = []
    for held_out in subjects:
        train = samples[samples["subject"] != held_out]
        test  = samples[samples["subject"] == held_out]
        if train.empty or test.empty:
            continue
        m = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        m.fit(train[FEATURE_COLS].to_numpy(), train[TARGET_COLS].to_numpy())
        pred_rel = m.predict(test[FEATURE_COLS].to_numpy())
        # Convert relative predictions back to original units
        if "baseline" in test.columns:
            base = test["baseline"].to_numpy().reshape(-1, 1)
            actual_abs = test[TARGET_COLS].to_numpy() * base
            pred_abs   = pred_rel * base
        else:
            actual_abs = test[TARGET_COLS].to_numpy()
            pred_abs   = pred_rel
        errors.append(float(np.abs(actual_abs - pred_abs).mean()))
    return float(np.mean(errors)) if errors else float("nan")


def _load_bundle(model_path: Path) -> Optional[dict]:
    if not model_path.exists():
        return None
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or bundle.get("schema_version") != MODEL_SCHEMA_VERSION:
        return None
    return bundle


# ----------------------------------------------------------------- core

def _new_forest(n_estimators: int = INITIAL_TREES, random_state: int = 42) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        warm_start=True,
        random_state=random_state,
    )


def _fit_round(
    model: RandomForestRegressor,
    samples: pd.DataFrame,
    trees_to_add: int,
) -> int:
    """Fit `trees_to_add` more trees on `samples` and return the new total tree count."""
    if samples.empty:
        return len(model.estimators_) if hasattr(model, "estimators_") else 0

    X = samples[FEATURE_COLS].to_numpy()
    Y = samples[TARGET_COLS].to_numpy()

    if not hasattr(model, "estimators_") or not model.estimators_:
        # First-ever fit — n_estimators already set; just fit.
        model.set_params(n_estimators=trees_to_add)
    else:
        model.set_params(n_estimators=len(model.estimators_) + trees_to_add)
    model.fit(X, Y)
    return len(model.estimators_)


def _now_iso() -> str:
    return datetime.datetime.now().replace(microsecond=0).isoformat()


def train_initial_from_demo(
    model_path: str | Path = DEFAULT_ML_MODEL_PATH,
    demo_csv: str | Path = DEMO_DATASET_CSV,
    n_initial_trees: int = INITIAL_TREES,
) -> Tuple[RandomForestRegressor, List[TrainingRound]]:
    """Create a brand-new model trained on the bundled demo CSV."""
    return train_fresh_from(
        csv_path=demo_csv,
        source_label=f"Bundled demo CSV",
        model_path=model_path,
        n_initial_trees=n_initial_trees,
    )


def train_fresh_from(
    csv_path: str | Path,
    source_label: str,
    model_path: str | Path = DEFAULT_ML_MODEL_PATH,
    n_initial_trees: int = INITIAL_TREES,
    compute_lopo: bool = True,
) -> Tuple[RandomForestRegressor, List[TrainingRound]]:
    """
    Reset and train a brand-new Random Forest using ONLY the given CSV.

    If `compute_lopo` is True, also runs a patient-level LOPO-CV pass with
    the same model family and caches the cohort-mean MAE in the bundle so
    the UI can show "expected MAE on a new patient" without re-running it.
    """
    df = load_dataset(csv_path)
    samples = _build_relative_samples(df)

    model = _new_forest(n_estimators=n_initial_trees)
    trees_after = _fit_round(model, samples, trees_to_add=n_initial_trees)

    lopo_mae = _compute_rf_lopo_mae(samples) if compute_lopo else None

    log = [TrainingRound(
        timestamp=_now_iso(),
        source_label=f"{source_label} ({len(df)} patients)",
        n_samples=len(samples),
        trees_before=0,
        trees_after=trees_after,
    )]
    _save_bundle(model, log, Path(model_path), lopo_mae=lopo_mae)
    return model, log


def train_incremental(
    csv_path: str | Path,
    source_label: str,
    model_path: str | Path = DEFAULT_ML_MODEL_PATH,
    demo_csv: str | Path = DEMO_DATASET_CSV,
    n_new_trees: int = DEFAULT_INCREMENTAL_TREES,
) -> Tuple[RandomForestRegressor, List[TrainingRound]]:
    """
    Add `n_new_trees` trees fitted on `csv_path` to the persisted model.
    If no persisted model exists, a fresh demo-trained one is created first.
    """
    model_path = Path(model_path)

    bundle = _load_bundle(model_path)
    if bundle is None:
        train_initial_from_demo(model_path, demo_csv)
        bundle = _load_bundle(model_path)
    assert bundle is not None
    model: RandomForestRegressor = bundle["model"]
    training_log = [TrainingRound(**r) for r in bundle["training_log"]]

    df = load_dataset(csv_path)
    samples = _build_relative_samples(df)
    if samples.empty:
        raise ValueError(f"No usable training rows derived from {csv_path}")

    trees_before = len(model.estimators_)
    trees_after = _fit_round(model, samples, trees_to_add=n_new_trees)

    training_log.append(TrainingRound(
        timestamp=_now_iso(),
        source_label=source_label,
        n_samples=len(samples),
        trees_before=trees_before,
        trees_after=trees_after,
    ))
    # Recompute LOPO-MAE so the cached benchmark reflects what the model now does.
    lopo_mae = _compute_rf_lopo_mae(samples)
    _save_bundle(model, training_log, model_path, lopo_mae=lopo_mae)
    return model, training_log


def get_or_train(
    model_path: str | Path = DEFAULT_ML_MODEL_PATH,
    demo_csv: str | Path = DEMO_DATASET_CSV,
) -> Tuple[RandomForestRegressor, List[TrainingRound], bool]:
    """Load the persisted model; train an initial demo model if missing."""
    model_path = Path(model_path)
    bundle = _load_bundle(model_path)
    if bundle is not None:
        model = bundle["model"]
        log = [TrainingRound(**r) for r in bundle["training_log"]]
        return model, log, False
    model, log = train_initial_from_demo(model_path, demo_csv)
    return model, log, True


def get_cached_lopo_mae(model_path: str | Path = DEFAULT_ML_MODEL_PATH) -> Optional[float]:
    """Return the LOPO-CV MAE cached at training time, or None if absent."""
    bundle = _load_bundle(Path(model_path))
    if bundle is None:
        return None
    val = bundle.get("lopo_mae")
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return float(val)


def reset_model(model_path: str | Path = DEFAULT_ML_MODEL_PATH) -> bool:
    """Delete the persisted model file. Returns True if a file was removed."""
    model_path = Path(model_path)
    if model_path.exists():
        model_path.unlink()
        return True
    return False


# ----------------------------------------------------------------- inference

def predict_future(model: RandomForestRegressor, baseline: float, fu1: float, fu2: float) -> dict:
    """Predict FU3..FU5 in the patient's own units (multiplies by `baseline`)."""
    if baseline <= 0:
        raise ValueError("baseline must be > 0")
    X = np.array([[fu1 / baseline, fu2 / baseline]])
    rel = model.predict(X)[0]  # shape (3,)
    return {
        "FU3": float(rel[0] * baseline),
        "FU4": float(rel[1] * baseline),
        "FU5": float(rel[2] * baseline),
    }


def predict_future_with_uncertainty(
    model: RandomForestRegressor, baseline: float, fu1: float, fu2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-tree mean and standard deviation across the forest, in original units."""
    if baseline <= 0:
        raise ValueError("baseline must be > 0")
    X = np.array([[fu1 / baseline, fu2 / baseline]])
    # Each estimator's predict returns shape (1, 3) for 3-target output
    tree_preds = np.array([t.predict(X)[0] for t in model.estimators_])  # (n_trees, 3)
    mean = tree_preds.mean(axis=0) * baseline
    std  = tree_preds.std(axis=0)  * baseline
    return mean, std
