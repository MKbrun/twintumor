"""Single source of truth for filesystem paths used across the project."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
DEMO_DIR = DATA_DIR / "demo"

# Canonical volume dataset built by src/pipelines/build_volumes_csv.py.
# Wide-format: subject, baseline, progression_FU1..FU5, remission_FU1..FU5
# Values are tumor volume in mm^3 computed from each timepoint's seg.nii.
TUMOR_VOLUMES_CSV = PROCESSED_DIR / "tumor_volumes.csv"

# Bundled demo dataset (105 patients of synthetic divergent trajectories).
# Used as the default training source so the demo always works on a fresh
# clone, and as a fallback when the user has too few real MRI patients to
# train a meaningful model from their own data.
DEMO_DATASET_CSV = DEMO_DIR / "consistent_tumor_analysis.csv"

# Persisted Random-Forest predictor produced by src/models/ml_predictor.py.
DEFAULT_ML_MODEL_PATH = MODELS_DIR / "rf_predictor.joblib"
