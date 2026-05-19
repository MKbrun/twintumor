# TwinTumor

Digital-twin framework for brain-metastasis volume trajectories from longitudinal MRI.

The framework reads segmentation masks (`seg.nii`) at each timepoint, computes
real tumor volumes in %, runs three forecasting models on the first three
timepoints (Baseline, FU1, FU2) to predict the next three (FU3, FU4, FU5),
and classifies each timepoint with a RANO-style rule agent.

## Expected folder layout

Two layouts are supported and auto-detected:

### AIMI layout (recommended — `E:/mri_series/series`)

```
<data-root>/
    Mets_005/
        baseline/{seg, t1_gd, t1_pre, flair}.nii
        progression/FU{1..5}/{t1_gd, t1_pre, flair}.nii   (no seg)
        remission/FU{1..5}/{t1_gd, t1_pre, flair}.nii      (no seg)
    Mets_010/
        ...
```

- `baseline/seg.nii` defines the tumor ROI.
- For every timepoint, the framework reads `t1_gd.nii` and applies Otsu
  thresholding inside the baseline ROI. The percentage of in-ROI voxels
  above threshold is the per-timepoint signal value.
- Two extractor modes (selectable via `mode=` in
  `extract_patient_row`):
  - **`baseline_anchored`** *(default)* — Otsu is computed once on the
    baseline's in-ROI intensities. At every FU the in-ROI intensities are
    rescaled so their mean equals baseline's mean, then the *baseline*
    threshold is applied. Removes per-FU intensity drift; trajectory shape
    becomes biologically interpretable.
  - **`per_timepoint`** — independent Otsu at every FU. Most literal
    reading of "Otsu within the segmentation ROI"; sensitive to scanner
    intensity drift.
- Either mode produces the same column schema as `consistent_tumor_analysis.csv`,
  but absolute values may differ from AIMI's reference output (their analysis
  script is not available to us). On a comparison of trajectory ratios,
  `baseline_anchored` reduces per-patient ratio MAE vs AIMI by ~25–80 %
  relative to `per_timepoint`. Documented limitation; published cohort
  results in the report use AIMI's reference CSV unmodified.

### Legacy layout (`E:/series`)

```
<data-root>/
    Mets_005/
        baseline/seg.nii
        progression/FU{1..5}/seg.nii
        remission/FU{1..5}/seg.nii
```

- Volumes are computed by counting voxels in each `seg.nii` × voxel volume.
- For the AIMI dataset, the per-FU `progression/seg.nii` files are
  byte-identical placeholders of `baseline/seg.nii` — so the framework
  surfaces a *Data quality* warning when it sees a flat trajectory and
  recommends switching to the AIMI layout.

### Dual-source design

| Use case | Source |
|---|---|
| Published cohort results in the report | Bundled `consistent_tumor_analysis.csv` (AIMI's authoritative output, unmodified) |
| Predicting on a new patient added to a folder | Framework's own Otsu-within-ROI extractor on the AIMI layout |
| Visualising real per-FU signal masks | AIMI layout's `t1_gd.nii` files — the *Predicted vs actual* heatmap mode loads them directly |

This way the numerical results in the thesis are AIMI's, and the
"framework can ingest new MRI data" claim is fulfilled by our own
in-framework extractor.

## Setup

On Windows:
```
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
On MacOS: 
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## Run the demo

```
streamlit run app.py
```

In the sidebar:

1. Set **Folder of Mets_* patient series** (use **Browse…** for a native folder picker) — for example `E:/series` or `data/raw`.
2. Click **Build / rebuild volume CSV from MRI**. The framework walks every patient, reads each `seg.nii`, and writes `data/processed/tumor_volumes.csv`.
3. Pick a patient, scenario, and toggle RANO pseudoprogression handling.

The Random-Forest predictor is **persisted to disk** at
`data/models/rf_predictor.joblib` after first training, so a clone-and-run
user does not retrain on every launch. Use **Retrain ML model** in the
sidebar to rebuild it on the current cohort (e.g. after adding new patients).

## Run from the command line

Build the volume dataset:

```
python -m src.pipelines.build_volumes_csv --data-root E:/series
```

Run the RANO agent on every patient (long-form CSV with per-timepoint statuses):

```
python -m src.pipelines.build_dataset --data-root E:/series --pseudoprogression
```

Inspect a single patient:

```
python -m src.pipelines.longitudinal_run --patient-dir E:/series/Mets_005 --pseudoprogression
```

## Tests

```
pytest
```

## Project layout

```
app.py                              Streamlit demo
src/
    agent/treatment_agent.py        RANO agent (Steps 1+2+3 with pseudoprogression)
    data/                           paths + dataset loaders
    io/nifti_loader.py              load .nii masks
    metrics/volume.py               voxel -> mm^3 / cm^3
    models/
        exponential_model.py        log-linear fit
        gompertz_model.py           saturating-growth fit
        ml_predictor.py             persisted Random-Forest predictor
        cohort_eval.py              run all models across the cohort
    pipelines/
        build_volumes_csv.py        wide CSV (volumes only) for ML / forecasting
        build_dataset.py            long CSV with RANO statuses per timepoint
        longitudinal_run.py         single-patient pipeline
data/
    raw/                            user-provided Mets_* folders
    processed/tumor_volumes.csv     built by build_volumes_csv
    models/rf_predictor.joblib      persisted ML model
tests/
```
