"""
Single-patient inference: take one folder shaped like

    <patient>/
        baseline/seg.nii
        progression/FU{1..N}/seg.nii   (optional)
        remission/FU{1..N}/seg.nii     (optional)

read whatever timepoints exist, compute their tumor volumes, and forecast
the future timepoints using the persisted continual-training Random Forest
+ Exponential + Gompertz. Bypasses the cohort-CSV build step entirely.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.io.nifti_loader import load_mask
from src.metrics.volume import compute_tumor_volume


SCENARIOS = ("progression", "remission")


@dataclass
class ScanVolumes:
    """Volumes for one scenario branch of a single patient."""
    scenario: str
    baseline: float
    timepoints: Dict[str, float] = field(default_factory=dict)  # "FU1" -> mm^3

    @property
    def ordered_followups(self) -> List[Tuple[str, float]]:
        return sorted(
            self.timepoints.items(),
            key=lambda kv: int(kv[0][2:]) if kv[0].startswith("FU") and kv[0][2:].isdigit() else 999,
        )

    @property
    def trajectory(self) -> List[float]:
        """[baseline, FU1, FU2, ...] with whichever follow-ups exist, in order."""
        return [self.baseline] + [v for _, v in self.ordered_followups]


def _vol_mm3(seg: Path) -> float:
    mask, dims = load_mask(seg)
    return compute_tumor_volume(mask, dims)["volume_mm3"]


def discover_scenarios(patient_dir: Path) -> List[str]:
    return [s for s in SCENARIOS if (patient_dir / s).is_dir()]


def discover_followups(scenario_dir: Path) -> List[Path]:
    return sorted(
        (d for d in scenario_dir.iterdir()
         if d.is_dir() and re.fullmatch(r"FU\d+", d.name)),
        key=lambda d: int(d.name[2:]),
    )


def read_patient_volumes(
    patient_dir: str | Path,
    scenario: str,
) -> ScanVolumes:
    """Read baseline + every FU{N}/seg.nii present for one scenario branch."""
    patient_dir = Path(patient_dir)
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}")

    baseline_seg = patient_dir / "baseline" / "seg.nii"
    if not baseline_seg.exists():
        raise FileNotFoundError(f"Missing baseline segmentation: {baseline_seg}")

    scenario_dir = patient_dir / scenario
    if not scenario_dir.is_dir():
        raise FileNotFoundError(f"Missing scenario folder: {scenario_dir}")

    fu_dirs = discover_followups(scenario_dir)
    if not fu_dirs:
        raise FileNotFoundError(f"No FU* folders under {scenario_dir}")

    volumes: Dict[str, float] = {}
    for fu in fu_dirs:
        seg = fu / "seg.nii"
        if not seg.exists():
            continue
        volumes[fu.name] = _vol_mm3(seg)

    return ScanVolumes(
        scenario=scenario,
        baseline=_vol_mm3(baseline_seg),
        timepoints=volumes,
    )


def forecast_for_patient(
    scan: ScanVolumes,
    ml_model,
) -> Dict[str, object]:
    """
    Forecast FU3, FU4, FU5 for a single patient given their first three
    observed timepoints (baseline + FU1 + FU2). Combines Exponential,
    Gompertz and the persisted continual-training Random Forest.
    """
    from src.models.cohort_eval import forecast_all_models
    from src.models.ml_predictor import predict_future_with_uncertainty

    traj = scan.trajectory
    if len(traj) < 3:
        raise ValueError(
            f"Need at least baseline + FU1 + FU2 to forecast — got {len(traj)} timepoint(s)"
        )

    observed = traj[:3]
    actual_future: Optional[np.ndarray] = (
        np.array(traj[3:], dtype=float) if len(traj) > 3 else None
    )

    forecasts = forecast_all_models(observed)

    ml_mean, ml_std = predict_future_with_uncertainty(
        ml_model, baseline=observed[0], fu1=observed[1], fu2=observed[2],
    )

    return {
        "observed": observed,
        "actual_future": actual_future,
        "forecasts": forecasts,
        "ml_mean": ml_mean,
        "ml_std": ml_std,
    }
