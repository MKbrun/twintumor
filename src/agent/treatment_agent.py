"""
Rule-based RANO response agent (Rule-Based-Agent.pdf).

Step 1 (volume extraction) lives in `src.metrics.volume`. SPD-style diameter
extraction is in `src.metrics.diameter` (added separately for completeness;
not used by the rules below because the pseudocode in the PDF works on
volume).

Step 2 — Response classification (PDF Step 2 pseudocode):
    - current == 0                      -> Complete Remission
    - pct_change_vs_baseline <= -50     -> Partial Remission
    - pct_change_vs_smallest >= +25     -> Progression
    - otherwise                         -> Stable Disease

Step 3 — Pseudoprogression handling (opt-in via `enable_pseudoprogression`):
    Follows the PDF pseudocode literally (flag-based, no time window):

        - The first time pct_vs_smallest >= 25, set is_flagged_for_growth
          and emit "Provisional Progression" (the PDF's "Possible PsP").
        - On the next scan:
            * if current >= last_volume        -> Confirmed Progression,
                                                  retroactively confirm the
                                                  flagged scan as PD.
            * if current  < last_volume        -> Pseudoprogression Resolved,
                                                  retroactively reclassify the
                                                  flagged scan as Pseudoprogression.
        - After resolution the flag is cleared and a future first-time growth
          event can flag again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Status constants
STATUS_BASELINE = "Baseline"
STATUS_COMPLETE_REMISSION = "Complete Remission"
STATUS_PARTIAL_REMISSION = "Partial Remission"
STATUS_STABLE = "Stable Disease"
STATUS_PROGRESSION = "Progression"
STATUS_PROVISIONAL_PROGRESSION = "Provisional Progression"
STATUS_PSEUDOPROGRESSION = "Pseudoprogression"


@dataclass
class TreatmentAgent:
    initial_volume: float
    enable_pseudoprogression: bool = False

    baseline_volume: float = field(init=False)
    smallest_volume: float = field(init=False)
    last_volume: float = field(init=False)
    is_flagged_for_growth: bool = field(init=False, default=False)
    history: List[float] = field(init=False)
    statuses: List[str] = field(init=False)
    _provisional_idx: Optional[int] = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.initial_volume <= 0:
            raise ValueError("initial_volume must be > 0")
        self.baseline_volume = float(self.initial_volume)
        self.smallest_volume = float(self.initial_volume)
        self.last_volume = float(self.initial_volume)
        self.history = [float(self.initial_volume)]
        self.statuses = [STATUS_BASELINE]
        self.is_flagged_for_growth = False
        self._provisional_idx = None

    # --------------------------------------------------------- Step 2 rules

    def _classify_step2(self, current: float, smallest_before: float) -> str:
        """PDF Step 2 pseudocode (volume-based)."""
        if current == 0:
            return STATUS_COMPLETE_REMISSION
        pct_vs_baseline = ((current - self.baseline_volume) / self.baseline_volume) * 100.0
        pct_vs_smallest = ((current - smallest_before) / smallest_before) * 100.0
        if pct_vs_baseline <= -50.0:
            return STATUS_PARTIAL_REMISSION
        if pct_vs_smallest >= 25.0:
            return STATUS_PROGRESSION
        return STATUS_STABLE

    # ---------------------------------------------------- evaluate one scan

    def evaluate(self, current_volume: float) -> Dict[str, float | str]:
        current_volume = float(current_volume)
        if current_volume < 0:
            raise ValueError("current_volume cannot be negative")

        smallest_before_update = self.smallest_volume
        last_before_update = self.last_volume

        pct_vs_baseline = ((current_volume - self.baseline_volume) / self.baseline_volume) * 100.0
        pct_vs_smallest = ((current_volume - smallest_before_update) / smallest_before_update) * 100.0

        raw_status = self._classify_step2(current_volume, smallest_before_update)

        if not self.enable_pseudoprogression:
            status = raw_status
        else:
            status = self._step3_resolve(current_volume, last_before_update, raw_status)

        # Update bookkeeping AFTER classification
        if current_volume < self.smallest_volume:
            self.smallest_volume = current_volume
        self.last_volume = current_volume
        self.history.append(current_volume)
        self.statuses.append(status)

        return {
            "current_volume": current_volume,
            "baseline_volume": self.baseline_volume,
            "smallest_volume_before_update": smallest_before_update,
            "percent_change_vs_baseline": pct_vs_baseline,
            "percent_change_vs_smallest": pct_vs_smallest,
            "raw_status": raw_status,
            "status": status,
            "is_flagged_for_growth": self.is_flagged_for_growth,
        }

    # ------------------------------------------------------ Step 3 helpers

    def _step3_resolve(
        self, current: float, last_before_update: float, raw_status: str
    ) -> str:
        """Apply PDF Step 3 (flag-based pseudoprogression)."""

        # Already flagged from a previous scan — resolve it now.
        if self.is_flagged_for_growth:
            if current >= last_before_update:
                # Growth confirmed: retroactively confirm the provisional scan.
                if self._provisional_idx is not None:
                    self.statuses[self._provisional_idx] = STATUS_PROGRESSION
                self.is_flagged_for_growth = False
                self._provisional_idx = None
                return STATUS_PROGRESSION
            else:
                # Tumor shrank from last → previous "growth" was pseudoprogression.
                if self._provisional_idx is not None:
                    self.statuses[self._provisional_idx] = STATUS_PSEUDOPROGRESSION
                self.is_flagged_for_growth = False
                self._provisional_idx = None
                # Now re-evaluate the *current* scan with Step 2 rules.
                return raw_status

        # Not yet flagged: any first ≥ +25% growth event is provisional.
        if raw_status == STATUS_PROGRESSION:
            self.is_flagged_for_growth = True
            self._provisional_idx = len(self.statuses)  # index this scan will occupy
            return STATUS_PROVISIONAL_PROGRESSION

        return raw_status

    # ------------------------------------------------------------- helpers

    def final_statuses(self) -> List[str]:
        """Per-timepoint statuses after pseudoprogression re-labelling."""
        return list(self.statuses)
