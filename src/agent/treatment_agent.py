from dataclasses import dataclass, field
from typing import Dict, List


# Rule-based longitudinal response classifier.
#
# Rules:
# - volume == 0 -> Complete Remission
# - <= -50% vs baseline -> Partial Remission
# - >= +25% vs smallest previous volume -> Progression
# - otherwise -> Stable Disease

@dataclass
class TreatmentAgent:
    initial_volume: float
    baseline_volume: float = field(init=False)
    smallest_volume: float = field(init=False)
    history: List[float] = field(init=False)

    def __post_init__(self) -> None:

        # Baseline must be positive for percentage calculations
        if self.initial_volume <= 0:
            raise ValueError("initial_volume must be > 0")

        # Store baseline and initialize the smallest observed volume
        self.baseline_volume = float(self.initial_volume)
        self.smallest_volume = float(self.initial_volume)

        # Keep full volume history
        self.history = [float(self.initial_volume)]

    def evaluate(self, current_volume: float) -> Dict[str, float | str]:
        
        # Convert to float for consistent calculations
        current_volume = float(current_volume)

        if current_volume < 0:
            raise ValueError("current_volume cannot be negative")

        # Compare against the smallest volume seen before this scan
        smallest_before_update = self.smallest_volume

        # Percentage change relative to baseline
        percent_change_vs_baseline = (
            (current_volume - self.baseline_volume) / self.baseline_volume
        ) * 100.0

        # Percentage change relative to the smallest previous volume
        percent_change_vs_smallest = (
            (current_volume - smallest_before_update) / smallest_before_update
        ) * 100.0

        # Apply rule-based response criteria
        if current_volume == 0:
            status = "Complete Remission"
        elif percent_change_vs_baseline <= -50.0:
            status = "Partial Remission"
        elif percent_change_vs_smallest >= 25.0:
            status = "Progression"
        else:
            status = "Stable Disease"

        # Update smallest observed volume after classification
        if current_volume < self.smallest_volume:
            self.smallest_volume = current_volume

        # Save current volume in history
        self.history.append(current_volume)

        return {
            "current_volume": current_volume,
            "baseline_volume": self.baseline_volume,
            "smallest_volume_before_update": smallest_before_update,
            "percent_change_vs_baseline": percent_change_vs_baseline,
            "percent_change_vs_smallest": percent_change_vs_smallest,
            "status": status,
        }