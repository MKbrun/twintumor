"""
Rule-based tumor response classifier using longitudinal tumor volumes.

Should implement:
- class TreatmentAgent
    - __init__(initial_volume)
    - evaluate(current_volume)

Rules:
- volume == 0 -> Complete Remission (CR)
- <= -50% vs baseline -> Partial Remission (PR)
- >= +25% vs smallest volume -> Progression (PD)
- otherwise -> Stable Disease (SD)
"""