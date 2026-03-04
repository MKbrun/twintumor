"""
Main pipeline that runs the tumor analysis workflow.

Steps:
1. Load mask files
2. Compute tumor volume for each scan
3. Run TreatmentAgent across volumes
4. Output results (scan, volume, response)

Expected output example:
Scan 1: 2.40 cm³ (baseline)
Scan 2: 2.10 cm³ → Stable Disease
Scan 3: 1.15 cm³ → Partial Remission
"""