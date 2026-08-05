# Validation record

- Validator: `python scripts/budgeted_risk/validate_stage0.py`
- Status: `VALID`
- Full-context result rows: 775
- Budget result rows: 113,925
- Temporal rows: 5,425
- Random rows: 108,500 (20 repeats)
- Clean source heads: 50
- Source cache files: 3,875
- Random decision batches: 5,425
- Source/head evaluation, calibration, formal, final, and CAP overlap total: 0
- `formal_calibration_opened`: false
- `internal_final_opened`: false
- `cap_opened`: false
- Full test suite: 156 passed, 3 warnings

The warnings are one `mord`/SciPy deprecation warning and two pre-existing
constant-input Spearman warnings in the V2 benefit-predictor tests.

