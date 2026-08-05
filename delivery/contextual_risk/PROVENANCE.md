# Provenance

The final run started from commit `4469a5b3132a51386e8c0b364446147225b07940` on `v4-contextual-risk-select-and-run`. Configuration, cohort, episode, cache, feature, screening-freeze, branch-result, and branch-selection hashes are recorded in `outputs/contextual_risk/RUN_STATE.json`.

One pre-final implementation audit found that an early shared-table builder derived unused formal-calibration/internal-final outcome surfaces. Although A/B code filtered to development, this violated the access contract. The affected output and delivery trees were moved intact to:

`/root/autodl-tmp/hsc_tta_eeg/invalidated/contextual_risk_preaccess_fix_20260805T0813Z`

The builder was corrected so non-development cache access ends after context probabilities and hashes. All downstream tables, freezes, reports, and the selection decision were then regenerated. `ISOLATION_VALIDATION.json` checks that the final surface contains only development rows, A/B use identical subject-seed-alpha keys, and neither reserved cohort appears in screening.

Raw EEG, token HDF5 data, model checkpoints, and per-subject source-cache NPZ files are excluded from Git. All code, aggregate and subject-level screening outputs, manifests, tests, reports, and validation records are included.
