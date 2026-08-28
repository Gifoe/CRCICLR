# Repair implementation log

## E1 — figure label access after raw metrics

- Bug: Repair-2 analysis wrote all compact numerical results, then failed before
  figures/reports because `setting_id` had been moved to the DataFrame index but
  the plotting loop accessed it as a tuple column.
- Diagnosis: presentation-only Pandas index/column mismatch.  No estimator,
  transport, control, gate, summary value, or terminal rule is involved.
- Fix declared before rerun: iterate over `ordered.iterrows()` and use the index
  as the label.  Add an engineering refreeze that records every pre-fix numeric
  output hash and requires exact equality after rerun.
- Before/after verification: `STAGE0_REPAIR2_E1_ENGINEERING_FREEZE.json` locked
  all seven pre-fix numerical outputs.  The repaired analyzer required exact
  SHA256 equality for every file; rerun passed, generated the figures, and did
  not change any numerical result or terminal.
