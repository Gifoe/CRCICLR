# Repair rationale R1

This is an implementation-only repair pass, made before the first server run.

1. The historical I003 audit referenced `pred_erase` without carrying the
   corresponding frozen logits into its merged frame.  The merge now includes
   both ERASE logits, while ERASE remains diagnostic-only.
2. The primary aggregate feature list accidentally included ERASE although the
   preregistered CGR-Fuse action bank is KEEP/AMPLIFY/GEOMETRY.  ERASE is now
   excluded from the primary feature matrix.
3. `SOURCE_PER_FOLD.csv` assigned each subject through a one-subject fold map,
   making every row fold 0.  Folds are now derived once from the complete
   biological-subject set within each dataset.
4. The minimum-support gate compared BA to a recipe delta.  It now compares
   final BA with the legal STRONGEST_KEEP BA baseline.
5. Protocol-level `FUSION_WEIGHTS.csv` and `ACTION_ADVANTAGE.csv` aliases are
   emitted by concatenating the per-dataset compact tables.

No outcome was inspected to choose a method, no scientific search dimension
was added, and no action/architecture/threshold/seed rule was changed.
