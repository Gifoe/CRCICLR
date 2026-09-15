# Seed-0 interim heldout amendment

Recorded before the interim heldout evaluation.

- Scope: seed 0 only, all four tasks and all five frozen folds.
- Training recipe, split, checkpoint selection, and model definition are unchanged.
- The seed-0 checkpoint family is hash-locked before any heldout signal or label is loaded.
- Outputs use the `SEED0_INTERIM_` prefix and do not replace the three-seed final lock or outputs.
- This read is descriptive only. It must not be used to tune, repair, select, or redefine any model.
- Seeds 1 and 2 remain exact replications of the already frozen method.
