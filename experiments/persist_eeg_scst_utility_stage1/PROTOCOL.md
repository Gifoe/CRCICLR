# Protocol

1. Revalidate ATCNet-CleanRoom from the frozen Stage-0 artifacts.
2. Correct the Stage-0 omission by source-only training of Braindecode EEGNeX.
3. Audit and separately train Braindecode ATCNet because the clean-room model is
   materially different from the mature implementation.
4. Determine eligibility using only source sessions on OpenBMI and WBCIC.
5. Hash and commit `SCST_STAGE1_TRAINING_LOCK.json` before reading WBCIC session 2.
6. Run matched ERM, Mixup, norm-matched random transport, SCST without
   consistency, and full SCST under identical folds, seeds, and budgets.
7. Use subjects as the statistical unit and 10,000 bootstrap draws.
8. Stop without accessing WBCIC outer 10 or the OpenBMI sealed holdout.

