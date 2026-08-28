# Pre-outcome FM input/competence amendment V1

The first source-validation-only run invalidated two engineering assumptions. WBCIC's materialized cache is approximately 1000 times smaller than its metadata-declared `microvolts/20` scale: applying `x20` produced median absolute amplitude 0.00311 µV and q99 0.03308 µV. The corrected `x20000` transform produces physiologically plausible q99 near 33 µV. No outcome or S2/S3 utility was available.

CBraMod with optional mean pooling remained at chance on source validation. The official BCIC-IV-2a release instead defaults to `all_patch_reps` and a higher downstream-head learning rate. The repair adopts that official classifier, exposes its fixed 200-D penultimate vector, and uses official multi-LR optimization. Maximum epochs increase from 12 to 20 with early stopping. Subjects, folds, seeds, checkpoints, channels, labels, sessions, scientific definitions, and gates are unchanged.

The abandoned V0 runtime checkpoints are quarantined and never reused. This is the single pre-outcome unit/competence repair allowed by the protocol.
