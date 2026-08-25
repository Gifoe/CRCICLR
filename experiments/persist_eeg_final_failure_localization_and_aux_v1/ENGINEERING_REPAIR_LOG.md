# Engineering repair log

1. Phase B completed 225 inner lambda fits plus 75 outer fits on the server (5 folds x 3 seeds x 5 methods x 3 lambdas).
2. The process crashed only while appending the Vanilla replay rows: the frozen export `source_only_raw.csv` contained auxiliary source-only methods but no `B0_VANILLA_EEGNET` rows.
3. The legal frozen `replay_per_subject.csv` artifact was verified to contain 120 B0 rows (5 folds 脳 3 seeds 脳 8 outcome subjects), with no holdout or WBCIC flags. It was used only for post-processing the already completed run.
4. `code/closure.py` now falls back to `replay_per_subject.csv` when the source-only CSV has no B0 rows. No architecture, split, seed, lambda, epoch, teacher, target, or metric definition changed.
5. Post-processing was rerun without any model training. It generated all PUD-Aux CSV/JSON reports and figures.
6. Validation: 720 rows, 6 methods 脳 120 rows; zero duplicate `(method, fold, seed, subject_id)` keys; exactly 8 subjects per method/fold/seed; no restricted-data flags.
