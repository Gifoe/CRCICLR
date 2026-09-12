# LiteBN-LightTail: OpenBMI MI seed0

**LIGHTTAIL_NO_USEFUL_SIGNAL**. All five original folds completed 60 epochs.
No additional seeds/tasks or tail weights were run for this experiment.

## Six requested answers

1. Exact manifest: yes, all 6,300 episodes match original subject/session/trial
   composition and order, with 21 episodes x 128 trials per epoch. All ten
   tail0 tests pass, including actual AdamW parameter updates.
2. Outer-development BA decreased from 79.150% to 77.625% (-1.525 pp).
   Fold directions: 0 positive, 4 negative, 1 tied.
3. Current internal heldout BA decreased from 74.486% to 73.814% (-0.671 pp).
   Paired subject bootstrap 95% CI: [-1.529, +0.171] pp. This is an already-open
   diagnostic population, not a sealed final test.
4. Lower-tail improvement was not observed. Baseline-hardest-quartile mean
   changes were -1.700 pp outer and -0.500 pp heldout. Delta q10 was -6.100 pp
   outer and -2.940 pp heldout; worst deltas were -14.000 and -3.400 pp.
5. Across-fold population SD increased from 3.758 to 3.971 pp (+0.213 pp).
6. Do not expand to seed1/2 or other tasks based on this diagnostic.

## Audit and interpretation

Both baseline and candidate made 6,300 update attempts and 6,297 successful
AMP updates, but successful counts differ by fold:

| Fold | Historical successful steps | LightTail successful steps |
| --- | ---: | ---: |
| 0 | 1260 | 1259 |
| 1 | 1259 | 1260 |
| 2 | 1259 | 1259 |
| 3 | 1259 | 1260 |
| 4 | 1260 | 1259 |

AMP/scaler policy was unchanged; overflow-skipped steps were not forced.
`TRAINING_EXPOSURE_MATCHED=YES` refers to exact attempted episodes and sample
exposure, not identical successful-step timing. The current Windows runtime
is not the original Linux binary environment. Exact initial tensors, manifests,
normalizers and baseline evaluation metrics were verified; a full same-runtime
ordinary-CE retraining control was not run. Therefore this result is a seed0
diagnostic, not a clean cross-platform causal isolation of every numerical effect.
See [engineering ledger](ENGINEERING_LEDGER.md).

Independent checks passed on the server and downloaded artifacts: 6,300 manifest
rows, 10 equivalence tests, 300 training epochs and 110 paired evaluation rows.
All original baseline BA/F1/accuracy values replayed within 1e-10 before comparison.
Selected checkpoint hashes and training records are in
`outputs/OPTIMIZER_STEP_AUDIT.json`; checkpoint binaries and EEG caches remain
on the server. Existing unrelated jobs were not terminated or modified.

## Reproduction

The explicit server entrypoint is `code/run_on_server.ps1` (paths are editable
engineering inputs). It uses the recovered historical code and isolated
`D:\nips-temp\TotalP\P1\lighttail_seed0_runtime`, with invariant-checked resume.
The complete original manifests, checkpoints and cache must already exist.
Run `python code/verify_outputs.py outputs` to independently verify the exported
artifact cardinalities and arithmetic without EEG data or a GPU.

The frozen protocol is in `protocol/LIGHTTAIL_PROTOCOL.md`; the full decision,
mandatory scope flags and metric tables are in `outputs/`.
