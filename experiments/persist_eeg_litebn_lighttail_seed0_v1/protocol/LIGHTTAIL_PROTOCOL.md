# Exact historical LiteBN LightTail seed0 protocol

Scope: OpenBMI MI only, seed0, original five folds. No ERP/SSVEP/WBCIC,
seed1/2, new split, new sealed test data, EMA, SWA, or Precise-BN.

The only mathematical training intervention is replacing ordinary CE with
`0.9 * mean(subject CE risks) + 0.1 * mean(hardest four subject risks)`.
Each actual historical episode has eight subjects, 16 examples per subject,
128 examples total. Subject grouping follows actual manifest row identities.
No resampling, sample reordering, or new episode construction is used in training.

## Immutable training

- Actual recovered `train_grid.train_model` is parsed and executed with exactly
  one loss-expression replacement and observational epoch logging. Its emitted
  function is saved for inspection. Existing historical routines provide model,
  cache/normalization, optimizer, validation, selection, and seeds.
- AdamW, lr 3e-4, weight decay 5e-4, original defaults otherwise, fp16 autocast
  and GradScaler, clip norm 5.0, 60 epochs. No early stop, no scheduler.
- Model initialization seed0, training RNG seed100000, cuDNN deterministic=True,
  benchmark=False, allow_tf32=True, matmul.allow_tf32=False.
- Selection: canonical inner future-session subject-equal BA; eligible epochs
  10..60; select only if BA exceeds best by >1e-12, retaining earliest ties.
- 21 episodes/epoch, 2,688 trial exposures/epoch, four S1 support subjects and
  four S2 query subjects, 1,344 exposures from each session. These S2 examples
  belong to inner-train subjects, as in the historical training. Removing them
  would change the protocol. Normalizer remains fitted on S1 inner-train only.
- Both optimizer attempts and actual AMP-successful updates/skips are logged.
  The AMP scaler is unchanged; loss-induced overflows, if any, must be disclosed.

## Gates before candidate training

All five stored manifests must match original provenance SHA256. Reconstruct
with actual deterministic historical sampler, original rows, original fold
seeds; require full JSON identity and identical LF-normalized SHA256. Audit all
6,300 episodes for row/trial identity, order, subjects and sessions. Do not train
on merely similar sampling. Store the exact original manifest for training.

Initialization must reproduce the original hash. Old/new Torch ZIP metadata and
Linux/Windows uniform-initializer rounding differ. The recovery module restores
the original CPU float32 uniform output bits, verifies the entire historical
initialization SHA256, and separately checks that container translation changes
no tensors. It is used only at initialization, then restored before training.

For each fold, run original CE and subject-loss weight0 on identical initial
models, identical real first episode, identical RNG, optimizer, and scaler.
Test float32 and historical AMP: scalar atol 1e-6; all gradients atol 1e-6,
rtol 1e-5; maximum parameter-update difference <=1e-6. An actual optimizer
update must occur; matching skipped steps is not sufficient. No formal training
until every test passes. Failed recovery is LIGHTTAIL_PROTOCOL_INVALID, never
negative evidence about tail-risk effectiveness.

## Evaluation and frozen interpretation

Finish all five folds before any candidate outer evaluation, then evaluate
current already-open internal heldout. Replay original selected baseline
against historical per-subject metrics (absolute tolerance 1e-10). Use original
weights/BN, never the Precise-BN recalibrated states. Heldout aggregation averages
five seed0 checkpoint metrics per subject, then subjects equally; bootstrap
10,000 paired subject draws with seed0. It is not a new untouched final test.

Report per-fold metrics and SD (ddof=0), per-subject deltas, means, median,
q25/q10/min deltas, positive ratio, and baseline-hardest-quartile mean gain.
Hardest-quartile membership is descriptive and does not select models.

Operational continuation thresholds frozen before candidate outcomes: both
outer and internal-heldout mean BA gains >=0.5 pp; no fold loss >=5 pp; fold
SD increase <=1 pp; positive mean gain in the baseline-hardest quartile in both
populations. If all hold, LIGHTTAIL_SIGNAL_FOUND, otherwise
LIGHTTAIL_NO_USEFUL_SIGNAL. Never expand tasks/seeds or tune coefficient from
these results without a new user request.

## Reproduction

Run `code/run_on_server.ps1` in a retained terminal session. External inputs:
`precisebn_recovered_historical`, `carrier_5fold_multiseed_stability_runtime`,
and the existing `srgeo_combined_cache` under server `D:/nips-temp/TotalP/P1`.
No original checkpoint/manifest is overwritten; outputs and candidate runtime
are isolated. Failed/resumed runs require the same invariant-bound code.
