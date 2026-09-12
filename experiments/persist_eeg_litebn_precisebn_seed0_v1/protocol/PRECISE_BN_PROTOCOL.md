# Source-only Precise-BN seed0 protocol

Status: **Recovered-source execution authorized; original blocker retained in
the initial audit files.** The user supplied a second server with original
three-seed checkpoints. All 60 selected files have been transferred and SHA256
verified; only the 20 seed0 checkpoints enter this diagnostic.

Requested branch: `codex/persist-eeg-litebn-precisebn-seed0-v1`.
Scope: exact historical `CompactLite` / `LiteBN_BASELINE`, four tasks
(OpenBMI MI, ERP, SSVEP, WBCIC MI), original five folds, seed0 only.
No optimizer, backward, learnable-parameter updates, retraining, architecture
substitution, seed1/2 experiment, or light-tail experiment is authorized here.

## Ordered gates

1. Locate all 20 selected checkpoints. Record hashes, task/fold/seed, selection,
   split, preprocessing/normalizer, source commit. Unknown provenance remains
   unknown; repository inspection commit is not checkpoint creation commit.
2. Strict-load the actual committed historical class. A task-size head defines
   the expected schema only; no checkpoint tensor may be replaced or reshaped.
3. Replay all original outer results using their original evaluation pipelines;
   replay already-open internal diagnostic heldout where historical results exist.
   No calibration until provenance and baseline replay pass for all sources.
4. Only then calibrate once over all inner-train source-session examples, in a
   deterministic order, fixed batch size 128 across tasks. Historical channels,
   preprocessing and source-fitted normalization must remain unchanged.
5. Overall model stays eval (including functional dropout); only BN modules use
   batch statistics. Freeze all parameters and use no_grad. For each BN input,
   float64 sum/squared-sum/count include every non-channel dimension. Write
   mean=sum/count and variance=max(sum2/count-mean^2,0). No old/new blending,
   momentum search, batch-size search, calibration fraction search or repeats.
6. Require bitwise identical named parameters and non-BN buffers before/after.
   Only BN running_mean, running_var and num_batches_tracked may change.
7. Evaluate paired original/recalibrated outer folds, then the already-open
   internal heldout with unchanged population and aggregation. No new sealed data.
8. Report BA, macro-F1, accuracy, per-fold and per-subject deltas, task means,
   fold SD changes, signs, equal-task mean/median/worst deltas, and historical
   paired bootstrap where supported. Report per-layer count, norms, absolute and
   relative moment shifts. Do not select procedures using these outcomes.

The signal labels require valid completed evidence. `PRECISE_BN_SIGNAL_FOUND`
requires majority-task improvement in both populations, positive equal-task
means, and no meaningful fold-SD deterioration. Opposite/zero/inconsistent
results support `PRECISE_BN_NO_USEFUL_SIGNAL`. A prerequisite failure is neither:
use `BLOCKED_SOURCE_ARCHITECTURE_MISMATCH`, with no scientific decision.
Before the recovered-source run, operationalize the SD criterion as: mean
four-task fold-SD change <= 0 and no individual task increase > 1 percentage
point; fold SD uses ddof=0. This threshold is frozen before any Precise-BN
outcome is computed. BA sign ties use 1e-8 pp; baseline replay tolerance is
1e-10 absolute for each subject's BA, macro-F1 and accuracy.

## Recovery amendment and active execution

The second server supplied `openbmi_task_generality_runtime` exact ERP/SSVEP
CompactLite selected checkpoints, as well as carrier MI checkpoints. Source
code/protocol/results are hash-verified against TRANSFER_RECEIPT.json. Training
commit and repository inspection commit are distinguished in the manifest.
The canonical model class is executed directly from the transferred Python
AST; no architecture approximation. Historical task/final evaluation functions
and normalizers are imported from that snapshot, with path overrides only.

`run_precisebn.py --phase all --repo ... --recovered ... --cache ... --runtime ...`
is the active runner. It writes an invariant-bound replay gate, then performs
one calibration per checkpoint and all outer evaluations before recalibrated
heldout evaluation. Original heldout replay necessarily occurs before
calibration, as required by the replay gate. Historical replicate averaging is
restricted to seed0 (five checkpoints, not fifteen); it is metric averaging per
subject, not logit ensembling. Subject bootstrap uses 10,000 draws, seed0.

Streaming batches replace large GPU dataset materialization without changing
normalization or prediction batching. Calibration source sessions are OpenBMI
S1 and WBCIC session indices 0/1. No calibration label array is read.

## Reproduce the executed source gate on the server

From this experiment's `code` directory, using the existing CUDA environment:

```powershell
$repo = 'D:\nips-temp\TotalP\P1\CRCICLR_BACKBONE_GEN_WORK'
$out = "$repo\experiments\persist_eeg_litebn_precisebn_seed0_v1\outputs"
& 'E:\Anaconda\envs\persist_stable_251\python.exe' audit_sources.py --root 'D:\nips-temp\TotalP\P1' --repo $repo --output $out
& 'E:\Anaconda\envs\persist_stable_251\python.exe' resolve_source_gate.py --repo $repo
```

The two initial audit scripts implement source inspection only. They never
read EEG, restore optimizers or write checkpoints. The active recovered-source
runner is separate, and its real unit tests cover unequal-batch population
moments, dropout suppression and state invariance.
Recovery inventories include other historical seeds solely for provenance;
no seed1/2 model forward/training was performed. Scientific CSVs with headers
only mean NOT RUN, never zero effect.
