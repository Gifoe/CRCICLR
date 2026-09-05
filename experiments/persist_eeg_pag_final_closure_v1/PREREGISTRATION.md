# PAG preregistration before sealed confirmation

This document freezes the confirmation protocol before any new sealed outcome is read. It is a confirmatory test of the Persistence-Actionability Gap, not a method search.

## Frozen cohorts and scope

- OpenBMI internal sealed holdout and WBCIC outer sealed subjects, only if the authorized resources are materialized and independently audited.
- No development outcome is used to change the protocol. No WBCIC outer-10 or OpenBMI sealed identifier is enumerated before the lock commit.
- No other folds, backbones, selectors, adapters, routing methods, or ablations are added.

## Frozen comparison

Primary method: PUD-Aux. Comparator: exact matched task-only control. Both use the identical single-path EEGNet F8/F16, embedding dimension 64, dropout 0.25, initialization, parent minibatch order, augmentations, optimizer, training budget, source-only validation, and target-label exclusion. PUD-Aux adds only the training-only linear auxiliary head with the frozen teacher target. Lambda is selected on source inner validation from the already frozen grid {0.05, 0.10, 0.25}; no target information selects lambda or epoch.

Optimization seeds are frozen to 0, 1, 2, 3, 4. A seed cannot be removed after its outcome is observed.

## Primary outcome and effect gate

Primary metric is subject-balanced balanced accuracy. The minimum meaningful actionability gain is +0.5 percentage points BA. Secondary metrics are macro-F1, per-subject BA, worst-subject BA, subject-level harm, mean, median, SD, and range.

PUD-Aux versus task-only uses paired subject analysis. Report mean and median paired subject delta, 95% subject bootstrap CI, one-sided upper confidence bound relative to +0.5 pp, fraction of subjects improved, catastrophic harm count, and the complete per-seed table. Seeds are not biological subjects.

## Frozen decision states

- `PAG_STRONG_CONFIRMATION`: both representation and shared-geometry premises replicate; PUD-Aux fails the meaningful-gain gate; the primary sealed cohort's 95% upper bound excludes +0.5 pp; no other cohort shows a robust opposite constructive success; no protocol violation.
- `PAG_PARTIAL_CONFIRMATION`: premises broadly replicate but the actionability CI contains +0.5 pp, datasets are mixed, or evidence is insufficient to exclude a meaningful gain.
- `PAG_PREMISE_NOT_REPLICATED`: persistent or geometry premise fails its preregistered continuous outcome.
- `PAG_FALSIFIED_BY_ACTIONABILITY`: PUD-Aux has consistent positive direction, at least +0.5 pp meaningful gain, statistical support, and no catastrophic subject harm.
- `PAG_MIXED`: cohort directions conflict or evidence is insufficient for a unified state.

No threshold, metric, seed, method, gate, or statistical test may be changed after the lock commit. If code is found faulty after outcome access, the original result is preserved and any repair is labeled `POST_UNBLINDING_REPAIR`.
