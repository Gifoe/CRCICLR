# PERSIST-EEG SCAA Stage-0

This independent Stage-0 tests whether same-target adaptation utility measured
on an independent WBCIC S2 certificate session prospectively transfers to S3.
It does not revive prior invariance, selector, guard, or transport methods, and
it does not implement SCAA.

The fixed development cohort has 41 subjects. Each subject is evaluated with
the EEGNet and EEGConformer checkpoint from the subject-disjoint fold in which
that subject was held out. Three matched anchor seeds are averaged within
subject; the subject, not seed or trial, is the statistical unit. The WBCIC
outer 10 have no identifiers in this experiment and remain untouched. OpenBMI
is forbidden.

Execution order is enforced:

1. `audit_data.py` writes and verifies the data-access lock.
2. `run_competence.py` selects one global head-only recipe using S1 only.
3. `freeze_protocol.py` hashes all analysis code and writes the prospective
   protocol lock; the lock must then be committed.
4. Only after that commit, `run_utility.py` evaluates the frozen M0/M1 pair on
   S2 and S3.
5. `analyze.py` performs the frozen 10,000-resample subject analysis and writes
   compact results/reports/figures.
6. `validate.py` checks counts, locks, purity, outputs, and decision logic.

Runtime trial predictions, feature caches, raw EEG, and checkpoints are not Git
artifacts.
