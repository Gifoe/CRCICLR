# V6 development design

## Split and information locks

- OpenBMI uses the frozen five subject folds from Stage-0: 54 subjects, 100 MI
  trials per session, S1 target history and S2 future scoring.
- WBCIC uses the five `audit_roles` folds in `DEVELOPMENT_SCOPE_LOCK.json`: 41
  authorized development subjects, S1/S2 target history and S3 future scoring.
- Model-fit, discovery, and outcome subjects are disjoint inside each fold.
  Model-fit future sessions may train the population rule; discovery future
  sessions may select a rule; outcome future labels are scoring-only.
- Fold-specific representations are never mixed across coordinate systems.
- The WBCIC sealed outer cohort remains outside the runtime scope.

## Decision rules

The primary target is at least +5 pp over the strongest legal generic method
with the same target-history budget. The user-approved secondary target is +5 pp
over the frozen EEGNet reference. Both are evaluated by mean subject-balanced
accuracy; uncertainty uses 20,000 paired subject-bootstrap draws. Macro-F1,
accuracy, NLL, Brier, ECE, fold positivity, subject positivity, and worst-subject
delta are retained as diagnostics.

Only one seed is evaluated. This is an exploratory screen, not a robustness
confirmation. Architecture, epoch, and adaptation choices are made on
discovery subjects, followed by refit on all non-outcome subjects. Fixed fusion
rules and history-only gates do not inspect outcome future labels.

## Stop rule and outcome

Search stops after structurally distinct adapter, geometry, prototype,
fine-tuning, protection, population-update, backbone-capacity, fusion, and
selective-head families fail to produce a dual-benchmark improvement. Reusing
the same outcome labels for additional post-hoc threshold search would make the
development estimate less credible.

The terminal state is `V6_OPENBMI_TARGET_ONLY`: OpenBMI passes the secondary
and pre-V6 matched-anchor +5 pp thresholds, WBCIC passes neither, and PERSIST
does not beat the strongest generic control on either benchmark. Therefore
`READY_FOR_OUTER_FREEZE=false`.
