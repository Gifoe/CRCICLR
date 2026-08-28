# Adaptation iteration ledger

## V0 head-only supervised adaptation

- Diagnosed requirement: test an ordinary target-S1 adaptation action without
  introducing a selector or new architecture.
- Proposed before S2/S3: freeze the encoder and adapt only the two-class head;
  select one global LR from `[0.0001, 0.0003, 0.001]` using S1 validation only.
- Predicted competence signature: nonzero parameter/prediction change, mean S1
  validation improvement >=0.5 pp, mean adapted BA >=0.60, and catastrophic
  fraction <=0.10.
- Actual S1-only result: selected LR `0.001`, mean BA delta
  `+0.01524`, competence pass `True`.
- Decision: `freeze head-only recipe; no last-block repair`.
- S2/S3 utility inspected: NO.

## Post-lock engineering compatibility repair 001

- The first frozen utility execution completed all 30 feature/adaptation units,
  then failed before writing or displaying any utility value.
- Failure: the server pandas version rejects
  `pd.to_numeric(..., errors="ignore")` during deterministic row sorting.
- Repair: replace only that sorting expression with an integer temporary
  subject-sort column, sort by the same backbone/subject/seed keys, and remove
  the temporary column.
- Adapter, trainable parameters, S1 split, LR, epochs, checkpoints, subjects,
  folds, backbones, S2/S3 evaluation, statistics, and certificate are unchanged.
- No utility result was persisted, printed, read, or used to choose this repair.
