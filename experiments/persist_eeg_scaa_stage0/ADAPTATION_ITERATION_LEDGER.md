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
