# Incremental relation frozen-feature pilot

This is a time-bounded seed-0, fold-0 pilot on OpenBMI and WBCIC. It reuses
the source-only canonical EEGNet checkpoint from
`persist_eeg_geosr_final_v1`, extracts the full 64-dimensional latent once,
and compares subject-balanced ERM with a generic residual, a generic class
prototype, and an explicit leave-one-source-subject-out cross-session
relation. The outer outcome is read only after both source audits and the
pre-outcome lock exist.

The pilot is screen-only. It cannot authorize a final claim or substitute for
the original five-fold protocol. Runtime feature arrays and any checkpoints
remain outside Git under the ignored `runtime/` directory; compact tables,
hash locks, provenance and validation are committed.

Run on the authorized server:

```text
<GPU-python> code/run_incremental_relation_pilot.py --phase train --root <pilot-root> --device cuda:0
<GPU-python> code/run_incremental_relation_pilot.py --phase outcome --root <pilot-root> --device cuda:0
<GPU-python> code/validate_incremental_relation.py --pilot <pilot-root> --require-runtime
```

The recorded terminal is `INCREMENTAL_RELATION_STOP_NO_CLEAR_GAIN`. OpenBMI
relation is +0.273 pp BA and WBCIC is +0.500 pp, but neither passes the
pre-registered cross-dataset and generic-control margins.
