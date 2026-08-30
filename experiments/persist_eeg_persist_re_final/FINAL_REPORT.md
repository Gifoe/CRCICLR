# Final report

## Terminal

`PERSIST_RE_SOURCE_NOT_SUPPORTED`

The CleanRoom source gate failed before any confirmation architecture was authorized.

## Source results

- OpenBMI: delta=-2.775557561562892e-18, paired CI=[-0.0004166666666666, 0.000375].
- WBCIC: delta=0.0002032520325203, paired CI=[-6.0975609756090846e-05, 0.0005284552845528].

## Controls

GroupDRO, ProspectiveOnly, and RandomEffectOnly comparisons are in `results/ABLATION_SUMMARY.csv`; no post-hoc rule was changed.

## Resource boundary

WBCIC outer subjects, WBCIC session-2 utility, and the OpenBMI sealed holdout were untouched.  Exploratory additional-backbone files, if present, are not confirmation evidence.

## Claim boundary

The implementation enforces decision-level random-effect quarantine and population-only inference; no utility improvement is supported by the CleanRoom source gate.
