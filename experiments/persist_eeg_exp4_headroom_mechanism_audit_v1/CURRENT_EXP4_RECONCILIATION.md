# Current Exp4 reconciliation

The preceding experiment remains `EXP4_PERSIST_GUARD_NOT_SUPPORTED`. Its clean
Strong Generic was 77.275% BA and its selected guard was 77.375% BA. The
historical 83.775% model remains illegal for the final 40/14 protocol.

`PREVIOUS_GATE_REPORTING_INCONSISTENCY`: the old check named
`G_beats_identity_and_confidence` was true because the implementation compared
guard **BA**, not risk AUROC. It must not be interpreted as mechanism-risk
superiority: PERSIST AUROC was 0.613 while confidence AUROC was 0.728. The old
FAIL conclusion is unchanged.
