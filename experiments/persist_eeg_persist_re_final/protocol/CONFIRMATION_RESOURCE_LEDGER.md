# Confirmation resource ledger

| Resource | Status | Rule |
|---|---|---|
| OpenBMI authorized development transition | DEVELOPMENT_KNOWN | source-only recipe search |
| WBCIC S1 -> S2 development transition | DEVELOPMENT_KNOWN | source-only recipe search; session 2 is not read before lock |
| WBCIC session-2 utility | UNTOUCHED | may open only after a committed source lock |
| WBCIC outer subjects | SEALED | never open in this task |
| OpenBMI sealed holdout | SEALED | never open in this task |

The ledger is updated only by the experiment scripts and is committed with the
protocol locks.
