# FM training ledger (pre-outcome)

Frozen search: CBraMod learning rates `[0.0001, 0.0003]`; LaBraM learning rates `[0.0001, 0.0005]`; AdamW weight decay `0.05`; at most `12` epochs; minimum `4`; patience `3`; BF16; batch `128`. Selection is mean subject-balanced validation BA over all five frozen folds at seed 0. The selected recipe is then run for seeds 1 and 2. No outcome subject or WBCIC S3 is used for selection.

Competence thresholds were frozen before FM outcome BA: OpenBMI `0.7519166667` from specialist `0.7719166667`; WBCIC `0.7684300821` from specialist `0.7884300821`.

## Source-validation selection complete

| fm      | dataset   |     lr |   mean_validation_BA |   minimum_fold_BA |   folds |
|:--------|:----------|-------:|---------------------:|------------------:|--------:|
| CBraMod | OpenBMI   | 0.0003 |             0.727375 |          0.699375 |       5 |
| CBraMod | WBCIC     | 0.0003 |             0.75683  |          0.699375 |       5 |
| LaBraM  | OpenBMI   | 0.0005 |             0.69775  |          0.66     |       5 |
| LaBraM  | WBCIC     | 0.0005 |             0.76849  |          0.693437 |       5 |

S1-only head adaptation (no S2/S3 access):

| fm      |    lr |   cells |   anchor_BA |   adapted_BA |     delta |      NLL |   prediction_change |   parameter_change |
|:--------|------:|--------:|------------:|-------------:|----------:|---------:|--------------------:|-------------------:|
| CBraMod | 0.001 |     123 |    0.741463 |     0.77561  | 0.0341463 | 0.486197 |           0.103252  |           0.165824 |
| LaBraM  | 0.001 |     123 |    0.744444 |     0.767886 | 0.0234417 | 0.48077  |           0.0852304 |           0.191819 |
