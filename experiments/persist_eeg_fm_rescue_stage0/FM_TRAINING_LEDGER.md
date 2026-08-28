# FM training ledger (pre-outcome)

Frozen search: CBraMod learning rates `[0.0001, 0.0003]`; LaBraM learning rates `[0.0001, 0.0005]`; AdamW weight decay `0.05`; at most `12` epochs; minimum `4`; patience `3`; BF16; batch `128`. Selection is mean subject-balanced validation BA over all five frozen folds at seed 0. The selected recipe is then run for seeds 1 and 2. No outcome subject or WBCIC S3 is used for selection.

Competence thresholds were frozen before FM outcome BA: OpenBMI `0.7519166667` from specialist `0.7719166667`; WBCIC `0.7684300821` from specialist `0.7884300821`.
