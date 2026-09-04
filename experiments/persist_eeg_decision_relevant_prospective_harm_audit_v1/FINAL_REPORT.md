# Final report

terminal = DECISION_ENDPOINT_UNDERPOWERED

Source/refit-only, frozen seed-0 EEGNet trajectory. No development outcome, WBCIC outer-10, OpenBMI sealed/confirmation cohort, seed 1/2, or second backbone was opened.

|dataset|BBR->H_BBR AUROC|CI|BBR->H_BBR Spearman|BBR->H_BER AUROC|CE->H_BER AUROC|H_BER harmful events|Q5-Q1 decision harm|underpowered|
|---|---:|---|---:|---:|---:|---:|---:|---|
|OpenBMI|0.7002073732718894|[0.6348026091671409, 0.7486749314777857]|0.44916779668600193|0.49999999999999994|0.4328078078078078|6|0.011111111111111112|True|
|WBCIC|0.9298670977011494|[0.8911753424776897, 0.9593843452456551]|0.905377676196177|0.41044070361263474|0.5515415169283148|17|-0.029836829836829837|True|

## Required answers

- BBR K4 cross-batch signal: {'OpenBMI': True, 'WBCIC': True}
- Same-subject specificity: {'OpenBMI': True, 'WBCIC': True}
- Exact decision alignment: {'OpenBMI': False, 'WBCIC': False}
- BBR over CE: {'OpenBMI': False, 'WBCIC': False}
- Fold robustness: {'OpenBMI': True, 'WBCIC': True}
- Exact decision endpoint uses H_BER and correct-to-wrong flips; rare-event power is reported without changing the frozen schedule.
- No new model or guard is automatically started.
