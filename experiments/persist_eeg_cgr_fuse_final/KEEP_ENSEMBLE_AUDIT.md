# KEEP ensemble audit

K0--K7 use only the frozen KEEP predictions. STRONGEST_KEEP is selected as the best legal KEEP-only candidate on the declared development selection pool (with deterministic first-candidate tie breaking) and then carried unchanged across datasets. Ordinary ensembling is therefore audited explicitly; CGR-Fuse is compared to this strongest legal baseline rather than to a single run.

| dataset   | method                     |   selection_pool_subjects |   selection_BA |       BA |   macro_f1 | OUTER_TEST_USED   |
|:----------|:---------------------------|--------------------------:|---------------:|---------:|-----------:|:------------------|
| OpenBMI   | K0_SINGLE_KEEP             |                        10 |       0.8595   | 0.859231 |   0.857035 | False             |
| OpenBMI   | K1_KEEP_MAJORITY           |                        10 |       0.8495   | 0.849615 |   0.846277 | False             |
| OpenBMI   | K2_KEEP_MEAN_LOGIT         |                        10 |       0.8585   | 0.858462 |   0.856276 | False             |
| OpenBMI   | K3_KEEP_MEAN_PROBABILITY   |                        10 |       0.8595   | 0.859231 |   0.857035 | False             |
| OpenBMI   | K4_KEEP_MEDIAN_PROBABILITY |                        10 |       0.852    | 0.853462 |   0.851229 | False             |
| OpenBMI   | K5_KEEP_CALIBRATED         |                        10 |       0.8585   | 0.858462 |   0.856276 | False             |
| OpenBMI   | K6_I003_PROTECTED_SAFE     |                        10 |       0.8585   | 0.858462 |   0.856276 | False             |
| OpenBMI   | K7_I003_FULL               |                        10 |       0.8585   | 0.858462 |   0.856276 | False             |
| WBCIC     | K0_SINGLE_KEEP             |                        41 |       0.765196 | 0.765196 |   0.759872 | False             |
| WBCIC     | K1_KEEP_MAJORITY           |                        41 |       0.764458 | 0.764458 |   0.760031 | False             |
| WBCIC     | K2_KEEP_MEAN_LOGIT         |                        41 |       0.766293 | 0.766293 |   0.761091 | False             |
| WBCIC     | K3_KEEP_MEAN_PROBABILITY   |                        41 |       0.765196 | 0.765196 |   0.759872 | False             |
| WBCIC     | K4_KEEP_MEDIAN_PROBABILITY |                        41 |       0.759345 | 0.759345 |   0.75155  | False             |
| WBCIC     | K5_KEEP_CALIBRATED         |                        41 |       0.766293 | 0.766293 |   0.761091 | False             |
| WBCIC     | K6_I003_PROTECTED_SAFE     |                        41 |       0.766293 | 0.766293 |   0.761091 | False             |
| WBCIC     | K7_I003_FULL               |                        41 |       0.766293 | 0.766293 |   0.761091 | False             |