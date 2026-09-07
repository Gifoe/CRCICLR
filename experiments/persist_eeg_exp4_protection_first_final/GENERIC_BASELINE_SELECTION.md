# Generic baseline selection

Generic selection was completed before the main Guard comparison using only S2 subject-held-out validation inside each model-fit set. No outcome-subject S3 labels entered selection. All candidates use the same S1 EEGNet anchor, the same zero-initialized 32-by-32 residual adapter, frozen classifier, S2 update data, and deterministic seed family.

| candidate | learning rate | epochs | weight decay | fold S2 BA | mean S2 BA |
|---|---:|---:|---:|---|---:|
| `GEN_LINEAR_LR3E4_E25` | 3e-4 | 25 | 5e-4 | 0.7258, 0.7842, 0.8492, 0.7625, 0.8125 | 0.7868 |
| `GEN_LINEAR_LR1E3_E25` | 1e-3 | 25 | 5e-4 | 0.7342, 0.7883, 0.8508, 0.7608, 0.8167 | **0.7902** |
| `GEN_LINEAR_LR3E4_E40` | 3e-4 | 40 | 5e-4 | 0.7317, 0.7867, 0.8483, 0.7633, 0.8150 | 0.7890 |

The frozen candidate was `GEN_LINEAR_LR1E3_E25`. It was not selected because it maximized Guard separation; it was selected because it had the highest pre-comparison generic S2 validation mean. Its S3 development BA was 0.7719, compared with 0.7012 for Frozen, confirming that the experiment had real adaptation headroom.
