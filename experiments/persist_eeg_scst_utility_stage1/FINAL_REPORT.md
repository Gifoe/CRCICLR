# Final report

Terminal: `SCST_UTILITY_NOT_SUPPORTED_IN_NEAR_ADMISSIBLE_SPACE`

Under the prospectively frozen Stage-1 recipe, Full SCST did not provide convincing subject-level improvement over matched ATCNet-CleanRoom ERM.

| Model            | Type              |     BA_ERM |     3NN |   Old_le_1p25 |   New_le_1p30 | Stable   | Subject_Fidelity   | Random_Advantage   | Class_Fidelity   |    SCST_BA |      Delta_BA | CI                    | Terminal          |
|:-----------------|:------------------|-----------:|--------:|--------------:|--------------:|:---------|:-------------------|:-------------------|:-----------------|-----------:|--------------:|:----------------------|:------------------|
| EEGNet           | Negative control  |   0.806528 | 1.32998 |             0 |             0 | True     | True               | True               | True             |   0.806081 |  -0.000447154 | [-0.001829, 0.000894] | SCST_NOT_POSITIVE |
| EEGConformer     | Negative control  |   0.808472 | 1.3141  |             0 |             0 | True     | True               | True               | True             |   0.809366 |   0.000894309 | [0.000000, 0.001789]  | SCST_POSITIVE     |
| CBraMod          | Secondary/control | nan        | 1.15007 |           nan |           nan | True     | True               | True               | True             | nan        | nan           | NOT_RUN               | NOT_RUN           |
| ATCNet-CleanRoom | Primary           |   0.805164 | 1.27358 |             0 |             1 | True     | True               | True               | True             |   0.80541  |   0.000245596 | [-0.001260, 0.001831] | SCST_NOT_POSITIVE |
| ATCNet-Official  | Secondary/control | nan        | 1.23753 |             1 |             1 | True     | True               | True               | True             | nan        | nan           | NOT_RUN               | NOT_RUN           |
| EEGNeX           | Secondary/control | nan        | 1.1863  |             1 |             1 | True     | True               | True               | True             | nan        | nan           | NOT_RUN               | NOT_RUN           |
