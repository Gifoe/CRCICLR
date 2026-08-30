# SCST training report

The fixed Option-A bank and matched 15-epoch budgets were used.

| model            | method             |       BA |   median_subject_BA |   CI95_L |   CI95_U |
|:-----------------|:-------------------|---------:|--------------------:|---------:|---------:|
| ATCNet-CleanRoom | ERM                | 0.805164 |            0.841667 | 0.762118 | 0.846667 |
| ATCNet-CleanRoom | Mixup              | 0.806176 |            0.843333 | 0.763733 | 0.846624 |
| ATCNet-CleanRoom | RandomTransport    | 0.806949 |            0.848333 | 0.763902 | 0.847804 |
| ATCNet-CleanRoom | SCST-NoConsistency | 0.806014 |            0.841667 | 0.764308 | 0.846871 |
| ATCNet-CleanRoom | Full-SCST          | 0.80541  |            0.845    | 0.762764 | 0.846626 |
| EEGNet           | ERM                | 0.806528 |            0.85     | 0.764022 | 0.846245 |
| EEGNet           | Mixup              | 0.805552 |            0.843333 | 0.762584 | 0.846268 |
| EEGNet           | RandomTransport    | 0.806362 |            0.845    | 0.76453  | 0.847297 |
| EEGNet           | SCST-NoConsistency | 0.80604  |            0.843333 | 0.763192 | 0.846936 |
| EEGNet           | Full-SCST          | 0.806081 |            0.848333 | 0.76429  | 0.847196 |
| EEGConformer     | ERM                | 0.808472 |            0.85     | 0.765324 | 0.848252 |
| EEGConformer     | Mixup              | 0.809163 |            0.848333 | 0.766114 | 0.851089 |
| EEGConformer     | RandomTransport    | 0.80896  |            0.85     | 0.765513 | 0.850653 |
| EEGConformer     | SCST-NoConsistency | 0.809204 |            0.85     | 0.7663   | 0.850342 |
| EEGConformer     | Full-SCST          | 0.809366 |            0.851667 | 0.766137 | 0.850928 |
