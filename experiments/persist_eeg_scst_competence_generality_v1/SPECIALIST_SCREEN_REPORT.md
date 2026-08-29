# Specialist screen report

| model          | type       | dataset   |       BA |   macro_F1 |      NLL |   threshold |   folds |   seeds |   parameters | competent   |
|:---------------|:-----------|:----------|---------:|-----------:|---------:|------------:|--------:|--------:|-------------:|:------------|
| ATCNet         | Specialist | OpenBMI   | 0.767042 |   0.766663 | 0.514092 |    0.751917 |       5 |       3 |        21698 | True        |
| ATCNet         | Specialist | WBCIC     | 0.7859   |   0.785647 | 0.431317 |    0.76843  |       5 |       3 |        21570 | True        |
| EEGInceptionMI | Specialist | OpenBMI   | 0.743167 |   0.74207  | 0.524175 |    0.751917 |       5 |       3 |        47170 | False       |
| EEGInceptionMI | Specialist | WBCIC     | 0.605991 |   0.602258 | 0.685742 |    0.76843  |       5 |       3 |        46978 | False       |
| FBCNet         | Specialist | OpenBMI   | 0.656708 |   0.655048 | 0.629571 |    0.751917 |       5 |       3 |         5186 | False       |
| FBCNet         | Specialist | WBCIC     | 0.59562  |   0.591832 | 0.708907 |    0.76843  |       5 |       3 |         4898 | False       |

| model          | terminal                            | competent_both_datasets   | admissible_both_datasets   |
|:---------------|:------------------------------------|:--------------------------|:---------------------------|
| ATCNet         | SPECIALIST_COMPETENT_NOT_ADMISSIBLE | True                      | False                      |
| EEGInceptionMI | SPECIALIST_NOT_COMPETENT            | False                     | False                      |
| FBCNet         | SPECIALIST_NOT_COMPETENT            | False                     | False                      |
