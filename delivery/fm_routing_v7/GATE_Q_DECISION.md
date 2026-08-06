# Gate Q decision

| dataset   | model   |   n_classes |   chance_balanced_accuracy |   dataset_balanced_accuracy |   median_subject_balanced_accuracy |   seed_ba_std |   nonconstant_subject_rate | Q1   | Q2    | Q3    | Q4    | Q5    | Q6    | Q7   | Q8    | Q9    | passed   |
|:----------|:--------|------------:|---------------------------:|----------------------------:|-----------------------------------:|--------------:|---------------------------:|:-----|:------|:------|:------|:------|:------|:-----|:------|:------|:---------|
| hmc       | cbramod |           5 |                       0.2  |                    0.487325 |                           0.492707 |    0.00707204 |                   1        | True | True  | True  | True  | True  | True  | True | True  | True  | True     |
| hmc       | labram  |           5 |                       0.2  |                    0.490922 |                           0.502399 |    0.0597872  |                   0.977778 | True | True  | True  | False | True  | True  | True | True  | True  | False    |
| hmc       | biot    |           5 |                       0.2  |                    0.489724 |                           0.497981 |    0.0779612  |                   1        | True | True  | True  | False | True  | True  | True | True  | True  | False    |
| eegmmidb  | cbramod |           4 |                       0.25 |                    0.253841 |                           0.255556 |    0.00428323 |                   1        | True | True  | True  | True  | False | False | True | True  | True  | False    |
| eegmmidb  | labram  |           4 |                       0.25 |                    0.250074 |                           0.25     |    0.00319913 |                   0.553846 | True | False | False | True  | False | False | True | False | False | False    |
| eegmmidb  | biot    |           4 |                       0.25 |                    0.258443 |                           0.257806 |    0.00408479 |                   0.876923 | True | True  | False | True  | False | False | True | True  | True  | False    |

All models pass: **False**
