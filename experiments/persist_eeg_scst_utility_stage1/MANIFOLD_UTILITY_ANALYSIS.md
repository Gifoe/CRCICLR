# Manifold–utility analysis

| model            |   independent_session_3NN_ratio |   off_manifold_excess_vs_random |   residual_stability |   subject_fidelity |   delta_BA_SCST_minus_ERM |
|:-----------------|--------------------------------:|--------------------------------:|---------------------:|-------------------:|--------------------------:|
| ATCNet-CleanRoom |                         1.27358 |                      -0.0619082 |             0.579632 |          0.0984082 |               0.000245596 |
| EEGNet           |                         1.32998 |                      -0.0884614 |             0.560378 |          0.0953423 |              -0.000447154 |
| EEGConformer     |                         1.3141  |                      -0.0232729 |             0.612548 |          0.108792  |               0.000894309 |

```json
{
  "n_models": 3,
  "tests": {
    "independent_session_3NN_ratio": {
      "pearson_r": -0.29115492160973333,
      "pearson_p": 0.8119209761500995,
      "spearman_r": -0.5,
      "spearman_p": 0.6666666666666666,
      "robust_slope": -0.012119086515783618,
      "robust_intercept": 0.015681918238993026
    },
    "off_manifold_excess_vs_random": {
      "pearson_r": 0.9921286272787588,
      "pearson_p": 0.07992923308366325,
      "spearman_r": 1.0,
      "spearman_p": 0.0,
      "robust_slope": 0.020575385830160327,
      "robust_intercept": 0.0013797822294452763
    },
    "residual_stability": {
      "pearson_r": 0.9857512514128817,
      "pearson_p": 0.10759708716882692,
      "spearman_r": 1.0,
      "spearman_p": 0.0,
      "robust_slope": 0.025657189620846354,
      "robust_intercept": -0.01481360305843278
    },
    "subject_fidelity": {
      "pearson_r": 0.94818550810575,
      "pearson_p": 0.2058325449799184,
      "spearman_r": 1.0,
      "spearman_p": 0.0,
      "robust_slope": 0.0911199002565794,
      "robust_intercept": -0.008970465979108997
    }
  }
}
```
