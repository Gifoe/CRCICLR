# Split and episode report

All split units are subjects. Five deterministic seeds were generated, and the validator confirmed exact subject coverage and pairwise role disjointness. CAP calibration uses deterministic proportional stratification over the pathology prefixes available in official record names; every represented pathology contributes at least one calibration subject when capacity permits.

| dataset | seed | role counts |
| --- | --- | --- |
| eegmmidb | 0 | {"conformal_calibration": 15, "final_test": 19, "meta_risk_train": 30, "task_head_train": 45} |
| eegmmidb | 1 | {"conformal_calibration": 15, "final_test": 19, "meta_risk_train": 30, "task_head_train": 45} |
| eegmmidb | 2 | {"conformal_calibration": 15, "final_test": 19, "meta_risk_train": 30, "task_head_train": 45} |
| eegmmidb | 3 | {"conformal_calibration": 15, "final_test": 19, "meta_risk_train": 30, "task_head_train": 45} |
| eegmmidb | 4 | {"conformal_calibration": 15, "final_test": 19, "meta_risk_train": 30, "task_head_train": 45} |
| hmc | 0 | {"conformal_calibration": 20, "final_test": 26, "meta_risk_train": 35, "task_head_train": 70} |
| hmc | 1 | {"conformal_calibration": 20, "final_test": 26, "meta_risk_train": 35, "task_head_train": 70} |
| hmc | 2 | {"conformal_calibration": 20, "final_test": 26, "meta_risk_train": 35, "task_head_train": 70} |
| hmc | 3 | {"conformal_calibration": 20, "final_test": 26, "meta_risk_train": 35, "task_head_train": 70} |
| hmc | 4 | {"conformal_calibration": 20, "final_test": 26, "meta_risk_train": 35, "task_head_train": 70} |
| cap | 0 | {"external_final_test": 78, "target_site_calibration": 25} |
| cap | 1 | {"external_final_test": 78, "target_site_calibration": 25} |
| cap | 2 | {"external_final_test": 78, "target_site_calibration": 25} |
| cap | 3 | {"external_final_test": 78, "target_site_calibration": 25} |
| cap | 4 | {"external_final_test": 78, "target_site_calibration": 25} |

| dataset | seed | episodes | excluded | context range | future range |
| --- | --- | --- | --- | --- | --- |
| eegmmidb | 0 | 109 | 0 | 24–38 | 48–76 |
| eegmmidb | 1 | 109 | 0 | 24–38 | 48–76 |
| eegmmidb | 2 | 109 | 0 | 24–38 | 48–76 |
| eegmmidb | 3 | 109 | 0 | 24–38 | 48–76 |
| eegmmidb | 4 | 109 | 0 | 24–38 | 48–76 |
| hmc | 0 | 151 | 0 | 180–180 | 290–1131 |
| hmc | 1 | 151 | 0 | 180–180 | 290–1131 |
| hmc | 2 | 151 | 0 | 180–180 | 290–1131 |
| hmc | 3 | 151 | 0 | 180–180 | 290–1131 |
| hmc | 4 | 151 | 0 | 180–180 | 290–1131 |
| cap | 0 | 103 | 0 | 175–180 | 248–1540 |
| cap | 1 | 103 | 0 | 175–180 | 248–1540 |
| cap | 2 | 103 | 0 | 175–180 | 248–1540 |
| cap | 3 | 103 | 0 | 175–180 | 248–1540 |
| cap | 4 | 103 | 0 | 175–180 | 248–1540 |

Leakage validation: 0 failures. Artifact validation: `valid=True` with 0 total failures. Sleep context uses the first 90 minutes of clock time from the first valid epoch; future begins at the boundary. MI context runs are 4/6 and future runs are 8/10/12/14.
