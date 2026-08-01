# HSC-TTA EEG CPU phase report

Generated: 2026-08-01T03:49:35.849540+00:00

## Environment and scope

- Host: autodl-container-a6a040afb3-b82784d4
- CPU: Intel(R) Xeon(R) Platinum 8470Q
- Python: 3.11.15 in `hsc_cpu`
- Container memory limit: 2147483648 bytes
- CUDA: disabled for every CPU-stage command (`CUDA_VISIBLE_DEVICES=`)
- Data disk: 350.00 GiB total, 283.28 GiB free
- GPU/foundation model work: not executed

## Downloads and audit

| dataset | manifest files | SHA verified | missing/failed |
| --- | --- | --- | --- |
| eegmmidb | 654 | 654 | 0 |
| hmc | 453 | 453 | 0 |
| cap | 324 | 324 | 0 |

| dataset | recordings | readable | eligible subjects | excluded subjects | raw | processed | audit eligible |
| --- | --- | --- | --- | --- | --- | --- | --- |
| eegmmidb | 654 | 654 | 109 | 0 | 1.55 GiB | 1.40 GiB | 109 |
| hmc | 151 | 151 | 151 | 0 | 15.73 GiB | 5.73 GiB | 151 |
| cap | 108 | 108 | 103 | 5 | 40.11 GiB | 2.17 GiB | 103 |

| dataset | reason | count |
| --- | --- | --- |
| cap | required_channels_missing | 5 |

## Preprocessing

| dataset | complete caches | failures | windows | one-channel caches | size |
| --- | --- | --- | --- | --- | --- |
| eegmmidb | 109 | 0 | 9837 | 0 | 1.40 GiB |
| hmc | 151 | 0 | 137243 | 0 | 5.73 GiB |
| cap | 103 | 0 | 103021 | 102 | 2.17 GiB |

| dataset | windows | mapped label counts |
| --- | --- | --- |
| eegmmidb | 9837 | {0: 2479, 1: 2438, 2: 2465, 3: 2455} |
| hmc | 137243 | {0: 23686, 1: 15548, 2: 50083, 3: 26671, 4: 21255} |
| cap | 103021 | {0: 18615, 1: 4551, 2: 37223, 3: 24559, 4: 18073} |

All eligible subjects have one complete, config-hashed HDF5 cache. Raw data remained read-only. The five excluded CAP records lack both required central-channel alternatives.

## Subject splits and deployment episodes

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

Independent artifact validation found zero split-role overlaps, zero U_s/V_s overlaps, zero sleep boundary violations, and zero MI run-protocol violations. No episode failed the minimum-future rule.

CAP target-site calibration is deterministically proportionally stratified by the pathology prefix encoded in the public record name.

## Statistical core and synthetic validation

Implemented components include prediction sets, empirical-Bernstein-style block bounds, grouped meta-risk prediction, finite-sample simultaneous residual quantiles, deterministic safe action selection, subject-level metrics/bootstrap, three action interfaces, schemas, and simulations A–E.

- Synthetic subjects: 120
- Calibration subjects: 30
- Simultaneous quantile q: 1.0
- Surface coverage: 1.0
- Certified Subject Rate: 0.0

Post-selection comparison:

| method | coverage |
| --- | --- |
| pointwise_proxy | 1.0 |
| simultaneous | 1.0 |

The default synthetic result is conservative but not useful: `q=1.0` and CSR=0. The certificate covers because it saturates, not because the method demonstrates nontrivial certification. With the configured block bound, the additive term `3 log(3/eta)/B` is already above 0.20 for small B; typical future horizons therefore make alpha 0.10/0.20 certification difficult or impossible. This must be resolved theoretically and empirically before claiming a successful method.

## Frozen GPU interface

| file | rows | bytes |
| --- | --- | --- |
| subject_context_features.parquet | 30 | 18516 |
| subject_action_surface.parquet | 1800 | 84948 |
| subject_decisions.parquet | 30 | 11935 |

All rows were validated with the Pydantic schemas. Fields derived from V_s are confined to offline action-surface/decision evaluation; context features are synthetic U_s-only interface rows with the exact future field names.

## Tests

- pytest: 25 passed, one expected conservative-quantile warning
- line coverage: 71%
- artifact validation: valid=True, failures=0, leakage failures=0

Coverage is adequate for the statistical core but weak in real EDF adapters/preprocessing branches; the real full-data run provides integration evidence but is not counted by pytest coverage.

## Completion and known limitations

CPU deliverables are complete: public data download and SHA manifests, unified audit, full preprocessing, five subject-disjoint splits, all deployment episodes, statistical components, simulations, frozen schemas, tests, and reports. Deliberately unfinished work is the prohibited GPU phase: checkpoint acquisition, real embeddings, real task-head/meta-risk training, real action surfaces, and final scientific results.

Known limitations:

1. The current bound/synthetic configuration produces trivial certification (`CSR=0`).
2. Five CAP recordings were excluded for missing required central channels.
3. The AutoDL container is capped at 2 GiB RAM; CAP required per-subject process isolation.
4. Pytest line coverage is 71%, below a strong production target.
5. CAP filenames are treated as recording-level subject IDs because no stronger cross-record identity mapping is available in the public metadata.

Current disk free space is 283.28 GiB, sufficient to proceed to the estimated GPU cache scenarios in `NEXT_GPU_PHASE.md`.
