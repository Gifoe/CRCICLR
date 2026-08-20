# PERSIST-EEG Final Constructive Model V6

V6 is a one-seed exploratory development study of representation-level subject
adaptation. The terminal state is `V6_OPENBMI_TARGET_ONLY`; it is not a dual-
benchmark PERSIST-SA success and is not ready for WBCIC outer evaluation.

## Final results

| Benchmark | EEGNet reference | Strong pre-V6 matched anchor | Best legal V6 result | Delta vs EEGNet (95% subject-bootstrap CI) | Delta vs matched anchor |
|---|---:|---:|---:|---:|---:|
| OpenBMI S1 -> S2, 54 subjects | 75.30 | 76.63 | **83.20** | **+7.91 pp** `[+5.63, +10.31]` | **+6.57 pp** `[+4.28, +8.94]` |
| WBCIC S1/S2 -> S3, 41 development subjects | 79.44 | 81.78 | **82.08** | **+2.65 pp** `[+1.68, +3.71]` | +0.30 pp `[-0.42, +1.12]` |

The relaxed +5 pp-over-EEGNet goal was reached only on OpenBMI. The OpenBMI
gain came from the generic MI-specific backbone and legal S1 adaptation, not
from PERSIST: its best PERSIST candidate was 1.19 pp below the strongest generic
control (95% CI `[-2.19, -0.19]`). On WBCIC, the best PERSIST candidate was
0.06 pp below the strongest generic control (95% CI `[-0.71, +0.45]`).

## Protocol

- OpenBMI: Session 1 labels are target history; Session 2 is scoring-only.
- WBCIC: Sessions 1/2 labels are target history; Session 3 is scoring-only.
- Five subject-disjoint outer folds contain model-fit, discovery, and outcome
  roles. Outcome future labels never select architectures, adapters, gates, or
  fusion rules.
- All reported estimates use seed `20260820`; no best-seed selection was used.
- `OUTER_TEST_USED=false`: the sealed WBCIC outer cohort was not opened,
  enumerated, featurized, or scored.

## What was tested

The retained audit includes matched linear/last-layer/prototype controls,
conditional affine and bilinear adapters, FBCSP, encoder fine-tuning, paired
Fisher protection controls, future-session population training, enlarged
MI-specific EEGNet candidates, fixed fusion, and history-only selective heads.
The synthetic positive control validates that the selective mechanism can
detect a constructed protected/adaptable/harmful decomposition, but it is not
evidence of benefit on real EEG.

## Reproduce the terminal evaluation

Run from the repository root after the frozen caches and five backbone
checkpoints have been generated:

```powershell
python experiments/persist_eeg_final_model_v6/code/backbones/recover_openbmi_mi_outcomes.py
python experiments/persist_eeg_final_model_v6/code/backbones/recover_wbcic_large_outcomes.py
python experiments/persist_eeg_final_model_v6/code/adapters/run_openbmi_selective_head.py
python experiments/persist_eeg_final_model_v6/code/adapters/run_wbcic_selective_head.py
python experiments/persist_eeg_final_model_v6/code/evaluation/finalize_v6.py
```

The authoritative decision is in `outputs/FINAL_DECISION.json`; detailed paired
statistics are in `outputs/final_candidate/DEVELOPMENT_RESULTS.json`, and
coverage/leakage checks are in
`outputs/protocol/FINAL_PREDICTION_INTEGRITY_AUDIT.json`.
