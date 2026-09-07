# PERSIST-EEG final closure report

## Terminal

`PUD_AUX_CONSTRUCTIVE_HYPOTHESIS_NOT_SUPPORTED`

Phase A authorized the one preregistered Phase-B family (A1–A5 passed). The complete 5-fold × 3-seed × 5-control × 3-lambda training run finished on the server. A post-training B0 lookup failure was repaired from the frozen legal `replay_per_subject.csv`; no model was retrained and no outcome label was used for lambda selection.

## Q1–Q18

**Q1 — Architecture factorization.** Dual task-only versus Vanilla was `−0.008500` BA (paired 95% CI `[-0.020250, 0.002250]`): a modest but real architecture/protocol tax, not the dominant PUD failure.

**Q2 — PUD supervision.** PUD source-only versus Dual was `−0.021167` BA (95% CI `[-0.029583, −0.013167]`), the largest isolated tax.

**Q3 — Target adaptation.** Adaptation helped PUD: `+0.007417` BA (95% CI `[+0.003167, +0.011752]`); it was not the primary failure.

**Q4 — Consequence to generalization.** Greater protected erasure consequence did not reliably predict better PUD future-session BA. The subject-level consequence/generalization audit is diagnostic and clustered by subject; it does not support using intervention harm as a utility proxy.

**Q5 — Source certificate transfer.** The frozen certificate transfer audit is recorded as `CERTIFICATE_TRANSFER_SUPPORTED`, but transfer is selective: source U versus outcome CE harm was Pearson `−0.249`/Spearman `−0.295`; source D versus outcome D was Pearson `−0.332`/Spearman `−0.302`; source P versus outcome BA harm was near zero (`0.052`/`0.033`). This does not establish universal transfer.

**Q6 — Functional persistence.** PUD contribution S1/S2 correlation was `0.594` with RMS change `0.2553`; persistent coordinates were not automatically stable decision contributions.

**Q7 — Brittle bottleneck.** Supported. PUD reliance concentration `R_P=0.683` versus dual control `0.503`; protected erase harm was approximately `+0.1356` versus dual `+0.0749`, consistent with concentrated reliance.

**Q8 — Gradient conflict.** Partial support. Frozen gradient diagnostics included negative task-vs-persistence cosines on some batches (e.g. `−0.132`, `−0.281`) with state hashes unchanged; this is diagnostic evidence, not proof of a training-time causal mechanism.

**Q9 — Calibration/margin versus representation/optimization.** Calibration alone was not sufficient to explain the BA loss. The strongest supported explanation is brittle hard factorization plus PUD supervision; calibration is not the primary diagnosis.

**Q10 — P versus P+U/D.** The frozen component table does not show monotonic utility from adding U/D; P-only/P+U/P+D/PUD are mechanistic ablations, not a license for post-hoc basis selection.

**Q11 — Best-supported explanation.** PUD is learned and task-consequential, but hard factorization concentrates reliance and its functional consequence is not stably useful across future sessions. This is the single primary diagnosis.

**Q12 — Was PUD-Aux authorized?** Yes. Phase-B authorization JSON has A1–A5 all true and both restricted-data flags false.

**Q13 — PUD-Aux versus Vanilla.** PUD-Aux BA `0.776833`; Vanilla BA `0.786167`; delta `−0.009333` (paired subject-bootstrap 95% CI `[-0.017250, −0.001333]`).

**Q14 — Random/Identity controls.** Random-Aux `0.776083`; Identity-Aux `0.771583`. PUD-Aux is above both controls, so G5 passes, but this does not rescue the negative primary comparison.

**Q15 — Full-teacher KD.** Full-Teacher-KD-Aux `0.779167`; PUD-Aux is `−0.002333` below it, within the G6 tolerance of `0.0025` (G6 passes). A strong PUD-specific improvement claim is not supported.

**Q16 — WBCIC transfer.** Not run. The OpenBMI gate failed, so external development and sealed outer access were correctly blocked.

**Q17 — Sealed holdouts.** OpenBMI internal 14-subject holdout: **NO**. WBCIC sealed outer: **NO**.

**Q18 — Strongest defensible claim.** Task-consequential persistent structure can be identified reliably in a trained EEG representation, but intervention consequence does not imply future-session generalization utility. Hard factorization concentrates reliance without improving generalization, and this final soft single-path PUD auxiliary formulation also did not produce a better predictor.

## PUD-Aux gates

| Gate | Result |
|---|---|
| G1 delta ≥ +0.005 | FAIL (`−0.009333`) |
| G2 CI lower > 0 | FAIL |
| G3 ≥4/5 positive folds | FAIL (`0/5`) |
| G4 ≥2/3 positive seeds | FAIL (`1/3`) |
| G5 > Random and Identity | PASS |
| G6 not >0.0025 below Full-KD | PASS |
| G7 purity/integrity | PASS |

Because G1 fails, no PUD-Aux V2/V3 or other constructive family was run.

## Reproducibility note

The engineering repair is recorded in `ENGINEERING_REPAIR_LOG.md` and in `code/closure.py`: if `source_only_raw.csv` lacks B0, the code reads the frozen `replay_per_subject.csv` B0 rows. The repaired post-processing produced 720 subject rows (6 methods × 5 folds × 3 seeds × 8 outcome subjects), zero duplicate keys, and exactly 8 subjects per method/fold/seed.
