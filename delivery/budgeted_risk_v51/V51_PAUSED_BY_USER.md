# V5.1 partial checkpoint — paused by user

Status: `USER_PAUSED_AFTER_RAW_DIAGNOSTIC`

The user requested termination while S2/S3/S4 were running. Those processes were stopped and their incomplete in-memory results were discarded. This is not a completed V5.1 decision package and must not be cited as a full S1–S4 comparison.

Completed and saved: repository/input audit, protocol freeze, ordinal ancestry audit, protected-cohort audit, exact restoration of the hash-locked S1 output, independent S1 index equivalence, raw predictor diagnostic, and subject-level S1 summaries. Current state remains `RAW_DIAGNOSTIC_COMPLETE`.

S1 reproduction matched 65,100/65,100 rows. Official continuous and index mismatches are zero. The independently refitted ordinal raw prediction drifted by at most 0.000260909, without changing any certified index.

## Saved temporal results

| dataset | requested_budget | raw_spearman | raw_mae_improvement | raw_gain | raw_gain_ci_low | calibrated_violation_mean | calibrated_gain | calibrated_gain_ci_low | calibrated_oracle_recovery | sentinel_delta | sentinel_transition_rate | q_mean | raw_gate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eegmmidb | 5 | 0.2330 | 0.5333 | -0.6144 | -1.2330 | 0.0338 | -0.0220 | -0.1399 | -0.1566 | 0.6246 | 0.6554 | 8.6453 | False |
| eegmmidb | 10 | 0.4558 | 0.6907 | -0.7235 | -1.3976 | 0.0369 | -0.0199 | -0.1221 | -0.1413 | 0.4738 | 0.4923 | 6.0080 | False |
| eegmmidb | 20 | 0.6057 | 0.7592 | -0.7443 | -1.4305 | 0.0615 | 0.0126 | -0.0183 | 0.0899 | 0.2769 | 0.2985 | 3.6075 | False |
| eegmmidb | 50 | 0.6350 | 0.7751 | -0.7720 | -1.4772 | 0.0308 | 0.0027 | -0.0520 | 0.0195 | 0.2738 | 0.2769 | 3.2543 | False |
| hmc | 5 | 0.4178 | 0.0795 | -0.0071 | -0.0910 | 0.0489 | 0.0302 | -0.1411 | 0.0768 | 0.0978 | 0.2711 | 5.0428 | False |
| hmc | 10 | 0.3474 | 0.0873 | -0.0298 | -0.1445 | 0.0489 | 0.0109 | -0.2161 | 0.0277 | 0.1044 | 0.2756 | 5.0586 | False |
| hmc | 20 | 0.4915 | 0.1321 | -0.0403 | -0.1979 | 0.0444 | 0.0185 | -0.1858 | 0.0471 | 0.1311 | 0.2933 | 5.1619 | False |
| hmc | 50 | 0.5444 | 0.1529 | -0.0470 | -0.2219 | 0.0511 | 0.0780 | 0.0145 | 0.1982 | 0.0800 | 0.2400 | 4.8090 | False |

The frozen raw gate fails for both datasets at every budget <=20 because raw set-size gain and its subject-bootstrap lower bound are negative. Therefore the provisional gate read is `V51_STOP_RAW_PREDICTOR_FAILURE`; S4 could have provided mechanism diagnostics but could not override the earlier raw gate. This is a provisional inference, not the missing full V5.1 comparison.

Not completed: S2 exact two-fold calibration, S3 scaled exact calibration, S4 pooled cross-fit diagnostic, calibration/evaluation LOO analyses, final figures, and the formal V5.1 verdict file.

Formal calibration was not opened. Internal final was not opened. CAP was not opened. Active acquisition was not run. The full method stage was not entered.
