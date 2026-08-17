# PERSIST-EEG External Actionability Audit V1

Terminal state: `EXTERNAL_AUDIT_NO_ACTIONABLE_HARMFUL`

Next action: `STOP_AGDI_NO_ACTIONABLE_TARGET`

## Scope

The prospective external dataset is PhysioNet EEGMMIDB v1.0.0: 109 subjects, 64 channels, 160 Hz, and six motor-imagery runs. The audit used 45 task-head subjects, 30 block-discovery subjects, and 15 confirmatory calibration subjects. The 19-subject outer test remained locked and was not materialized.

This is a repeated-run motor-imagery replication. It is not evidence for true multisession, multisite, or multidevice persistence.

The frozen task head is weak: subject-mean validation BA on context runs was 0.346, and mean baseline BA on confirmatory future runs was 0.286 (four-class chance = 0.250). This materially limits the strength of a negative actionability conclusion.

## Frozen DDA interpretation

DDA-A remains permanently failed and its behavioral-null explanation remains falsified. DDA-B and DDA-C authorized this prospective external audit; they did not authorize AGDI by themselves.

## Confirmatory block results

| Block | H1 | H2 | H3 | H4 | H5 | u_spec | finite ratio | BA specific | Assignment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P01_04 | False | True | True | False | False | -0.230991 [-0.363592, -0.112184] | 2.571 [2.068, 3.093] | -0.0028 [-0.0241, 0.0185] | UNCERTAIN |
| P05_08 | False | False | True | False | True | -0.031940 [-0.078278, 0.003370] | 2.687 [2.474, 2.875] | 0.0230 [-0.0023, 0.0503] | UNCERTAIN |
| P09_16 | False | False | True | False | False | 0.016785 [-0.000684, 0.033758] | 1.795 [1.610, 1.966] | 0.0035 [-0.0149, 0.0218] | UNCERTAIN |
| P17_32 | False | False | False | False | False | 0.022202 [0.010551, 0.034715] | 0.925 [0.796, 1.057] | -0.0096 [-0.0180, -0.0008] | UNCERTAIN |

## Supporting PERSIST-CF rescue/harm decomposition

Frozen interpretation: `CASE_1_RESCUE_APPROX_HARM`. This analysis is not an authorization gate and does not change `DDA_A_FAIL`.

CF rescue rate was 0.0085 and harm rate was 0.0092; net rescue was -0.0007 (95% cluster-bootstrap CI [-0.0019, 0.0007]).

Relative to exact matched-random offsets, net rescue changed by -0.0003 (95% CI [-0.0009, 0.0003]).

## Limitations

The confirmatory sample has 15 subjects, giving wide intervals. EEGMMIDB has repeated runs rather than independent sessions/sites/devices. Official sampling-rate and trial-count anomalies were retained under the recorded pre-outcome data amendment. Most importantly, the task head is only modestly above chance. Therefore the terminal state means that this frozen audit did not identify an actionable target; it does not establish that no actionable persistent structure could exist under a stronger frozen representation.

## Conclusion

No real block passed all H1-H5. AGDI is not authorized; constructive model search stops at this falsifiable negative boundary.

All confidence intervals and sign-flip tests use subjects as the inference unit. Holm correction was applied separately within each gate family across the four pre-registered blocks.
