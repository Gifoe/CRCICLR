# PERSIST-EEG WBCIC EEGNet actionability report

Terminal state: `WBCIC_AUDIT_NO_ACTIONABLE_HARMFUL`

## Frozen scope

The primary cohort is the 51-subject Yang2025/NEMAR 2C core. Forty-one deterministic development subjects are used for task-only selection and five-fold actionability cross-fitting. Ten outer subjects remain sealed.

Primary session target: `S1+S2 -> S3`. Backbone route: EEGNet only.

## Representation competence

Selected task-only configuration: `EEGNET_STABLE`.

Mean subject BA: 0.7944; 95% subject-bootstrap CI [0.7488, 0.8383]; fraction above chance: 0.976.

## Prospective H1-H5 audit

| Block | H1 | H2 | H3 | H4 | H5 | u_spec | finite ratio | BA specific | Assignment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P01_04 | True | False | True | False | False | 0.05496 | 1.620 | -0.0530 | PROTECTED |
| P05_08 | True | True | False | False | False | -0.00466 | 1.037 | -0.0002 | DECISION-NULL / WEAKLY ACTIVE |
| P09_16 | False | False | False | False | False | 0.00245 | 0.574 | -0.0083 | UNCERTAIN |
| P17_32 | False | False | False | False | False | 0.07791 | 1.385 | -0.0340 | UNCERTAIN |

Decision: `STOP_AGDI_NO_ACTIONABLE_TARGET`

## Interpretation

No prospectively tested WBCIC block passed all H1-H5.

A negative gate is not converted into a positive claim by changing blocks, thresholds, subjects, preprocessing, or backbone. AGDI is run only when H1-H5 jointly authorize a target.

## Reproducibility

- Git commit recorded at report time: `47523e8e96a03755da2c404b626309b25a92c827`
- Seed: `20260817`
- Random controls per block/fold: `100`
- Subject bootstrap draws: `10000`
