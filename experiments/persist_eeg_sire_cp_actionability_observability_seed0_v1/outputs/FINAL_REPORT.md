# SIRE-EEG C-direction actionability and observability

This report uses only inner-train and discovery EEG arrays. The canonical seed-zero neural checkpoints and BN states were frozen. Historical source-audit records disclose earlier final-heldout diagnostic exposure; this run accessed zero final-heldout EEG arrays.

## Q1. Does SIRE have structured C-direction oracle headroom?

Oracle-1dir and sequential Oracle-top3 balanced-accuracy gains are reported below with biological-subject 20,000-draw bootstrap 95% CIs. The random-direction oracle uses the same source-complement rank.

| Task | Baseline BA | Oracle-1 ΔBA [95% CI] | Oracle-top3 ΔBA [95% CI] | Random ΔBA [95% CI] | Structured−random [95% CI] |
|---|---:|---:|---:|---:|---:|
| OpenBMI_MI | 0.7737 | +0.0261 [+0.0168, +0.0380] | +0.0310 [+0.0205, +0.0443] | +0.0007 [+0.0001, +0.0013] | +0.0303 [+0.0199, +0.0439] |
| OpenBMI_SSVEP | 0.9201 | +0.0139 [+0.0082, +0.0202] | +0.0167 [+0.0097, +0.0243] | +0.0000 [+0.0000, +0.0000] | +0.0167 [+0.0096, +0.0243] |

## Q2. Is optimal routing trial-conditional?

The locked criterion counts a subject/session/direction profile when at least 10% of its trials choose α<1 and at least 10% choose α>1. The fraction of discovery profiles meeting that rule is in `EEGNET_SIRE_OBSERVABILITY_COMPARISON.csv`; per-profile distributions are in `TRIAL_CONDITIONALITY.csv`.

## Q3. How much oracle headroom does the same label-free router recover?

The diagnostic router is fit on inner-train subjects with grouped OOF CV and evaluated on disjoint discovery subjects. It uses the same logistic-regression family and action labels as the EEGNet audit, with deterministic train-only PCA compression to the EEGNet feature budget when needed. Per-task gain, recovery fraction, and subject-bootstrap intervals are in `ROUTER_PERFORMANCE.csv`.

## Q4. Did actionability observability improve versus EEGNet?

| Task | Backbone | Baseline BA | Oracle1 ΔBA | Oracle3 ΔBA | Random ΔBA | Conditional profiles | Router action BA | Router ΔBA | Oracle recovered |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenBMI_MI | EEGNet | 0.801 | 0.040 | 0.053 | 0.003 | 0.998 | 0.600 | 0.001 | 0.010 |
| OpenBMI_MI | SIRE-EEG | 0.774 | 0.026 | 0.031 | 0.001 | 0.997 | 0.342 | -0.001 | -0.046 |
| OpenBMI_SSVEP | EEGNet | 0.965 | 0.010 | 0.011 | 0.001 | 0.992 | 0.553 | -0.001 | -0.051 |
| OpenBMI_SSVEP | SIRE-EEG | 0.920 | 0.014 | 0.017 | 0.000 | 0.999 | 0.543 | -0.000 | -0.028 |

## Interpretation

- **OpenBMI_MI:** `STRUCTURED_BUT_UNOBSERVABLE_SIRE_ACTIONABILITY`; point-estimate headroom recovery is -0.046 for SIRE-EEG versus 0.010 for EEGNet.
- **OpenBMI_SSVEP:** `STRUCTURED_BUT_UNOBSERVABLE_SIRE_ACTIONABILITY`; point-estimate headroom recovery is -0.028 for SIRE-EEG versus -0.051 for EEGNet.

The comparison uses the frozen prior EEGNet output CSVs without changing them. Checkpoint provenance and historical heldout diagnostic exposure are disclosed in `SIRE_SOURCE_AUDIT.json`; current-run heldout exclusion receipts are in `FINAL_HELDOUT_EXCLUSION_AUDIT.json`. Current performance CIs use biological-subject bootstrap, and this development-only audit does not estimate final-heldout performance.
