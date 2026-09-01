# CANONICAL BASELINE STATUS

This report describes the frozen vanilla EEGNet baseline only. It is not a claim about adaptation or utility.

## Protocol

The exact protocol and legality/provenance audits are in `CANONICAL_PROTOCOL.md`, `BASELINE_LEGALITY_AUDIT.md`, `BASELINE_PROVENANCE_AUDIT.md`, and `HISTORICAL_REFERENCE_AUDIT.md`.

## Primary results

Primary = average the three seed probabilities per trial, then compute biological-subject BA; CI = 10,000-draw subject bootstrap.

| dataset | mean subject BA | median | SD | 95% CI | subjects |
|---|---:|---:|---:|---|---:|
| OpenBMI | 0.827963 | 0.820000 | 0.096903 | [0.801481, 0.853519] | 54 |
| WBCIC | 0.797503 | 0.840000 | 0.143351 | [0.753786, 0.839149] | 41 |

## Secondary seed robustness

The secondary statistic is the mean of the three single-seed aggregate subject BAs; it is not substituted for the primary summary.

| dataset | mean single-seed BA | mean accuracy | mean NLL |
|---|---:|---:|---:|
| OpenBMI | 0.810247 | 0.810247 | 0.434412 |
| WBCIC | 0.769821 | 0.769827 | 0.442441 |

## Fold and seed means

- OpenBMI fold means: [82.18%, 83.36%, 81.18%, 82.45%, 85.00%]
- OpenBMI seed means: [81.91%, 79.19%, 81.98%]
- WBCIC fold means: [81.67%, 79.25%, 79.88%, 78.85%, 78.88%]
- WBCIC seed means: [78.63%, 78.40%, 73.92%]

## Scope conclusion

The OpenBMI analysis covers all 54 Stage-0-frozen Lee2019 MI subjects. The WBCIC analysis covers only the 41 frozen development subjects. The WBCIC sealed outer 10 and any OpenBMI sealed/internal holdout were not accessed. Runtime/checkpoints/cache/raw EEG are not deliverables.

## Required validity answers

1. OpenBMI is raw EEGNet evaluation rather than a frozen historical embedding plus sklearn head: YES.
2. WBCIC uses exactly the same outer evaluation logic (frozen subject roles, discovery-only epoch selection, one final future-session score): YES.
3. All non-outcome subjects were used in the final refit: YES.
4. Outcome future labels were excluded from all fitting and selection: YES.
5. Outcome-subject history was excluded from vanilla adaptation: YES; no adaptation is performed.
6. The WBCIC sealed outer 10 were untouched: YES.
7. Trial predictions use the frozen outcome trials and can be used for direct paired comparison with Ours: YES.

terminal = CANONICAL_EEGNET_BASELINE_ESTABLISHED
