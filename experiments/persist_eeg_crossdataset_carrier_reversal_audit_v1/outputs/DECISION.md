# Cross-dataset carrier reversal audit

POST-HOC DIAGNOSTIC ONLY. All models were frozen and hash-verified; no training, optimizer, or backward pass occurred.

1. Original reversal: OpenBMI EEGNet 0.7986, LiteBN 0.8300, delta +3.143 pp. WBCIC EEGNet 0.7927, LiteBN 0.7582, delta -3.455 pp.
2. WBCIC signal-scale shift larger: **NO**. Mean log-RMS difference WBCIC-minus-OpenBMI is -0.053, CI [-0.1445239571668693, 0.03295370768268272].
3. WBCIC mu evidence less stable: **UNCLEAR**; the mean contrast-shift difference is +0.136 with CI crossing zero.
4. WBCIC beta evidence less stable: **UNCLEAR**; the mean contrast-shift difference is +0.090 with CI crossing zero.
5. WBCIC spatial covariance less stable: **YES**. Normalized covariance shift mean difference is +0.512, CI [0.2762788740575164, 0.8163549123595971].
6. Latent class-direction stability alone explains reversal: **NO**. LiteBN direction cosine is high in both environments (see `REPRESENTATION_SHIFT.csv`), while the sign of BA reversal changes.
7. Separation magnitude/margin is more informative than cosine alone: **UNCLEAR**. It changes in WBCIC but this audit did not establish a cross-model performance association for it.
8. Strongest primary separating perturbation: **WBCIC motor-channel mask**. LiteBN differential sensitivity is -4.182 pp; the matched non-motor differential is -1.091 pp. OpenBMI named motor masking is unavailable because its cache lacks a verified channel-name map.
9. Strongest LiteBN subject association: OpenBMI `mu_contrast_shift` rho=-0.455; WBCIC `cov_fro_shift` rho=-0.560. Both bootstrap intervals are reported in `SUBJECT_ASSOCIATIONS.csv`; they are small-n post-hoc evidence, not proof.
10. Compact and LiteGN compatible evidence: **NO** for the primary WBCIC motor-mask differential; neither reproduces LiteBN's increased vulnerability relative to EEGNet.
11. Strongest supported explanation: a **spatial-covariance / motor-channel reliability clue** for LiteBN on WBCIC: greater population covariance shift, negative per-subject covariance-shift association (rho=-0.560), and stronger motor-mask vulnerability. This is not a causal claim.
12. Strongest remaining alternative: fold-0 checkpoint-selection variance and the small WBCIC inner-validation/outer-development subject counts; the spatial sensitivity does not triangulate across Compact and LiteGN.
13. Enough evidence to constrain a future constructive model: **PARTIAL**. The future principle should test reliability-aware handling of spatial evidence, rather than force global invariance.
14. Final terminal: **REVERSAL_MECHANISM_PARTIALLY_SUPPORTED**.
