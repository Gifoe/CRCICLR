# Compact-TG Stage-1 seed0 relational-geometry decision

Representation-only audit of frozen Stage-1 checkpoints. No retraining, new cohort, holdout access, or cross-fold latent arithmetic. Artifact hashes, frozen normalizers, session mapping, and fold-local coordinate audit passed.

| Metric | OpenBMI | WBCIC |
|---|---:|---:|
| Absolute centroid alignment (ERM) | 0.7438 | 0.8020 |
| Direction consensus (ERM) | 0.8692 | 0.8445 |
| Source to future direction cosine (ERM) | 0.9227 | 0.9162 |
| Direction null p (ERM) | <1e-4 | <1e-4 |
| Session persistence (ERM) | 0.9044 | 0.8847 (early to S3) |
| Offset / separation ratio (ERM) | 0.6577 | 1.0016 |
| Absolute prototype BA (ERM) | 77.65% | 77.44% |
| Centered prototype BA (ERM) | 79.18% | 78.52% |
| Centered improvement (ERM) | +1.53 pp | +1.08 pp |
| TG minus ERM absolute alignment | +0.0361 | +0.0285 |
| TG minus ERM Fisher change | +0.1132 | +0.1799 |
| TG minus ERM margin change | -0.1660 | -0.2698 |
| TG minus ERM class separation change | +0.8087 | +0.9434 |

H1 discriminative-direction transfer: PASS. Both datasets have positive observed transfer, p<0.05, and 3/3 positive fold means.

H2 absolute-location mismatch: PASS. WBCIC unlabeled centering improves ERM source-prototype BA by +1.08 pp overall; all 3 folds are positive.

H3 alignment-discrimination tradeoff: PASS (narrowly). TG raises absolute alignment in 2/3 WBCIC folds, while future classifier margin is lower in 2/3 folds and -0.2708 overall (95% subject bootstrap CI [-0.4691, -0.0700]). Fisher and class separation increase, so this is a margin tradeoff, not a blanket loss of discriminability.

Failure diagnosis: Compact-TG did not fail because relational geometry is absent. Directions transfer strongly and are far outside the sign-flip null. The failure is that TG optimizes absolute alignment while degrading the frozen classifier margin; on WBCIC this yields Compact-TG BA 75.62% versus Compact-ERM 77.44% (-1.82 pp), and centered-prototype BA remains 2.10 pp below ERM. The evidence supports the limited statement align relations, not absolute locations, but does not justify a new rescue method in this audit.

Protocol invalidity: NO.

Terminal: RELATIONAL_GEOMETRY_SUPPORTED_STOP
