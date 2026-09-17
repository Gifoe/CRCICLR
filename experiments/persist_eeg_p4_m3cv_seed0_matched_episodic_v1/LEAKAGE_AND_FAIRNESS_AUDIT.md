# Leakage and fairness audit

- Eligibility comes exclusively from pre-training trial availability. Its two exclusions are fixed before models or results exist.
- Splits are frozen before optimizer steps; all outer partitions are subject-disjoint and each eligible biological subject appears once in outer test.
- Episode supports/queries are subject-disjoint and inner-train only. The identical frozen fold manifest feeds EEGNet and SIRE.
- Normalization pools all valid S1 epochs from inner train, matching the authoritative pooling behavior. This intentionally weights subjects in proportion to valid S1 trial count; it never uses S2, validation or outer signals.
- Selection consumes only mean inner-val S2 subject-equal BA. Outer labels are never used for training, normalization, checkpoint selection or model choices.
- Outer metrics are subject-equal aggregates, while each subject's metric uses all of its valid cached trials; the 8-trial rule applies only within episodes.

## Fold normalizer provenance
- Fold 0: `b72356cf5c583f8e9d2e6d9871d32757ffc5ee1ccf3e375bcce53493b171f37a`; 10250 pooled S1 epochs.
- Fold 1: `bf879ac3499ff1ca787906b6652cd29eb4671aec0cd16e6a94c8f8b2133598fe`; 10265 pooled S1 epochs.
- Fold 2: `2376a87a0522f7df70e2c258b5afb82bdecd5659564da134d091eca4a9ebb131`; 10260 pooled S1 epochs.
- Fold 3: `057fced0e08c513a64a976c72abea0b5795f70947313ef7c132fd02ab90e4fc1`; 10408 pooled S1 epochs.
- Fold 4: `c862cb3129a05fab0dc2030846e7b8241440fe4813586fe7cc44fc680b838ff8`; 10428 pooled S1 epochs.
