# Alternative explanations audit

Alternative explanations are evaluated against frozen comparisons; no new model is introduced in Phase A.

- Parameter/capacity: B0 and B1 are single-path EEGNet references; dual PUD is lower (not higher) in frozen BA despite its two paths.
- Dual-path architecture tax: A2−Vanilla = -0.0085 BA.
- Stronger baseline: B1−Vanilla = 0.0053 BA.
- Adaptation: PUD-after-adaptation−PUD-source = 0.0074 BA; adaptation helps partially, so it is not the primary failure.
- Random/identity controls: Random and identity source-only remain above PUD but below/near Vanilla, rejecting a purely random branch artifact.
- Leakage/normalization/seed/fold: see HOLDOUT_PURITY_AUDIT.md and canonical provenance; all frozen integrity flags are required false for holdout/outer access.
- Teacher quality: teacher outcome BA is retained in canonical raw rows; a teacher-quality-only explanation is not accepted without a positive interaction, and the global PUD damage persists in the frozen aggregate.
