# Previous PDA forensic audit (development-known source only)

This is a forensic audit of the already committed PERSIST-PDA source outcome. It does not reinterpret the prior negative result and does not open WBCIC S2, EEGNeX future outcomes, outer, or sealed resources.

- Full-PDA minus population future BA and ordinary-adapter minus population are in `results/PREVIOUS_PDA_UTILITY_PREDICTIVENESS.csv`.
- Historical cross-fit diagnostic versus future full-PDA gain: Pearson `+0.3393`, Spearman `+0.2428`, AUROC for future gain > 0 `+0.5722` (n=243).
- Historical-positive mean future gain: `-0.0077`; historical-negative mean: `-0.0314`.
- Persistent/transient norms, ratios, correct-vs-wrong and correct-vs-shuffled differences are included per subject in the CSV.

The old PDA always reused a subject adapter despite this mixed/negative transfer evidence. U-PDA treats that evidence as a reason to certify or fall back to population, not as evidence of universal personalization.
