# Frozen feature definitions

For binary logits, the correct-class margin is `z[y] - z[1-y]`. This avoids the degeneracy of a cosine between one-dimensional centered binary-logit summaries.

- Adaptation-effect stability: trial effect is adapted minus anchor correct-class margin. Within each class, measure the absolute S1-validation-to-S2 mean shift divided by pooled SD; average classes and negate. Higher is more stable.
- Decision stability: apply the same class-conditioned standardized-shift definition to frozen-anchor correct-class margins. Higher is more stable.
- Certificate precision: within each seed, run a 2,000-resample paired class-stratified S2 bootstrap of balanced-accuracy Delta2. Record SE, `Delta2/SE`, and the fixed 90% one-sided LCB.
- Representation stability: use only the frozen anchor final embedding. Within class, normalize the S1-validation-to-S2 centroid distance by pooled within-session RMS radius; average classes and negate. Higher is more stable.
- Simple controls: raw Delta2, S1 head parameter-relative change, and S1-validation anchor confidence.
- Identity: unavailable as a legal target-subject frozen scalar. No new identity model is fit.

Every scalar is computed per seed and then averaged over the three matched seeds within subject/backbone. No feature uses S3.

