# Mathematical audit

"
        "For gradient descent θ←θ−lr·g, an update is first-order descent for uniform risk when dot(g_update,g_uniform)>0. Let r=g_risk−g_uniform. If dot(r,g_uniform)<0, the implemented projection r−dot(r,g_uniform)/(||g_uniform||²+1e−12)·g_uniform removes the conflicting component; otherwise r is unchanged. Hence dot(g_update,g_uniform)=||g_uniform||²+β·dot(r_projected,g_uniform)≥0 (up to numerical tolerance). A deterministic quadratic toy test is run before training and stored in `results/MATH_TOY_TEST.json`. The formula is valid for binary and multi-class CE because it operates on flattened parameter gradients, not labels or logits dimensions.

"
        "Subject/session losses average class means and then sessions, preventing trial-count dominance. Risk ratios are clipped and renormalized with a bounded water-fill so every source subject has positive mass. SVD signs use the largest-magnitude right-vector element. No target subject, target label, target BN, or target adaptation is used.
