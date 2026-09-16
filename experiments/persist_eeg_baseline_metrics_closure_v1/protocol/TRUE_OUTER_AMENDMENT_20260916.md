# True-outer cohort and incomplete-cell amendment

Recorded before any new frozen inference in this closure continuation, following explicit user approval on 2026-09-16.

1. WBCIC final evaluation uses exactly `sub-4, sub-8, sub-10, sub-15, sub-20, sub-39, sub-40, sub-43, sub-46, sub-51` across the prescribed sessions. The disjoint V8 internal cohort is diagnostic only and must not enter a final metric or regression target.
2. The ModernTCN and Medformer WBCIC expected BA/F1 values in the original prompt were computed on the V8 internal cohort and are retired **only as true-outer regression targets**. They remain historical results. No replacement numerical target is invented. For true outer, freeze the cohort, session mapping, 15 checkpoint identities and normalizers first; then compare new frozen inference against independently persisted true-outer predictions or session results where available, and record whether an independent regression is possible. Do not call a first calculation an independent regression pass.
3. OpenBMI future-session regression targets, all other scientific definitions, exact checkpoints, preprocessing, session notation, no-training restriction, and no heldout-informed selection remain unchanged.
4. LiteBN cells with absent prescribed checkpoints remain `INCOMPLETE`; do not train, substitute checkpoints, infer three-seed means from fewer checkpoints, or silently place such values in the complete primary comparison.
5. Any final table must retain explicit completion/provenance flags. If a mandatory independent true-outer regression or another acceptance gate cannot be met, report that gate as unresolved and label the affected result provisional rather than claiming manuscript-ready closure.
