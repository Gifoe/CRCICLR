# Data legality audit

Only the frozen OpenBMI manifest and the frozen WBCIC development cache are used. The primary audit indexes model-fit subjects only, with no outcome index construction. It does not open WBCIC sealed outer-10 subjects, any OpenBMI sealed/internal holdout, or any outcome labels. Discovery data are not read by the primary audit and are only permitted for forensic localization of the inherited WBCIC fold-1 collapse; this implementation does not need it. Seed 0 is the only seed.

A/B pseudo-environments are subject-disjoint and use the exact PMG-fast five meta-fold partitions. No target adaptation, task prior, router, ensemble, new split, or scientific coefficient search is present.
