# WBCIC fold-1 forensic diagnosis

The inherited PMG-fast result was WBCIC fold-1 discovery BA 0.501250 (M2), versus M0 0.784375 and M1 0.771250. Because the PMG-fast runtime contains no M2 checkpoint, optimizer state, or step log, this branch performed exactly one deterministic forensic reproduction with the original locked recipe. No scientific repair was applied.

- rows logged: 83 (every 10 optimizer steps plus epoch-final steps)
- final predicted class-1 fraction on fixed model-fit probe: 0.480000
- fraction of logged steps with clip activated: 0.265060
- median 0.5*||g_harm||/||g_A||: 0.250919
- maximum BatchNorm running-stat displacement: 0.1401416
- embedding variance first/last: 0.98351485 / 0.98711997

Primary diagnosis: `D_batchnorm_running_stat_drift`.

This is a localization statement for the known fold-1 collapse, not a method claim and not a permission to modify PMG.
