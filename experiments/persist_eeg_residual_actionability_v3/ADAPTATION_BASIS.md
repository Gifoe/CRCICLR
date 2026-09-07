# V3 finite adaptation basis

The M1-M5 code was frozen at server commit
`3f40aa2fbf1e399fa4b310f40c8be0660e0635cb` before grouped OOF outcomes were
read. That run passed and terminated as
`RESIDUAL_ACTION_SIGNAL_BUT_NO_NET_GAIN`.

An audit then identified that learnability computed over all action rows was
inflated by `action_boundary_cross`, because rescue and harm are impossible on
non-crossing rows. Conditional held-out diagnostics on crossing candidates
showed:

- AMPLIFY AUROC approximately 0.44-0.49;
- GEOMETRY AUROC approximately 0.51-0.58;
- ERASE AUROC approximately 0.69-0.72;
- ERASE rescue prevalence 0.245 and harm prevalence 0.755.

This supports exactly one finite adaptive formulation: train direct
action-specific rescue-versus-harm heads on boundary-cross candidates and let
inner calibration choose among KEEP-only, FULL, protected-safe, and ERASE-only
menus. I006 uses regularized logistic regression and I007 uses a small
depth-controlled histogram gradient booster. The outer folds, feature legality,
success criteria, and sealed-outer prohibition are unchanged.

This adaptation was chosen after M1-M5 OOF inspection. Its confidence intervals
are descriptive development estimates, not fresh confirmation. No MLP,
multi-seed search, arbitrary expert subsets, or WBCIC outer access is allowed.

`OUTER_TEST_USED=false`
