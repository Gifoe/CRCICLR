# Protocol

Development uses exactly 40 V8_SEARCH subjects in the frozen five folds. For an evaluated subject, Session 1 is legal labeled history and Session 2 is scoring-only. Fold source subjects may use S1->S2 as legal meta-training episodes. The 14-subject internal holdout is removed before raw tensors are materialised. The historical 83.775% V6 anchor is excluded from final-holdout use because it predates the 40/14 partition.
