# Identity probe audit

The primary identity quantity is the exact Experiment-3 symmetric cross-session identity skill: `0.5*((log(K)-CE_S1_to_S2)+(log(K)-CE_S2_to_S1))`. Both directions use the same standardized ridge multiclass linear classifier with alpha 1.0. No probe parameter is tuned per model.

The primary manipulation scope is the 24 inner-train source subjects because these are the domains that DANN/CORAL/MMD can directly manipulate. An all-32-source-subject sensitivity is also retained. Subject-ID top-1 accuracy and chance-normalized accuracy accompany the skill value, but the stress-test x-axis uses the frozen skill value.
