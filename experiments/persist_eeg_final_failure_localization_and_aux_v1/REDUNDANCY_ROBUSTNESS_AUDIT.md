# Redundancy and perturbation robustness audit

Channel dropout (10%), contiguous temporal masking (10% window), and Gaussian noise (sigma=0.05×source-train channel std) were locked before evaluation with 32 deterministic draws. Perturbations were applied identically to all models and never used for training or strength selection. The table retains per-draw BA/logit/margin summaries; branch-erasure harm is in reliance_metrics.csv. A two-subject-per-fold runtime cap is declared for this secondary robustness audit and does not change the primary frozen BA.
