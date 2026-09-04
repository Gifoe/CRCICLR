# Cumulative displacement audit

For every window, Delta_epoch is theta_1-theta_0 after exact AdamW steps. The runner also sums each exact per-step parameter displacement and requires max absolute agreement <=5e-5.
