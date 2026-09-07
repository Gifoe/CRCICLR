# PERSIST-PSG seed-0 method

At each deterministic subject-disjoint step, A supplies the task gradient and B supplies a dropout-free guard gradient. The ordinary AdamW proposal is executed first from g_A (with the registered clip contract), then the exact parameter displacement is measured. If h = g_B^T Delta_task is positive, only the declared ALL or LATE subspace receives c = min(1, kappa ||Delta_S|| / ||h g_B,S / ||g_B,S||^2||) h g_B,S / ||g_B,S||^2. The correction is applied to parameters after optimizer.step while AdamW moments are retained. No anchor fusion or extra hyperparameter was used.

This is seed 0 development evidence, not sealed confirmation.
