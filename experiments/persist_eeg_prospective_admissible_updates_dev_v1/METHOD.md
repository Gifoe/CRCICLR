# PERSIST-AU seed-0 development method

PERSIST-AU computes g_A and g_B on deterministic subject-disjoint, class-balanced 64-trial batches and applies the predeclared C1--C5 admissible-update rules. C0/ERM2 is the matched two-batch control. All candidates start from the exact canonical EEGNet seed-0 refit checkpoint and use AdamW (lr=3e-5, weight decay=5e-4, clip=5.0, five epochs). BN running statistics are frozen and asserted unchanged after every epoch. Scope B freezes early convolution/BN parameters; C4/C5 retain ERM updates in early layers and project only the late block.

The outcome role is an authorized development evaluation role. This pilot is seed 0 only and is not sealed confirmation.
