# Method

The canonical EEGNet carrier emits a 64-D embedding `h`.  Both new models learn
an offset `g(h)`, use `P_rel(h-g(h))` and `P_res(h)`, concatenate the two 32-D
projected components, and classify with a 64-D head.  Decomp-CE uses CE only.
Decomp-Offset adds the pre-registered `0.25 *` per-dimension MSE from `g(h_i)`
to a detached, label-free leave-one-out center of the other 15 trials from the
same training subject/session episode group.  At inference, both models only
consume the current trial.
