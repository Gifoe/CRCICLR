# Method

The final predictor was selected before the held-out labels were opened:

\[
z_{final}=0.5z_{EEGNet}+0.5z_{LiteBN}.
\]

For each held-out subject, the evaluation uses the same 15 frozen matched carrier pairs (five folds times three seeds) used in the carrier/fusion development audit. OpenBMI uses future session S2; WBCIC uses future session S3 (cache session index 2). Every fold normalizer is recomputed only from that carrier fold's original `inner_train_subjects` and the historical source sessions (OpenBMI S1; WBCIC sessions 0 and 1), then hash-checked against the carrier manifest.

The primary analysis first computes each subject's metric under each fixed replicate, then averages that subject metric across 15 replicates. Dataset means, paired bootstrap intervals, and gates use these subject averages; replicate summaries are secondary stability diagnostics only.

The evaluation implementation has no training code path. All carrier parameters are disabled for gradients, all carriers are in evaluation mode, and all forward passes execute under `torch.no_grad()`.
