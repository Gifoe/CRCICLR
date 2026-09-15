# Cross-backbone PEEH v1

This is a frozen-checkpoint, inference-only seed-0 analysis. It compares
EEGNet, CBraMod, TeCh, ModernTCN, and Medformer on OpenBMI MI/ERP/SSVEP and
WBCIC MI over five canonical folds (100 run cells).

TFFormer is excluded by explicit user instruction. SGN is deferred because
the server contains only 14 of its required 20 selected seed-0 checkpoints;
this analysis is not allowed to train or select a replacement checkpoint.

The runner hooks the tensor immediately entering each frozen classification
head. No logits are used as representations, no common-dimensional projection
is imposed, and all models remain in eval/inference mode. The protocol lock
records checkpoint hashes before analysis.

Runtime cell files live outside the repository in
`D:\nips-temp\TotalP\P1\crossbackbone_peeh_runtime`. Only compact source,
protocol, audit, result, and report files are committed.
