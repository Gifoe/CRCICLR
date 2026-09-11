# OpenBMI ERP XS seed1/2 stability protocol

Branch: codex/persist-eeg-xs-erp-seed12-stability-v1

Scope is restricted to OpenBMI ERP, seeds 1 and 2, five existing outer folds, and the existing canonical inner train/validation subjects. No new inner splits are created. LiteBN baseline checkpoints are reused only after strict state-dict verification and recorded provenance. LiteBN-X uses the original architecture, preprocessing, normalization, class-weighted CE, optimizer, learning rate, weight decay and batch protocol. Selection is canonical inner-validation subject-mean balanced accuracy, with eligibility at epoch 10 and a 60 epoch maximum; the best checkpoint is restored. This implementation uses the same selection rule for both seeds; historical partial XS runtime was not mixed because it lacked complete provenance.

Outer-development ERP subjects are accessed for the requested development/stability analysis. Final heldout/test data are not accessed.

FINAL_HELDOUT_ACCESSED = NO
