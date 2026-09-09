# Seven-backbone four-task three-seed SEARCH protocol freeze

This document freezes every analysis decision used before the first new
SEARCH outer-development prediction. It is valid only together with its
SHA-256 sidecar and the committed `BACKBONE_ADMISSION.json`.

## Scope

- Backbones: EEGNet, LiteBN, TCFormer, ST-EEGFormer-small, LaBraM-base,
  CBraMod, and TeCh. Each is a single trained classifier; no ensemble,
  fusion, routing, or distillation is permitted.
- Tasks: OpenBMI MI, OpenBMI ERP, OpenBMI SSVEP, and WBCIC MI.
- Frozen folds: the existing five folds of `CARRIER_5FOLD_MULTISEED_STABILITY_V1`.
  Seeds are exactly `0, 1, 2`.
- Training input is each fold's source-session `inner_train_subjects` only.
  Checkpoints are selected only by subject-equal BA on the frozen future
  `inner_val_subjects`, starting at epoch 10. The selected checkpoint is
  evaluated once on that fold's future-session `outer_dev_subjects`.
- No fixed historical held-out subject is loaded by code in this experiment.
  Held-out evaluation remains forbidden until a second, result-bearing
  SEARCH freeze is committed and pushed.

## Common data contract

The complete admitted cache mirrors are:

- OpenBMI: `D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full\outputs\persist_eeg_stage0\cache\openbmi`
- WBCIC: `D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC\experiments\persist_eeg_wbcic_independent_replication_v1\runtime\cache\wbcic_epochs`

The cache admission ledger has already verified byte identity for the shared
OpenBMI MI and WBCIC MI samples and records no previous outer or fixed-heldout
access. For every cell, channelwise mean/std is computed on that cell's source
inner-train samples only, then applied unchanged to its inner validation and
outer-development samples. No model receives labels other than its own
inner-train labels during fitting.

## Architecture-bound adapters

- EEGNet, LiteBN, TCFormer and TeCh receive the normalized cache at 250 Hz.
- ST-EEGFormer-small receives deterministic `scipy.signal.resample_poly(2,5)`
  output at 100 Hz (400 samples for MI/SSVEP and 100 for ERP), then its
  official forward performs its published 100-to-128-Hz interpolation.
- LaBraM-base and CBraMod receive deterministic
  `scipy.signal.resample_poly(4,5)` output at 200 Hz and reshape only into
  non-overlapping 200-sample model-native patches (4 patches for MI/SSVEP,
  1 for ERP).
- LaBraM uses the frozen `FM_INPUT_PROTOCOL_LOCK.json` channel-index maps,
  including the source-required leading index. ST-EEGFormer uses the release
  channel-index map. Its vocabulary lacks OpenBMI `F9`/`F10`; these two
  channels use the published neighbouring `FT9`/`FT10` coordinates. This
  fixed, documented alias is not a result-driven choice.

## Fixed fitting recipes

| family | epochs | batch | AdamW lr | weight decay | clip |
| --- | ---: | ---: | ---: | ---: | ---: |
| EEGNet / LiteBN / TCFormer | 60 | 128 | 3e-4 | 5e-4 | 5.0 |
| ST-EEGFormer-small | 60 | 16 | 1e-4 | 5e-4 | 5.0 |
| LaBraM-base / CBraMod | 60 | 64 | 1e-4 | 5e-4 | 5.0 |

TeCh uses the previously locked official inner-only selection: TECH-T for
OpenBMI MI, OpenBMI ERP and WBCIC MI; TECH-TC for OpenBMI SSVEP. Its frozen
official recipe specifies 60 or 40 epochs respectively, batch 128, Adam
learning rate 1e-4 and the locked official augmentation configuration.

All shuffles, initialization and dropout streams are deterministic functions
of task, model, fold, seed and epoch. A cell writes a resumable state every
five epochs and whenever its selected checkpoint improves; checkpoint cadence
does not change training computation. Cells acquire an exclusive directory
lock, so duplicate workers cannot train the same model/task/fold/seed.

## Admission gate

`BACKBONE_ADMISSION.json` contains 28 synthetic-only records (seven models by
four task shapes). All records have correct logit shape, finite CE and
gradients, and zero eval-repeat difference. It also records exact checkpoint
loading reports, parameter counts, model-boundary shapes and peak batch-two
allocation. No cache, label, prediction, inner score, outer score, or heldout
score was accessed by the admission audit.
