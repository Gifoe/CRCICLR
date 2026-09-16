# SIRE implementation audit

## Authoritative current-server chain

- Final-heldout manifest: `/root/rivermind-data/CRCICLR_FINAL_CONFIRM_WORK/experiments/persist_eeg_final_heldout_confirmation_v1/protocol/SOURCE_CHECKPOINTS.json`.
- Frozen final-heldout checkpoint tested: `/root/rivermind-data/carrier_5fold_multiseed_stability_runtime/openbmi_fold0_seed0_litebn/selected_best.pt` (SHA-256 `6ac976196e0658f66888c0fe23c909c6745c567e57c05cb7e5b2f2113adc5c8c`); strict-loads into the current C=62, K=2 class with 47,978 trainable parameters.
- Training script/model source: `/root/rivermind-data/CRCICLR_FINAL_CONFIRM_WORK/experiments/persist_eeg_carrier_dualdataset_screen_v1/code/run_carrier_screen.py`; SHA-256 `920af131aabc272317da128f42be9961d5592619ce85ca99192029d1181f126f`; imported directly at runtime as `CompactLite`.
- Frozen training config: `/root/rivermind-data/CRCICLR_FINAL_CONFIRM_WORK/experiments/persist_eeg_carrier_5fold_multiseed_stability_v1/protocol/TRAINING_PROTOCOL.json`; SHA-256 `e03ba36815df3d8f808e558211a2cba739b7fc53c0179fc9cc8c6cab31adf936`. Recipe: AdamW 3e-4, weight decay 5e-4, 60 epochs, clip 5.0, no scheduler, and inner future-session mean-subject BA selection in epochs 10--60 with earliest tie.
- Diagnostic embedding: the 64-D input to `CompactLite.head`, equivalently the output of `CompactLite.embedding` before final dropout/head in evaluation mode.

## Exact architecture and P4 adaptation

- Temporal branches: Conv2d 1->8 with kernels 15/63/127, BatchNorm2d(8), ELU, branch spatial Conv2d 8->16 with groups=8, BatchNorm2d(16), pool (1,4), dropout 0.20.
- Concatenated width 48; depthwise temporal k=15 48->48 plus pointwise 48->64/BatchNorm2d(64), then depthwise k=31 64->64 plus pointwise 64->64/BatchNorm2d(64); each refinement has pool (1,2) and dropout 0.15.
- AdaptiveAvgPool2d((1,8)); Linear(512,64), ELU, LayerNorm(64), dropout 0.25, Linear(64,K).
- Parameter formula: `44,872 + 48*C + 65*K`. For Shin2017B C=30, K=2: **46,442 trainable parameters** (instantiated and asserted before every corrected fold).
- Only C=30, T=2000 (parameter-invariant due adaptive pooling), and K=2 are dataset-bound. Kernels, widths, pooling, normalization, dropout, optimizer and selection rule are otherwise unchanged.

## Rejected prior P4 implementation

- Prior P4 `run_p4.py::LiteBN` was not authoritative: it used 48->32 refinement, k=9/7, 4-bin pooling and Linear(128,64). Its formula was `13,512 + 48*C + 65*K`, or 15,082 parameters at C=30, K=2.
- The historical seven-backbone LiteBN has the same obsolete structure (`experiments/persist_eeg_seven_backbone_fourtask_3seed_v1/code/backbone_models.py::LiteBN`) and was explicitly excluded. It is historical main-table provenance, not the source used here.
- Invalid old-SIRE checkpoints and mixed outputs were preserved at `/root/p4_shin2017b_seed0/provenance/invalid_old_sire_seed0_pre_final_repair`; EEGNet artifacts were preserved and never retrained.

## Corrected SIRE checkpoint selection

| Fold | Selected epoch | Inner S3 subject-equal BA | Checkpoint SHA-256 |
|---:|---:|---:|---|
| 0 | 23 | 0.690000 | `44332f76d4da2b177e77bfc5539a75c6ede62b1216c62ea6d6a75f87ceb90a1f` |
| 1 | 38 | 0.680000 | `0393902ab83297fd20fbe7206ccbb72ac361e6aa19548641f9aca429e692e110` |
| 2 | 35 | 0.800000 | `296c49041d17a0b292aa7f4f3b737ca56f43e5bd77dcc9704aa583a508ec4c5e` |
| 3 | 10 | 0.770000 | `f7c250c60318d9f72b84c5949ac2cb6feccbeea1b1ced63668583b5fc2e21636` |
| 4 | 40 | 0.640000 | `d1174083330c752810f7d415a02c8e419964c00ab4cc6ad7819b4543e21a8064` |

All five corrected checkpoints passed source/config/state_dict/architecture/parameter audits before training. Any mismatch raises an error; no copied LiteBN fallback exists.
