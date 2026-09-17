# Authoritative model audit

Both models are imported directly after SHA-256 validation. No model class is copied or reconstructed in this experiment.

| Model | Final class | SHA-256 | C=64,T=1000,K=2 parameters |
|---|---|---|---:|
| EEGNet | `eegnet_locked.EEGNet(64,1000)` | `f7c513c3f3cd1f326a74b4e419bd693e15ee8980a7c378f1c0bee8215b8b89dd` | 34,194 |
| SIRE-EEG | `run_carrier_screen.CompactLite(64,'bn')` | `920af131aabc272317da128f42be9961d5592619ce85ca99192029d1181f126f` | 48,074 |

SIRE is the final CompactLite BN architecture: temporal k=15/63/127 branches (1->8), branch spatial grouped 8->16, concatenate 48, depthwise k=15 48->48 then pointwise 48->64, depthwise k=31 64->64 then pointwise 64->64, adaptive 8 bins, 512->64 embedding, ELU, LayerNorm(64), and 64->2 head. The per-fold assertion is `44,872 + 48*64 + 65*2 = 48,074`.

## Critical state tensor shapes
### EEGNet
- `temporal.weight`: `(8, 1, 1, 64)`
- `spatial.weight`: `(16, 1, 64, 1)`
- `depth.weight`: `(16, 1, 1, 16)`
- `point.weight`: `(16, 16, 1, 1)`
- `embedding.0.weight`: `(64, 496)`
- `head.weight`: `(2, 64)`
### SIRE-EEG
- `temporal.0.weight`: `(8, 1, 1, 15)`
- `temporal.1.weight`: `(8, 1, 1, 63)`
- `temporal.2.weight`: `(8, 1, 1, 127)`
- `spatial.0.weight`: `(16, 1, 64, 1)`
- `spatial.1.weight`: `(16, 1, 64, 1)`
- `spatial.2.weight`: `(16, 1, 64, 1)`
- `depth1.weight`: `(48, 1, 1, 15)`
- `point1.weight`: `(64, 48, 1, 1)`
- `depth2.weight`: `(64, 1, 1, 31)`
- `point2.weight`: `(64, 64, 1, 1)`
- `embedding.0.weight`: `(64, 512)`
- `head.weight`: `(2, 64)`

- Final episode source: `/root/rivermind-data/CRCICLR_FINAL_CONFIRM_WORK/experiments/persist_eeg_r2eeg_stage1_v1/code/run_stage1.py` SHA-256 `40cd24d90a1919db14d9d7b5a0b976bc5e4943e82e8a9fec10bbb82d70d78b75`.
- Final core source: `/root/rivermind-data/CRCICLR_FINAL_CONFIRM_WORK/experiments/persist_eeg_r2eeg_stage1_v1/code/stage1_core.py` SHA-256 `5277ec7055974c953acf32df3fe58a05761e0439b66c43eca3049563940c21ce`.
- Final training config: `/root/rivermind-data/CRCICLR_FINAL_CONFIRM_WORK/experiments/persist_eeg_carrier_5fold_multiseed_stability_v1/protocol/TRAINING_PROTOCOL.json` SHA-256 `e03ba36815df3d8f808e558211a2cba739b7fc53c0179fc9cc8c6cab31adf936`.
- Final training source: `/root/rivermind-data/CRCICLR_FINAL_CONFIRM_WORK/experiments/persist_eeg_carrier_5fold_multiseed_stability_v1/code/train_grid.py` SHA-256 `23f67f2c6ee75ce0babaab56a9e37b3210fa7b589d0347a417d1437fb1290c61`.
- Inspected final repository commit: `cf1db5a6d8f337626544b44f13057e86abe54dc8` on `codex/persist-eeg-final-heldout-confirmation-v1`.
