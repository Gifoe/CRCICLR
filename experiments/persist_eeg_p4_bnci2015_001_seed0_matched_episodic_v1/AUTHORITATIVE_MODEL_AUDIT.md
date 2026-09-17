# Authoritative model audit

Both classes are imported directly from the final benchmark sources. This experiment does not maintain a copied/reconstructed model.

| Model | Final class/source | SHA-256 | C/T/K parameters | Forward output |
|---|---|---|---:|---|
| EEGNet | `eegnet_locked.EEGNet`<br>`/root/rivermind-data/CRCICLR_FINAL_CONFIRM_WORK/experiments/persist_eeg_carrier_dualdataset_screen_v1/code/eegnet_locked.py` | `f7c513c3f3cd1f326a74b4e419bd693e15ee8980a7c378f1c0bee8215b8b89dd` | 33,378 | logits [1,2], embedding [1,64] |
| SIRE-EEG | `run_carrier_screen.CompactLite(kind='bn')`<br>`/root/rivermind-data/CRCICLR_FINAL_CONFIRM_WORK/experiments/persist_eeg_carrier_dualdataset_screen_v1/code/run_carrier_screen.py` | `920af131aabc272317da128f42be9961d5592619ce85ca99192029d1181f126f` | 45,626 | logits [1,2], embedding [1,64] |

## Frozen SIRE structure

Temporal branches k=15/63/127 (1->8); branch spatial grouped convolutions 8->16 groups=8; concatenated width 48; depthwise k=15 then pointwise 48->64; depthwise k=31 then pointwise 64->64; adaptive pool 8; Linear 512->64; ELU; LayerNorm(64); dropout; Linear 64->2.

SIRE count assertion: 44,872 + 48*13 + 65*2 = 45,626. Any state shape, count, or forward-shape mismatch raises before training. EEGNet's independently audited final count is 33,378.

## Critical state shapes

### EEGNet
- `temporal.weight`: (8, 1, 1, 64)
- `spatial.weight`: (16, 1, 13, 1)
- `depth.weight`: (16, 1, 1, 16)
- `point.weight`: (16, 16, 1, 1)
- `embedding.0.weight`: (64, 496)
- `embedding.2.weight`: (64,)
- `head.weight`: (2, 64)

### SIRE-EEG
- `temporal.0.weight`: (8, 1, 1, 15)
- `temporal.1.weight`: (8, 1, 1, 63)
- `temporal.2.weight`: (8, 1, 1, 127)
- `spatial.0.weight`: (16, 1, 13, 1)
- `spatial.1.weight`: (16, 1, 13, 1)
- `spatial.2.weight`: (16, 1, 13, 1)
- `depth1.weight`: (48, 1, 1, 15)
- `point1.weight`: (64, 48, 1, 1)
- `depth2.weight`: (64, 1, 1, 31)
- `point2.weight`: (64, 64, 1, 1)
- `embedding.0.weight`: (64, 512)
- `embedding.2.weight`: (64,)
- `head.weight`: (2, 64)
