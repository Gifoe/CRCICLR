# V7-0B repair protocol

```json
{
  "frozen_at": "2026-08-06T17:12:31.689177+00:00",
  "post_hoc_development_repair": true,
  "starting_commit": "86b6ea46c8b5a3497e68472d75f490520503637d",
  "models": [
    "cbramod",
    "labram",
    "biot"
  ],
  "checkpoint_hashes": {
    "cbramod": "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178",
    "labram": "7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c",
    "biot": "40f55f5d23e83796495616c8145c8336fcff2b901c42e8ba5115223081c2ad70"
  },
  "code_commits": {
    "cbramod": "0ff6be918985689e7df679bc731ffb70e6c6224f",
    "labram": "c431221e6cfd23dbfa9950e0180682fb322b0548",
    "biot": "d138e32634e52ae9fa6ec98ac9c4087b14ca869a"
  },
  "checkpoint_paths": {
    "cbramod": "/root/autodl-tmp/hsc_tta_eeg/checkpoints/cbramod/pretrained_weights.pth",
    "labram": "/root/autodl-tmp/hsc_tta_eeg/external/LaBraM/checkpoints/labram-base.pth",
    "biot": "/root/autodl-tmp/hsc_tta_eeg/external/BIOT/pretrained-models/EEG-PREST-16-channels.ckpt"
  },
  "canonical_manifest_hash": "c947f7aa64e4a072d98d03d928682e39842d036c5221e5d20951294499b9d997",
  "datasets": [
    "hmc",
    "eegmmidb"
  ],
  "outer_folds": [
    0,
    1,
    2,
    3,
    4
  ],
  "seeds": [
    0,
    1,
    2,
    3,
    4
  ],
  "fold_roles": "evaluation=e; validation=(e+1)%5; training=remaining three; refit=all four non-evaluation folds",
  "adapter_priority": [
    "checkpoint configuration",
    "same-checkpoint official inference",
    "official downstream code",
    "minimal deterministic mapping"
  ],
  "structured_representation": {
    "cbramod": "final channel-patch tokens with channel and patch identity",
    "labram": "final patch/channel tokens; HMC three 10-second subwindows concatenated in time order",
    "biot": "final transformer token sequence with checkpoint channel-token and patch order"
  },
  "readout_families": {
    "H0_GLOBAL_LOGREG": {
      "C": [
        0.01,
        0.1,
        1.0,
        10.0
      ],
      "solver": "LBFGS",
      "max_iter": 2000,
      "class_weight": "training-fold inverse frequency"
    },
    "H1_TOKEN_ATTENTION_POOL": {
      "architecture": "Linear(input_dim,64)->LayerNorm->one learned query->4-head cross-attention->residual 2-layer FFN(hidden=128)->LayerNorm->Linear(n_classes)",
      "optimizer": "AdamW",
      "lr": [
        0.0001,
        0.0003,
        0.001
      ],
      "weight_decay": [
        0.0001,
        0.001
      ],
      "dropout": [
        0.0,
        0.1
      ],
      "max_epochs": 100,
      "patience": 15,
      "gradient_clip": 1.0,
      "loss": "class-weighted cross entropy",
      "validation_metric": "mean subject balanced accuracy",
      "mini_batch": true,
      "deterministic": true,
      "parameter_cap": 100000
    }
  },
  "head_tie_rule": "choose H0 when validation mean-subject BA difference is <=0.005",
  "adapter_gate": {
    "F1": "all adapters have official code/checkpoint basis",
    "F2": "no performance-driven adapter selection",
    "F3": "100% canonical subject coverage",
    "F4": ">=95% canonical sample coverage per model",
    "F5": "labels and sample IDs exactly match V7",
    "F6": "all structured tokens finite",
    "F7": "mask, channel identity and temporal order recoverable"
  },
  "anchor_gate": {
    "C1": "HMC BA differs from V6 by <=0.005",
    "C2": "EEGMMIDB BA differs from V6 by <=0.005",
    "C3": "sample counts match for every seed/fold",
    "C4": "no evaluation leakage",
    "C5": "overlap probability max abs diff <=1e-6"
  },
  "qualification_gates": {
    "R1": "probability sanity",
    "R2": "all true classes present",
    "R3": ">=4/5 seeds predict all classes",
    "R4": "every fold predicts >=4 HMC or >=3 EEGMMIDB classes",
    "R5": "nonconstant-subject rate >=0.95",
    "R6": "seed BA std <=0.05",
    "R7": "HMC dataset BA >=0.28",
    "R8": "HMC median-subject BA >=0.25",
    "R9": "EEGMMIDB dataset BA >=0.33",
    "R10": "EEGMMIDB median-subject BA >=0.30",
    "R11": "repaired expert no more than 0.15 BA below CBraMod anchor",
    "R12": ">=4/5 seed BAs above chance",
    "anchor_extra": "HMC BA>=0.55; EEGMMIDB BA>=0.38; Gate C passed"
  },
  "primary_risk": "training-fold inverse-frequency class-balanced error; subject risk first; average seeds within subject; equal-weight subjects",
  "best_fixed": "select minimum inner-validation risk per fold/seed with tie M0,M1,M2; freeze for evaluation fold",
  "full_oracle": "minimum full-subject labeled risk; G_full=(R_best_fixed-R_oracle)/R_best_fixed",
  "transfer_oracle": "HMC chronological halves; EEGMMIDB condition/run alternating halves; select on A evaluate B and vice versa",
  "bootstrap": {
    "unit": "subject",
    "repetitions": 5000,
    "seed": 20260812,
    "aggregate_seed_within_subject_first": true
  },
  "subject_shuffle_null": {
    "repetitions": 500,
    "seed": 20260813,
    "within_dataset_fold": true
  },
  "same_backbone_null": "CBraMod source-head seeds r,(r+1)%5,(r+2)%5",
  "oracle_gates": {
    "A1": "G_full>=0.15",
    "A2": "G_full CI lower>0",
    "A3": ">=40% positive-gain subjects",
    "A4": ">=2 experts winner share>=0.15",
    "A5": "every winner share<=0.80",
    "A6": ">=4/5 seed G_full>0",
    "A7": "all leave-one-fold-out G_full>0",
    "A8": "mean subject rescuable error>=0.15",
    "A9": "G_transfer>=0.08",
    "A10": "G_transfer CI lower>0",
    "A11": "G_excess_shuffle>=0.05",
    "A12": "G_excess_shuffle CI lower>0",
    "A13": ">=4/5 seed G_transfer>0",
    "A14": "G_excess_backbone_full>=0.05",
    "A15": "G_excess_backbone_full CI lower>0",
    "A16": "G_excess_backbone_transfer>=0.03",
    "A17": "G_excess_backbone_transfer CI lower>0",
    "A18": "all leave-one-subject-out G_transfer>0"
  },
  "verdicts": [
    "V7R_STOP_ADAPTER_FIDELITY_FAILURE",
    "V7R_STOP_CBRAMOD_ANCHOR_FAILURE",
    "V7R_STOP_EXPERT_QUALIFICATION_FAILURE",
    "V7R_STOP_NO_CROSS_MODEL_HEADROOM",
    "V7R_STOP_NO_STABLE_SUBJECT_COMPLEMENTARITY",
    "V7R_CONTINUE_TO_UNLABELED_ROUTING_SCREEN",
    "V7R_TECHNICAL_BLOCK"
  ],
  "protected_flags": {
    "formal_calibration_opened": false,
    "internal_final_opened": false,
    "cap_opened": false,
    "sleep_edf_opened": false,
    "bcic2a_opened": false,
    "router_developed": false,
    "abstention_developed": false,
    "scout_developed": false,
    "full_method_entered": false
  }
}
```
