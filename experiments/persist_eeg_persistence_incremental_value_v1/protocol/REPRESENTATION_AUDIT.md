# Frozen classifier-input representation audit

The first five heads are exactly the formal PEEH forward-pre-hooks. EEGConformer uses the input to the full task-specific `model.classifier` sequential head, not its logits or an intermediate Transformer token. FBCNet uses the input to `model.head` after its fixed filter bank, spatial convolution, and log-variance operation. No representation layer is selected by heldout outcomes. Strict loading of the frozen fold0/seed0 checkpoints and forward-pre-hook shape checks passed for EEGConformer/FBCNet on each of the four tasks using zero-valued input, without reading outcome labels.

- EEGNet/OpenBMI_MI: `model.head forward-pre-hook`; dimension 64
- EEGNet/OpenBMI_ERP: `model.head forward-pre-hook`; dimension 64
- EEGNet/OpenBMI_SSVEP: `model.head forward-pre-hook`; dimension 64
- EEGNet/WBCIC_MI: `model.head forward-pre-hook`; dimension 64
- CBraMod/OpenBMI_MI: `model.head forward-pre-hook`; dimension 200
- CBraMod/OpenBMI_ERP: `model.head forward-pre-hook`; dimension 200
- CBraMod/OpenBMI_SSVEP: `model.head forward-pre-hook`; dimension 200
- CBraMod/WBCIC_MI: `model.head forward-pre-hook`; dimension 200
- TeCh/OpenBMI_MI: `model.model.projector forward-pre-hook`; dimension 128
- TeCh/OpenBMI_ERP: `model.model.projector forward-pre-hook`; dimension 128
- TeCh/OpenBMI_SSVEP: `model.model.projector forward-pre-hook`; dimension 256
- TeCh/WBCIC_MI: `model.model.projector forward-pre-hook`; dimension 128
- ModernTCN/OpenBMI_MI: `model.model.model.head_class forward-pre-hook`; dimension 246016
- ModernTCN/OpenBMI_ERP: `model.model.model.head_class forward-pre-hook`; dimension 63488
- ModernTCN/OpenBMI_SSVEP: `model.model.model.head_class forward-pre-hook`; dimension 246016
- ModernTCN/WBCIC_MI: `model.model.model.head_class forward-pre-hook`; dimension 230144
- Medformer/OpenBMI_MI: `model.model.projection forward-pre-hook`; dimension 72576
- Medformer/OpenBMI_ERP: `model.model.projection forward-pre-hook`; dimension 18432
- Medformer/OpenBMI_SSVEP: `model.model.projection forward-pre-hook`; dimension 72576
- Medformer/WBCIC_MI: `model.model.projection forward-pre-hook`; dimension 72576
- EEGConformer/OpenBMI_MI: `model.classifier forward-pre-hook`; dimension 2440
- EEGConformer/OpenBMI_ERP: `model.classifier forward-pre-hook`; dimension 440
- EEGConformer/OpenBMI_SSVEP: `model.classifier forward-pre-hook`; dimension 2440
- EEGConformer/WBCIC_MI: `model.classifier forward-pre-hook`; dimension 2440
- FBCNet/OpenBMI_MI: `model.head forward-pre-hook`; dimension 1152
- FBCNet/OpenBMI_ERP: `model.head forward-pre-hook`; dimension 1152
- FBCNet/OpenBMI_SSVEP: `model.head forward-pre-hook`; dimension 1152
- FBCNet/WBCIC_MI: `model.head forward-pre-hook`; dimension 1152
