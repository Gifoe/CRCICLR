# Model pretraining provenance

Unknown exposure is retained as unknown, not rewritten as no overlap.

## biot

- Family: frequency-patch linear-attention transformer
- Code commit: `d138e32634e52ae9fa6ec98ac9c4087b14ca869a`
- License: MIT
- Checkpoint SHA256: `40f55f5d23e83796495616c8145c8336fcff2b901c42e8ba5115223081c2ad70`
- Parameters in checkpoint: 3442192
- Frozen encoder parameters: 3186192
- Pretraining objective: unsupervised contrastive pretraining
- Pretraining data: 5 million unlabeled PREST resting EEG samples
- Unsupervised overlap: False
- Exposure unknown: False
- Task labels used: False
- Admissible: True

## cbramod

- Family: criss-cross transformer
- Code commit: `0ff6be918985689e7df679bc731ffb70e6c6224f`
- License: MIT
- Checkpoint SHA256: `0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178`
- Parameters in checkpoint: 4924000
- Frozen encoder parameters: 4924000
- Pretraining objective: masked EEG reconstruction
- Pretraining data: TUEG, approximately 9,000 hours
- Unsupervised overlap: False
- Exposure unknown: False
- Task labels used: False
- Admissible: True

## labram

- Family: neural-tokenizer transformer
- Code commit: `c431221e6cfd23dbfa9950e0180682fb322b0548`
- License: MIT
- Checkpoint SHA256: `7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c`
- Parameters in checkpoint: 7466728
- Frozen encoder parameters: 5819936
- Pretraining objective: vector-quantized neural spectrum prediction and masked neural-code prediction
- Pretraining data: about 2,500 hours across around 20 datasets; complete corpus list not verified from checkpoint metadata
- Unsupervised overlap: False
- Exposure unknown: True
- Task labels used: False
- Admissible: True
