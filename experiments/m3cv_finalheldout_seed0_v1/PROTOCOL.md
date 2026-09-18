# M3CV final-heldout supervised protocol

- Analysis cohort: 93 cached eligible subjects; 75 development and 18 final-heldout.
- Final-heldout assignment: `rank SHA256('m3cv-final-heldout-v1|' + subject_id), take first 18`. It uses subject IDs only and is fixed before model construction.
- Every train subject contributes both ses-01 and ses-02 LH/RH trials; LH=0, RH=1.
- Per development fold: 48 train, 12 validation, 15 outer-development subjects; all partitions are subject-disjoint.
- Input: native direct 64x1000 float32 EEG cache at 250 Hz, with preserved 64-channel order.
- Models: direct imports of final CompactLite_BN/SIRE-EEG and canonical EEGNet. Only C=64 is adapted.
- Optimizer: ordinary CE, AdamW lr=3e-4, weight_decay=5e-4, gradient clip=5.0, maximum 60 epochs.
- Selection: validation ses-02 subject-equal BA; eligible epoch >=10; earliest strict tie. Stop after 8 non-improving eligible epochs.
- Normalizer: channel mean/std fit on fold TRAIN subjects using both sessions only; frozen for validation, outer-development and final-heldout.
- Final-heldout signals/labels are opened only after all five development checkpoints and their normalizers are frozen.
- Scope: seed 0 only; no seed 1/2 is launched by this run.
