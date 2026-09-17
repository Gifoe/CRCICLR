# Matched episodic Shin2017B protocol

This is a clean, separate seed-0 comparison. The existing IID P4 directory is read only and is retained as a control.

- Dataset: Shin2017B / nm000268, EEG only; 29 subjects, 30 channels in cached order, 2,000 samples/trial.
- Session mapping: S1=`1arithmetic`, S2=`3arithmetic`, S3=`5arithmetic`.
- The existing five subject-disjoint folds are copied byte-for-byte from the IID P4 manifest (SHA-256 `f1088d106dfbbfded0f4f69fae3ca90ae04976cf8e59b1204f1f7afa3dd4daea`).
- A source episode uses four inner-train support subjects and S1/S2, sampling four trials per class per session per subject (64 trials). A query uses four different inner-train subjects and S3, sampling eight trials per class per subject (64 trials). Thus every episode is 128 trials and support/query subjects are disjoint.
- The final `run_stage1.make_manifest` is called directly through its WBCIC metadata branch: numerical sessions 0/1/2 are exactly S1/S2/S3. This preserves its 20 steps/epoch, 60 epochs, fixed RNG formula, and 4/4 subject construction rather than using shuffled IID mini-batches.
- Mean/std is fit only on inner-train S1+S2. Inner-validation and outer subjects never occur in an episode or checkpoint selection. Query S3 is only from inner-train subjects.
- Both directly imported final models use AdamW (lr 3e-4, weight decay 5e-4), ordinary cross entropy over the complete support+query episode, gradient clip 5.0, CUDA AMP when available, 60 epochs, and S3 inner-validation subject-equal BA selection over epochs 10--60 with earliest strict tie.
- All outer S1/S2/S3 evaluations happen only after all 10 fold/model trainings finish; the frozen selected checkpoint is used.

## Imported authoritative models

- EEGNet: `/root/rivermind-data/CRCICLR_FINAL_CONFIRM_WORK/experiments/persist_eeg_carrier_dualdataset_screen_v1/code/eegnet_locked.py`; SHA-256 `f7c513c3f3cd1f326a74b4e419bd693e15ee8980a7c378f1c0bee8215b8b89dd`; C=30,T=2000,K=2 parameters 65,394.
- SIRE-EEG/CompactLite: `/root/rivermind-data/CRCICLR_FINAL_CONFIRM_WORK/experiments/persist_eeg_carrier_dualdataset_screen_v1/code/run_carrier_screen.py`; SHA-256 `920af131aabc272317da128f42be9961d5592619ce85ca99192029d1181f126f`; C=30,T=2000,K=2 parameters 46,442.
- Final episode source: `/root/rivermind-data/CRCICLR_FINAL_CONFIRM_WORK/experiments/persist_eeg_r2eeg_stage1_v1/code/run_stage1.py`; SHA-256 `40cd24d90a1919db14d9d7b5a0b976bc5e4943e82e8a9fec10bbb82d70d78b75`. Final grid source: `/root/rivermind-data/CRCICLR_FINAL_CONFIRM_WORK/experiments/persist_eeg_carrier_5fold_multiseed_stability_v1/code/train_grid.py`; SHA-256 `23f67f2c6ee75ce0babaab56a9e37b3210fa7b589d0347a417d1437fb1290c61`.
