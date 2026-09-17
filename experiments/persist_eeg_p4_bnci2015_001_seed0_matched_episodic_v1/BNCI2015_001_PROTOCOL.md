# BNCI2015-001 matched episodic protocol

Independent seed-0 external replication. It is separate from all Shin2017B directories and contains no PEEH, PSWA, ScaleCollapse, PRD, BN-state, suppression, adaptation, tuning, extra baseline, or seed-1/2 run.

- Dataset/task: BNCI2015-001 / NEMAR nm000140; right-hand MI (class 0) versus feet MI (class 1).
- Cohort/session contract: 12 subjects; S1=0A and S2=1B only. 2C is never downloaded, cached, normalized, sampled, selected, or evaluated.
- Signal contract: fixed native 512 Hz official data; deterministic MNE polyphase resampling to 250 Hz; from each trial marker retain the fixed sustained imagery interval [cue+1.25 s, cue+5.25 s), yielding 13x1000. No frequency search, BNCI-specific band, CSP, or post-hoc crop appears.
- Splits: seed-0 five-fold outer subject CV; one inner-validation subject and at least eight inner-train subjects per fold. Outer and validation subjects are excluded from episodes, normalization and checkpoint selection.
- Episode: direct call to final `run_stage1.make_manifest` through its OpenBMI branch. Four S1 support subjects and four disjoint S2 query subjects, each 8 trials/class, produce 64+64=128 ordinary-CE trials.
- Normalization: channel mean/std fit only on inner-train S1 trials.
- Frozen recipe: AdamW lr=3e-4, weight decay=5e-4, 60 epochs, clip=5.0, AMP on CUDA, eligible epochs 10..60, S2 inner-validation subject-equal BA, earliest strict tie.
- Primary endpoint: subject-equal outer S2 BA. Secondary: outer S2 Macro-F1 and WS-BA=min(S1 BA,S2 BA).

## Provenance
- MOABB official BNCI downloader and its _convert_mi MAT-to-Raw converter; only explicit A/B URLs were requested.
- MOABB version: `1.7.2`.
- Final SIRE: `/root/rivermind-data/CRCICLR_FINAL_CONFIRM_WORK/experiments/persist_eeg_carrier_dualdataset_screen_v1/code/run_carrier_screen.py` SHA-256 `920af131aabc272317da128f42be9961d5592619ce85ca99192029d1181f126f`.
- Final EEGNet: `/root/rivermind-data/CRCICLR_FINAL_CONFIRM_WORK/experiments/persist_eeg_carrier_dualdataset_screen_v1/code/eegnet_locked.py` SHA-256 `f7c513c3f3cd1f326a74b4e419bd693e15ee8980a7c378f1c0bee8215b8b89dd`.
- Final episode source: `/root/rivermind-data/CRCICLR_FINAL_CONFIRM_WORK/experiments/persist_eeg_r2eeg_stage1_v1/code/run_stage1.py` SHA-256 `40cd24d90a1919db14d9d7b5a0b976bc5e4943e82e8a9fec10bbb82d70d78b75`.
