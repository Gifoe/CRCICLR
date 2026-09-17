# M3CV matched episodic protocol

This is a seed-0 external replication on M3CV / NEMAR nm000166. The task is left-hand versus right-hand motor execution, not motor imagery.

- Downloaded cohort: 95. Analysis cohort: 93 participants with at least eight valid trials per class in both sessions. Excluded for pre-training availability only: sub-035, sub-080.
- S1=`ses-01`; S2=`ses-02`; 64 EEG channels; 250 Hz; direct native 4-s segments, T=1000. No resampling, window search, CSP/xDAWN/Riemannian or handcrafted features.
- Cache retains every valid LH/RH source epoch for every eligible subject/session. Variable subject/session counts are preserved; outer evaluation uses all valid cached trials.
- Fixed seed-0 five-fold outer subject CV: 19,19,19,18,18 outer subjects; 10 inner validation; remaining 64/65 inner train. EEGNet and SIRE use identical splits/manifests.
- The direct final sampler supplies 4 S1 support and 4 different S2 query inner-train subjects, 8 LH+8 RH per subject: 64+64=128 ordinary-CE trials. It permits repetitions only across episodes.
- Normalization pools all valid S1 epochs from inner-train subjects, exactly as the authoritative implementation; variable trial counts therefore give subjects proportional trial weighting.
- User-authorized training correction: retain the final AdamW/ordinary-CE/AMP/clip settings and 60-epoch maximum, but train at least 10 epochs then stop after 8 consecutive strict non-improvements in inner-val S2 subject-equal BA. Strict `>` preserves earliest-tie checkpoint selection.
- Hard stop after this corrected seed-0 execution: no other seed, diagnostic, ablation, ScaleCollapse, rank matching, PRD, BN-state, adaptation, suppression, baseline or tuning.
