# PERSIST-EEG WBCIC EEGNet Actionability V2

This experiment is the prospective three-session WBCIC/Yang2025 actionability audit. The first and only primary backbone route is EEGNet. It does not search for a positive intervention target: AGDI is authorized only if a frozen block passes H1--H5 under subject-disjoint five-fold cross-fitting.

## Frozen primary design

- Dataset: NEMAR `nm000348` 2C core, 51 subjects and three sessions.
- Split: 41 deterministic development subjects and 10 sealed outer subjects.
- Target: S1+S2 to future-session S3.
- Preprocessing: 59 EEG channels, Pz subtraction/drop to 58 channels, 0.5--40 Hz, 1000 to 250 Hz, event-relative 0--4 s imagery, fixed amplitude transform.
- Representation: EEGNet only, two prospectively fixed ordinary optimizer configurations, one seed (`20260817`), selected solely by cross-fitted S3 task performance.
- Audit: outcome `F_k`, discovery/decision `F_(k+1)`, model-fit remaining three folds.
- Inference unit: subject; 100 same-rank controls; 10,000 subject bootstraps; Holm correction across four blocks.

## Commands

Run from the repository root on the WBCIC machine:

```powershell
E:\Anaconda\python.exe experiments\persist_eeg_wbcic_actionability_v2\code\protocol.py prepare --raw-root D:\nips-temp\TotalP\P2\nm000348_v1.0.4_bids --header-workers 4
E:\Anaconda\python.exe experiments\persist_eeg_wbcic_actionability_v2\code\cache.py build --raw-root D:\nips-temp\TotalP\P2\nm000348_v1.0.4_bids --workers 4 --batch-size 4
D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe experiments\persist_eeg_wbcic_actionability_v2\code\pipeline.py competence --device cuda --workers 4
D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe experiments\persist_eeg_wbcic_actionability_v2\code\pipeline.py audit --device cuda --workers 4
D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe experiments\persist_eeg_wbcic_actionability_v2\code\pipeline.py agdi --device cuda --workers 4
```

If and only if AGDI passes development selection and creates `FINAL_OUTER_EVALUATION_LOCK.json`, run the one-time outer command:

```powershell
D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe experiments\persist_eeg_wbcic_actionability_v2\code\outer.py evaluate --raw-root D:\nips-temp\TotalP\P2\nm000348_v1.0.4_bids --device cuda --workers 2
```

The development runtime never opens the sealed outer split. Outer preprocessing materializes only S3 and is impossible before the final lock.

## Stop rules

- Competence failure: `REPRESENTATION_COMPETENCE_FAIL`; do not compute H1--H5.
- No joint H1--H5 block: `WBCIC_AUDIT_NO_ACTIONABLE_HARMFUL` and `STOP_AGDI_NO_ACTIONABLE_TARGET`.
- Authorized target but no development gain/specificity: stop before outer evaluation.
- Only a frozen nonzero AGDI that passes development gates may open the ten outer subjects once.

Raw data, epoch caches, embeddings, and checkpoints are derivatives and are not committed. Protocol locks, compact result tables, the final decision, report, and reproducibility record are committed after execution.
