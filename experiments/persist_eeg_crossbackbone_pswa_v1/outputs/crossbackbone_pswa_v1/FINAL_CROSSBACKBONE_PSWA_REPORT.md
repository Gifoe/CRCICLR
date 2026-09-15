# Final cross-backbone PSWA v1 report

## Outcome

PSWA is **not estimable** under the frozen-artifact rules. No numeric PSWA value was fabricated.

The corrected PEEH run persisted checkpoint provenance, Protected ranks/assignments, and subject-level erasure summaries. It did not persist the frozen pre-classifier representation matrices, the canonical spectral transform (mean/basis/scale/directions), or the exact 100 final random-coordinate sets. Retained-subspace probe performance cannot be recovered from erased-subspace BA summaries.

## Primary seed-0 coverage

|Model|Task|Protected rank (mean over recorded folds)|Protected coverage|Status|
|---|---|---:|---:|---|
|EEGNet|OpenBMI_MI|4.80|5/5|INCOMPLETE|
|EEGNet|OpenBMI_ERP|6.40|5/5|INCOMPLETE|
|EEGNet|OpenBMI_SSVEP|6.40|5/5|INCOMPLETE|
|EEGNet|WBCIC_MI|2.00|5/5|INCOMPLETE|
|CBraMod|OpenBMI_MI|2.40|3/5|INCOMPLETE|
|CBraMod|OpenBMI_ERP|3.40|4/5|INCOMPLETE|
|CBraMod|OpenBMI_SSVEP|0.60|1/5|INCOMPLETE|
|CBraMod|WBCIC_MI|3.00|3/5|INCOMPLETE|
|TeCh|OpenBMI_MI|5.60|5/5|INCOMPLETE|
|TeCh|OpenBMI_ERP|7.00|5/5|INCOMPLETE|
|TeCh|OpenBMI_SSVEP|7.20|5/5|INCOMPLETE|
|TeCh|WBCIC_MI|4.80|5/5|INCOMPLETE|
|ModernTCN|OpenBMI_MI|0.20|1/5|INCOMPLETE|
|ModernTCN|OpenBMI_ERP|4.00|5/5|INCOMPLETE|
|ModernTCN|OpenBMI_SSVEP|4.60|5/5|INCOMPLETE|
|ModernTCN|WBCIC_MI|1.20|3/5|INCOMPLETE|
|Medformer|OpenBMI_MI|3.40|5/5|INCOMPLETE|
|Medformer|OpenBMI_ERP|5.60|5/5|INCOMPLETE|
|Medformer|OpenBMI_SSVEP|7.20|5/5|INCOMPLETE|
|Medformer|WBCIC_MI|3.60|5/5|INCOMPLETE|
|SGN|OpenBMI_MI|NA|0/5|INCOMPLETE|
|SGN|OpenBMI_ERP|NA|0/5|INCOMPLETE|
|SGN|OpenBMI_SSVEP|NA|0/5|INCOMPLETE|
|SGN|WBCIC_MI|NA|0/5|INCOMPLETE|

## Integrity conclusion

- Training performed: NO
- Inference rerun: NO
- Protected assignments recomputed: NO
- Persistence spectrum recomputed: NO
- Random controls redrawn: NO
- Numeric PSWA estimates: 0

A future PSWA run requires a separately authorized artifact-recovery amendment that deterministically regenerates and then freezes the exact representations, spectral transforms, and random coordinate sets. That amendment is outside this locked analysis.
