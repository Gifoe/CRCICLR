# V3 data protocol

- V3 inherits V2 subject partitions and records the SHA256 of each inherited split.
- The original Future indices are preserved byte-for-value at the array level.
- Original context is split chronologically into A and P. EEGMMIDB is split at whole-run boundaries; HMC and CAP use the same chronological 1:1 rule.
- A/P/V are nonempty, pairwise disjoint, ordered, and A+P exactly reconstructs context.
- Action search and Oracle Stage 0 use only `meta_risk_train`. Calibration, final/outer evaluation, and CAP labels are excluded from selection.
- CAP remains an external-site replication because it was observed in earlier project versions; it is not an untouched confirmation set.
