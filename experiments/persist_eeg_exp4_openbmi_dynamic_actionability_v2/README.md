# PERSIST-EEG Exp4 OpenBMI Dynamic Actionability V2

This directory contains the pre-registered Phase-A trajectory/gradient audit. It is a new branch and does not overwrite the earlier Exp4 negative result.

Terminal state: **EXP4_OPENBMI_DYNAMIC_ACTIONABILITY_NOT_SUPPORTED**

Data scope: OpenBMI MI, V8_SEARCH only (40 development subjects), five subject-only folds. The 14-subject internal holdout and historical outer test were not loaded. No WBCIC data were used.

The implementation uses the legal materialised MI-specific embedding/logit cache. The deployment surrogate is a residual linear head initialized at the frozen logit anchor and updated with five full-batch Session-1 BCE steps. This is an audit of prospective signal validity, not a claim that the cached feature head is the complete raw-EEG Generic.

See `DYNAMIC_ACTIONABILITY_AUDIT.md` and `DYNAMIC_DEV_PROTOCOL.json` for the exact gate and provenance.
