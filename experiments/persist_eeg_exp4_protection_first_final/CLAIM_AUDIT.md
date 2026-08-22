# Claim audit

## Supported

1. On the 41-subject WBCIC development cohort, Generic S2 adaptation has real future-session headroom over a Frozen S1 EEGNet anchor.
2. V1 hard projection preserves the selected Protected coordinates numerically (`3.7e-8` normalized drift) while still producing complement-space adaptation.
3. The observed Guard−Generic mean is small and uncertain; it is not evidence of a reliable improvement.
4. The V2 response-preserving correction demonstrates that the Exp3 decision-response constraint can be implemented, but it is harmful in this setting.

## Not supported

* `PERSISTGuard` improves future-session generalization.
* Protection reduces negative transfer: V1/V3 are worse than Generic on this endpoint.
* The protected criterion is specific: Random/PCA/persistence-only/identity controls are not separated.
* The result generalizes to sealed WBCIC subjects, another backbone, OpenBMI, or EEG generally.
* Subject identity is useless, or the complement is nuisance.

The only defensible paper-level Exp4 statement is a negative one: a strong generic update existed, but these protection-first variants did not convert the frozen task-protected block into a validated safer future-session update under the preregistered gates.
