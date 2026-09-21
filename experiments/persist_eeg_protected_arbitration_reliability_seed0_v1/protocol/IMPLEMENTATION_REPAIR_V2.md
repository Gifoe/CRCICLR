# Implementation correction record

The original user specification is authoritative. This correction restores its definitions; no backbone, native head, protected subset, dataset split, final-heldout access, or intervention grid is changed.

The preliminary fold0/fold1 jobs were stopped before producing terminal result JSONs. Their logs remain on the original server. They are engineering attempts and must not be counted as scientific outcomes.

Corrected implementation defects:

- P-only/C-only decisions, margins, entropy and supervised correctness targets include z0.
- Reliability routing requires disagreement and rP-rC > tau, with suppression alpha in {0.25, 0.5, 0.75}; tau candidates retain the pre-existing {0.25, 0.5, 0.75} set.
- Primary direct-arbiter fitting excludes ambiguous both-wrong trials and reports their counts.
- Feature A excludes centroid features; B/D use nearest two TRAIN Protected-centroid distances.
- Pathway features include evidence consistency and across-layer consensus. Only the requested stages are used.
- Layer mapping reuses the preceding standardized GPU dual-kernel solver and its fixed ridge-alpha grid and five subject groups. Nested held-subject predictions exclude held subjects from alpha selection. Independent target matrix multiplication shapes are preserved.
- Kernels/eigendecompositions are shared across Protected and the first 20 fixed random targets. Each target retains independent parameter selection. Runtime checkpoints are keyed by inputs and implementation hash.
- Gain targets use independent Ridge parameters, shared design matrices and nested subject selection. All 100 random geometry controls include gain and reliability policies.
- Trial surfaces remain compressed in runtime storage. Scientific results require all fixed controls and corrected implementation provenance.

Synthetic tests cover target-wise solver equality, held-subject label exclusion for the gain estimator, bias-aware geometry, agreement-preserving routing, taxonomy counts and pathway checkpoint shapes. They are not substitutes for validating real-cell outputs or for the final scientific report.

The original PROVENANCE.json and its checksum are preserved. This file records the correction separately; it does not replace the frozen upstream-source lock.
