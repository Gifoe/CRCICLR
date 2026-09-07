# P4B Engineering Recovery — Serialized Basis SHA

P4B stopped before the first S5 future-outcome evaluation because the P4A
source cube's whole-matrix `persistence_basis_sha256` did not match the
serialized `persistence_basis.npz` matrix. This is a metadata-only mismatch:

- all 15 S5 checkpoints remain byte-exact;
- 11/15 S5 runs have all 8 serialized direction hashes byte-exact;
- the remaining four S5 runs have only 1, 1, 1, and 2 byte-SHA differences;
- for every differing direction, the frozen source `D_finite`, `O_task`,
  geometry strength, and cross-session persistence reproduce the source cube
  within `rtol=1e-7, atol=1e-9` (maximum absolute discrepancy below
  `3.6e-15`);
- no S5 future utility and no S4/S6 future utility was accessed during this
  audit.

The engineering recovery therefore retains the serialized center/basis and
requires either an exact direction SHA or strict source-side numerical
equivalence for every direction. It does not change direction order, signs,
normalization, discovery/reserve assignment, models, alpha, thresholds,
outcome definitions, bootstrap, or terminal gates. Any failed numerical
equivalence still aborts the pipeline.
