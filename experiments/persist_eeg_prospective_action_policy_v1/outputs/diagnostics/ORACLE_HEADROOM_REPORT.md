# Oracle headroom decomposition

The exact historical OpenBMI action menu was reconstructed from the stored
subject-cross-fitted OOF logits. The primary values below average balanced
accuracy within subject before averaging subjects and runs.

## OpenBMI sample router

- Oracle action gain: `0.118701` BA.
- Best non-trivial fixed action: `AMPLIFY` with
  `-0.014902` BA.
- Best fixed action including NO_OP: `0.000000` BA.
- Action-selection value: `0.118701` BA.
- Subjects with at least one oracle rescue: `1.000`.
- Largest subject contribution: `0.060` of total oracle gain.
- Largest run contribution: `0.187`.

All three fixed interventions are net harmful even though each creates some
oracle rescues. This is exactly the rare-rescue/frequent-harm regime; oracle
headroom is not evidence that a legal router can recover it.

## Block families

- DDA oracle gain: `0.000633`;
  selection value `0.000633`.
- WBCIC development oracle gain: `0.001614`;
  selection value `0.001614`.

The block values are diagnostic upper bounds from choosing suppression only
when its realised held-out consequence is positive. They are not deployable
policies. `OUTER_TEST_USED = false`.
