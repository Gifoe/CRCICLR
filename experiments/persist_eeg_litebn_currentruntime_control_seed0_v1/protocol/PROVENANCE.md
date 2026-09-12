# Provenance and fail-closed checks

The runner imports the same recovered historical sources and current-runtime
execution infrastructure used by the prior TSW/HE96 experiments. It uses the
project `legacy_init.construct_exact` recovery procedure, the original stored
OpenBMI episode manifest, historical fold split, historical normalizer record,
and frozen LiteBN result files for metric replay. Every source hash and actual
runtime setting is written to `CURRENT_RUNTIME_METADATA.json`.

Initialization, manifest, or normalizer disagreement aborts before training.
Historical metrics are replayed subject-by-subject before reporting each
current-runtime comparison.
