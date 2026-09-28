# Operational failure disclosures

- Fold0 TRAIN extraction attempt V1 (`PERSIST_EEG_CONTEXT_RELATIVE_F0_TRAIN_EXTRACT_V1`) ended with scheduled-task result 1 after creating an empty feature directory. It produced no scientific feature file or provenance. The V1 PowerShell `*>>` logging path did not retain a Python traceback; the precise cause is unconfirmed. The empty directory and task log remain on the original server. Attempt V2 used a new output directory and separate captured stdout/stderr; it completed with result 0 and immutable-model/hash checks. No V1 result was overwritten or used.

Any later task failures must be appended here before publication. Warnings emitted by PyTorch about `padding='same'` are not treated as result failures.
