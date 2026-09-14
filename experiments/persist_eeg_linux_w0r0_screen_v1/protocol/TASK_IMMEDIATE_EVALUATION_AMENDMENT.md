# Task-immediate evaluation amendment

User-directed sequencing amendment: once all five W0R0 seed-0 checkpoints for
one task are frozen, that task is immediately evaluated against the strict
historical B0 replay on both development outer and
`DEVELOPMENT_MODEL_SELECTION_DATA`. Training of later tasks does not alter the
already frozen task checkpoints. No new sealed or final-test cohort is opened.

This changes only reveal timing and reporting order. Architecture, data splits,
normalizers, optimizer, batches, epochs, checkpoint selection, and evaluation
metrics remain unchanged.
