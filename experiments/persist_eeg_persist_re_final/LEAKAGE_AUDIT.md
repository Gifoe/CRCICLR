# Leakage audit

Source recipe selection reads only model-fit/validation roles.  Source outcome
rows are read after selection for reporting.  No trial IDs, session IDs, labels
as random-effect inputs, test-time adaptation, or test subject IDs are used.
The code path contains no outer/sealed loader.  The paired bootstrap unit is a
biological subject.  Runtime caches and checkpoints remain outside Git.

