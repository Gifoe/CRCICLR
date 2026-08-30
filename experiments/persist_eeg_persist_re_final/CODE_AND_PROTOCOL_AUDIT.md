# Code and protocol audit

This audit records the implementation and data boundary used by the PERSIST-RE
experiment.  It is written before the source search; outcome sessions are not
used for recipe selection.

## Implementations and representation point

- The development backbone is the clean-room `ATCNet` in
  `experiments/persist_eeg_scst_competence_generality_v1/code/specialist_models.py`.
  Its `forward_features` returns the 32-dimensional feature after the
  attention/TCN block and `head` is a two-class linear classifier.
- The official ATCNet and EEGNeX implementations are the corresponding
  braindecode models exposed by `stage1_common.py`; their feature extraction
  points and final linear heads are recorded in that file.
- PERSIST-RE consumes the frozen clean-room feature extractor output and adds a
  trainable final feature block (`Linear` initialized to identity followed by
  `LayerNorm`) and population head.  The low-rank random-effect branch is
  decision-level and receives `stop_gradient(z)` only.

## Development data and splits

- OpenBMI is loaded through the frozen P2 loader and uses the authorized
  inner-train/inner-validation/outcome subject roles.  Source sessions are
  sessions 1 and 2; no sealed holdout is enumerated.
- WBCIC is loaded through the frozen P3 loader.  Source roles are model-fit,
  validation-discovery, and outcome; source sessions are 0 and 1.  Session 2
  is reserved for a post-freeze utility evaluation.
- Subject/session indexing is normalized to string subject IDs and integer
  session IDs.  All training losses and all reported metrics are averaged at
  the biological-subject level.

## Existing evidence and controls

The repository preserves the prior negative terminals (invariance, SCST,
mixed-effect transport, and constructive-search exhaustion).  Existing matched
ERM, DANN/MMD/GroupDRO-related implementations were inspected in the P2/P3
experiments.  This experiment does not silently reinterpret those results as
confirmation.  GroupDRO, Mixup, prospective-only, random-effect-only, and an
adversarial mixed-effect control are implemented with the same feature scope,
epochs, folds, and seeds.

## Bootstrap and authorization

Paired bootstrap resamples biological subjects, not trials.  The source gate is
evaluated only on the authorized development transitions.  `DATA_ACCESS_LOCK`,
`PERSIST_RE_SOURCE_LOCK`, and `PERSIST_RE_FINAL_METHOD_LOCK` are created before
any confirmation architecture is opened.  Outer WBCIC subjects and the
OpenBMI sealed holdout are not read by this experiment.

## Resource ledger

| Resource | Status | Use |
|---|---|---|
| OpenBMI P2 inner train/validation + source outcome | DEVELOPMENT_KNOWN | source search |
| WBCIC P3 model-fit/validation-discovery + source outcome | DEVELOPMENT_KNOWN | source search |
| WBCIC session 2 | UNTOUCHED before source lock | confirmation utility only |
| WBCIC outer 10 | SEALED | never opened |
| OpenBMI sealed outer holdout | SEALED | never opened |

