# UGCR-V3: Utility-Guided Selective Complement Routing

Seed-zero EEGNet experiment for the four prespecified tasks and folds 0–4. It
reuses the byte-verified canonical V1/V2 final-refit EEGNet checkpoints and
freezes every parameter and batch-normalization buffer. It adds no trainable
parameters.

The final-refit spatial and successor projectors, the unsupervised top-16
spatial-complement PCA basis, and V2.5 P-mediated direction utilities are
recomputed using only the non-final-heldout refit-training pool. The protocol
lock is written before the first V3 heldout array is loaded. Since these
subjects were accessed in prior V1/V2 work, all final results are labelled
`POST_HELDOUT_DEVELOPMENT_EVALUATION`.

## Stages

Run on the configured experiment server with the V1/V2 caches, checkpoints,
anchors, and Python dependencies available:

```bash
python experiments/persist_eeg_selective_cp_routing_v3_seed0/code/run.py preflight
python experiments/persist_eeg_selective_cp_routing_v3_seed0/code/run_queue.py prepare --workers 2
python experiments/persist_eeg_selective_cp_routing_v3_seed0/code/run.py lock
python experiments/persist_eeg_selective_cp_routing_v3_seed0/code/run_queue.py final-eval --workers 2
python experiments/persist_eeg_selective_cp_routing_v3_seed0/code/run.py aggregate
```

Two workers were used on the configured server because its experiment container
has a 58 GiB memory limit; four-worker ERP preparation exceeded that limit.

`preflight` verifies source locks and baseline checkpoint hashes without
loading EEG arrays. `prepare` reads only the non-final-heldout refit-training
pool. `lock` freezes all fold-local utilities, signs, coefficients, shuffle
and random controls, checkpoint/projector/basis hashes, metric definitions,
and code hashes. `final-eval` checks this lock before reading any heldout EEG.
`aggregate` computes five-fold probability means, subject-level scores,
paired bootstrap contrasts, rescue/harm, and post-hoc mechanism diagnostics.

## Variants

`BASELINE`, `PCA_POSITIVE_ONLY_ROUTING`, `SHUFFLED_UTILITY_ROUTING`,
`PCA_SIGNED_UTILITY_ROUTING` (UGCR-V3), and five deterministic random-basis
controls. Shuffle and random controls preserve the exact primary routing
deviation multiset. The random-basis controls use five prelocked orthonormal
bases in the spatial complement.

Do not change code, bases, utilities, routing, metrics, or evaluation after
`V3_PROTOCOL_LOCK.json` is written. Do not describe the heldout results as
untouched confirmation or interpret functional utility as biological
causality.
