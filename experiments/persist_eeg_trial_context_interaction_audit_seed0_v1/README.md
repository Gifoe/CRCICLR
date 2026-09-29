# Frozen EEGNet trial-context interaction audit

EEGNet / OpenBMI_MI / seed 0 / folds 0–4. This direction-qualification experiment compares explicit trial–context interaction against trial-only, subtraction and additive-context controls. It reuses the SHA-verified frozen 64D EMBEDDING caches from the immediately preceding recording-context audit; no neural model is trained or updated. OUTER_DEVELOPMENT was historically exposed and is evaluation-only here. Formal final-heldout EEG was never opened.

## Result

The tested interaction premise failed its qualification gates. TRAIN-selected interaction minus TRAIN-selected best noninteraction had pooled subject-equal BA **−0.00163**, 20,000-draw biological-subject bootstrap 95% CI **[−0.01541, +0.01247]**. The selected explicit interaction term's ablation gain was **−0.00083**. True current context exceeded wrong-subject context by **+0.00474** BA, below the prespecified +0.010 threshold, with CI crossing zero. Gates A/B/C failed; D/E passed only their weaker stated conditions. The predeclared decision is `NO_TRIAL_CONTEXT_INTERACTION_ADVANTAGE`, next action `STOP_CONTEXT_INTERACTION_DIRECTION`. The detailed Q1–Q12 answers and fold table are in `outputs/FINAL_REPORT.md`.

This excludes support for the **tested** low-capacity interaction forms, not every possible nonlinear architecture. A single fold was positive while four were negative on the TRAIN-selected primary contrast. No OUTER stage or model was selected after seeing these results. The observed low-rank compatibility coordinates are descriptive and are not biological interpretations.

## Design and provenance

- Frozen sources: previous audit commit `163907c436b47e9c1e3a8a992ac0aa15cab66699`; original source provenance SHA256 `18a7a914f8759c0cc94c353e598eb87cd98ba027be9e8b7e66cfe0cf8991507c`. `protocol/SOURCE_PROVENANCE.json` preserves those exact bytes. The 64D representation is the input to the frozen canonical EEGNet `model.head`, a 64-to-2 affine classifier.
- The source checkpoint bytes and existing frozen embedding NPZ bytes were rehashed at each read. Split, input-normalizer, model-source and pre/post-model-state hashes were compared to the original source provenance; the previous extraction reported equal model-state hashes and zero final-heldout EEG reads. This audit performed no new neural extraction.
- TRAIN_GEOMETRY S1 alone supplied decoder fitting and common h/r scaling. Subject-grouped TRAIN pseudo-target S2 selected B, linear C, bilinear rank/C, and both family winners. All five selections and 200-per-fold TRAIN pairing nulls were completed and hash-linked before any new OUTER interaction evaluation. Selected B across folds: 32, 64, 64, 64, 64.
- The bilinear arm is specifically a TRAIN-label-supervised class-difference cross-moment SVD followed by regularized logistic regression on additive h/r and K products. It is **not** a jointly trained free bilinear matrix. `[h,r,h-r]` is linearly redundant with `[h,r]`; its separate L2 parameterization is a regularization-geometry control. Random linear expansions also vary effective L2 geometry, not nonlinear capacity.
- Each OUTER biological subject supplied an unlabeled first-B current-session reference, then only the remaining trials were scored identically across arms. Context labels were consulted solely for the separately marked, non-deployable class-balance diagnostic. Wrong-subject and TRAIN pairing controls used 200 deterministic derangements per fold; each interaction family had 100 parameter-dimension-matched random linear controls. The bootstrap unit was biological subject, never trial.
- Only compact CSV/JSON/Markdown and code are versioned. Runtime embeddings, checkpoint copies and raw task logs remain on the original server. `protocol/OUTPUT_HASHES.json` maps compact outputs to SHA256 and records the unpushed runtime JSON hashes.

## Failure disclosure

The first fold-0 scheduled TRAIN launch (`PERSIST_EEG_TRIAL_CONTEXT_TRAIN_F0_V1`) failed **before** opening the TRAIN cache because the server's free RAM fell below its 24GB resource gate while unrelated jobs ran. It produced no scientific output and was not retried in place. After memory recovered, the distinct V2 task completed. TRAIN folds 0–4 used `run_v2.py`/`run_fold_v2.ps1`; TRAIN pairing nulls and OUTER folds used version-forward `run_v3.py` (checked in as `code/run.py`)/`run_fold_v3.ps1`. V1/V2 wrappers and V2 TRAIN code are retained for audit. No unrelated server process was stopped or modified.

## Audit surface

`protocol/PROTOCOL_LOCK.json` is the frozen operational design. `protocol/REPRESENTATION_PROVENANCE.json` records checkpoint/split/normalizer/model-state hashes and roles. `outputs/` contains the requested 15 CSV tables, two JSON decisions/audits, and report. `code/validate.py` checks exact file inventory, row counts, per-arm outcome fairness, model-state equality, and fold/pooled arithmetic. `code/test_run.py` covers first-B disjointness, feature dimensions, derangements, and low-rank dimensions.
