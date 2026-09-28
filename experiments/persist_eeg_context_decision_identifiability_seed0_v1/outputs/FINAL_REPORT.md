# Context decision identifiability, seed 0

This is a frozen-backbone direction-qualification test, not a Context Reprogramming model. OUTER_DEVELOPMENT was historically exposed in earlier mechanism work; formal final-heldout EEG reads: 0.

Q1. Labelled S1 oracle minus strongest TRAIN-chosen global head: -0.0578 BA, macro-F1 -0.0583, NLL +0.3488; >+1pp 17.5%, <-1pp 82.5%, median -0.0500, worst-quartile mean -0.1540; Gate A FAIL.
Q2. Subject corrections: median norm 2.9342; median head cosine 0.4815; median pairwise correction cosine 0.0517 (IQR -0.0519–0.1674). Numerical heterogeneity alone does not establish transferable context information.
Q3. Oracle minus mean correction: -0.0587 BA.
Q4. Parameter cosine mean 0.0946; function-response cosine mean 0.1237. Future-session BA remains primary.
Q5. Context minus global: -0.0030 BA; context minus mean correction: -0.0040 BA; Gate B FAIL.
Q6. Pooled labelled-oracle headroom recovery: undefined (nonpositive headroom).
Q7. Fold-level context exceeds shuffled-context p95 in 1/5 folds.
Q8. Final TRAIN-selected descriptor counts: {'CTX_MOMENTS': 0, 'CTX_COV': 1, 'CTX_PRED': 3, 'CTX_COMBINED': 1}. TRAIN LOSO family audits are in CONTEXT_DESCRIPTOR_AUDIT.csv; OUTER never selects a family.
Q9. P/C augmentation minus matched generic combined descriptor: -0.0015 BA (secondary; no P/C classifier intervention).
Q10. STOP_CONTEXT_MODEL_DIRECTION. Primary interpretation: NO_MEANINGFUL_PERSONALIZATION_HEADROOM.
Unseen subject IDs have no learned embedding and are not usable as a transductive input; no subject ID enters the context mapper.

## Fold table

|fold|TRAIN subjects|OUTER subjects|dim|global type|global BA|oracle BA|headroom|mean BA|nearest BA|context BA|context gain|vs mean|recovery|parameter cosine|function cosine|rank|ridge alpha|descriptor|full trials|B8|B16|B32|B64|shuffle p95|PC gain|interpretation|
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
|0|26|8|64|NATIVE_POPULATION_HEAD|0.7850|0.7312|-0.0537|0.7775|0.7375|0.7512|-0.0337|-0.0262|undefined|0.105|0.210|4|0.1|CTX_PRED|100|0.7500|0.7562|0.7438|0.7500|0.7838|+0.0000|headroom_negative|
|1|26|8|64|NATIVE_POPULATION_HEAD|0.7750|0.7425|-0.0325|0.7800|0.7800|0.7812|+0.0062|+0.0012|undefined|0.154|0.413|8|100.0|CTX_PRED|100|0.7850|0.7800|0.7838|0.7850|0.7825|-0.0038|headroom_negative|
|2|26|8|64|NATIVE_POPULATION_HEAD|0.6887|0.6262|-0.0625|0.6950|0.6550|0.7037|+0.0150|+0.0087|undefined|-0.033|-0.248|8|100.0|CTX_COV|100|0.6975|0.7013|0.6987|0.6987|0.7050|-0.0012|headroom_negative|
|3|26|8|64|GLOBAL_REFIT_HEAD|0.7163|0.6225|-0.0938|0.7200|0.6750|0.7200|+0.0037|+0.0000|undefined|0.155|0.157|1|100.0|CTX_PRED|100|0.7163|0.7137|0.7137|0.7200|0.7176|-0.0025|headroom_negative|
|4|26|8|64|NATIVE_POPULATION_HEAD|0.7163|0.6700|-0.0463|0.7137|0.6950|0.7100|-0.0062|-0.0037|undefined|0.094|0.087|4|1.0|CTX_COMBINED|100|0.7063|0.7238|0.7125|0.7125|0.7213|-0.0000|headroom_negative|

Pooled oracle headroom: -0.0578; pooled context gain: -0.0030; pooled recovery: None.
Gate A: FAIL; Gate B: FAIL.
Primary interpretation: `NO_MEANINGFUL_PERSONALIZATION_HEADROOM`. Exact next action: `STOP_CONTEXT_MODEL_DIRECTION`.
Formal final-heldout EEG reads: 0. Branch: `codex/persist-eeg-context-decision-identifiability-seed0-v1`.
Final commit SHA: populated after reviewed GitHub push in the accompanying commit record.
