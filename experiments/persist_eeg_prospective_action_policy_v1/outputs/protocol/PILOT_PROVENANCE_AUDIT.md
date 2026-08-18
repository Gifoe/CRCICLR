# Pilot provenance audit

`OUTER_TEST_USED = false`

No sealed WBCIC outer subject identifier, sample, embedding, label, or outcome
was loaded. All selected artifacts explicitly report no outer-test use.

| id | decision_unit | actions | outcome | model_use | eligible |
| --- | --- | --- | --- | --- | --- |
| signed_utility_v3_1 | run x persistence block | KEEP, ERASE_BLOCK | signed CE utility | audit only; direct U is not reused on its own outcome group | True |
| shared_geometry_v1_2 | run x paradigm x block | diagnostic only | cross-paradigm geometry/utility | provenance and feature semantics only | True |
| p5_p6 | run and subject for strict-inductive fusion | BASE, PROTECTED_FUSION, ALL_PERSISTENCE, RANDOM | subject-balanced BA | oracle/action history audit; incompatible with sample router | True |
| historical_router | individual OpenBMI trial | KEEP, ERASE, AMPLIFY, GEOMETRY | subject-balanced BA from trial decisions | sample-family modelling with subject-grouped validation | True |
| persist_cf | run/configuration; nested subject folds | BASE, CF, DUPLICATE, FULL, HISTORICAL, RANDOM | BA/NLL by configuration | audit only; configuration-level unit incompatible with trial/block units | True |
| dda_v1 | run x audit fold x persistence block | NO_OP, SUPPRESS_BLOCK | held-out outcome-role CE and BA change | block-family modelling; same-cell U excluded and cross-fitted U constructed | True |
| wbcic_eegnet | backbone x cross-fit fold x persistence block | NO_OP, SUPPRESS_BLOCK | development-only S3 subject-balanced delta BA | WBCIC development block family; cross-fitted U only | True |
| wbcic_multibackbone | backbone x cross-fit fold x persistence block | NO_OP, SUPPRESS_BLOCK | development-only S3 subject-balanced delta BA | competent WBCIC backbones only; FBCNet excluded after competence failure | True |

## Pooling decision

The historical router acts per trial, DDA acts per run/fold/block, P5/P6 acts
per run or subject, and the WBCIC audit acts per backbone/fold/block. Treating
these rows as exchangeable would be pseudo-replication. This experiment fits
and evaluates a separate policy inside each compatible decision family.

Same-cell realised signed utility is an outcome, not a legal predictor. DDA
and WBCIC therefore receive only leave-group-out utility estimates constructed
without the target outcome cell.
