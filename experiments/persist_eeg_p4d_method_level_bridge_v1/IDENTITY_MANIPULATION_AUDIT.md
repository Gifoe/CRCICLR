# Identity Manipulation Audit

Canonical lambdas were selected from the 15 complete S4 source-only runs per lambda using `S_I_abs = identity_symmetric_ERM - identity_symmetric_method`. Competence requires median suppression above zero and at least 60% positive runs. Ties select the smaller lambda. Future BA/F1/CE was not accessed.

| method | status | lambda_star | median_S_I_abs | fraction_positive |
| --- | --- | --- | --- | --- |
| DANN | IDENTITY_MANIPULATION_COMPETENT | 0.100000 | 0.015014 | 0.800000 |
| MMD | IDENTITY_MANIPULATION_INCOMPETENT | nan | nan | nan |
| CORAL | IDENTITY_MANIPULATION_INCOMPETENT | nan | nan | nan |
