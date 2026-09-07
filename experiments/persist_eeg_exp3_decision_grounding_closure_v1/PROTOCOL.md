# Protocol lock

The protocol is frozen in `protocol/EXP3_PROTOCOL_LOCK.json` before the new
identity-versus-decision comparison. The primary rows are the 215 cells in the
historical DDA cross-fit table: six OpenBMI MI runs, five audit folds per run,
and all canonical blocks. The outcome is the frozen `outcome_ce_effect`.

The only identity metric is V1.2's symmetric cross-session subject-ID identity
skill, `0.5*((log(K)-CE_S1_to_S2)+(log(K)-CE_S2_to_S1))`. Identity evidence for
a block is the full fit-subject skill minus skill after erasing that block. The
subject-ID probe is fit separately on each DDA fit bank; outcome subjects are
not used to fit it. A subject-level evaluation-label permutation null checks
that the full representation has measurable cross-session identity evidence.

Decision dependence is not redefined: finite `decision_logit_rms` is the
primary DDA quantity and Jacobian/local dependence remains corroborating.
Protected assignments, block definitions, controls, splits, outcome, and
source DDA outputs are immutable.

Consequence prediction uses predeclared LORO models with train-run
standardization and ridge alpha 1.0 inherited from DDA-C:

* M0: persistence, geometry, rank;
* MI: M0 plus identity evidence;
* MD: M0 plus finite decision dependence;
* MID: M0 plus both.

Primary inference is run-clustered (six runs), with deterministic bootstrap
and exact six-run sign-flip tests. WBCIC is only a supporting audit; its ten
outer subjects remain sealed.
