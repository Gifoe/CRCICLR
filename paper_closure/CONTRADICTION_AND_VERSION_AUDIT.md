# PERSIST-EEG Contradiction and Version Audit

## Audit status and resolution rules

This audit resolves result conflicts for paper closure. It is not a chronological
summary and it does not treat the newest timestamp as automatically authoritative.
For every chain below, authority is assigned by frozen scientific semantics,
implementation validity, provenance, validator status, and scope.

Four recurrent conflict classes are distinguished:

- **Superseded preliminary evidence:** a partial, single-seed, or development
  result was later evaluated under the intended frozen multi-seed or prospective
  protocol.
- **Invalid implementation or provenance:** an earlier result cannot support a
  scientific conclusion because its sampling, reconstruction, metric, or upstream
  provenance was not reproducible or did not implement the stated estimand.
- **Scoped partial support versus a negative primary gate:** a secondary,
  subgroup, or downstream mechanism result is valid within scope but cannot
  override the preregistered primary failure.
- **Experiment-local purity versus project-global data status:** a resource can
  be untouched in one workflow while already observed, or otherwise invalid for
  confirmation, across the project.

The notation `NO_GIT_COMMIT_SERVER_ONLY_EVIDENCE` below is deliberate. The early
Stage-0/P2/P3 package was server-only and is therefore identified by the SHA256
records in `paper_closure/evidence_sources/server_only_early/SOURCE_MANIFEST.csv`,
not by an invented Git commit.

No sealed outcome was opened for this audit. In particular, the content of
`experiments/persist_eeg_final_model_v8/outputs/final_candidate/INTERNAL_HOLDOUT_RESULTS.json`
was not read. Only the already-audited existence of that non-empty artifact, its
lock, and its partition identity are used.

## 1. OpenBMI Stage-0 seed-0 partial result -> P2 multi-seed closure

**Conflict class:** superseded preliminary evidence.

1. **Earlier result.** The server-only Stage-0 summary was explicitly
   `PARTIAL_CORE_ONLY`, used seed 0, and left the official gate
   `NOT_EVALUATED`. It nevertheless showed an early MI erasure signal: EEGNet
   BA delta `-0.047600` and ConvTransformer BA delta `-0.021209`; the corresponding
   EEGNet ERP and SSVEP deltas were `+0.000010` and `-0.010436`. Authoritative
   source: `paper_closure/evidence_sources/server_only_early/delivery/persist_eeg_stage0/OPENBMI_STAGE0_SEED0_CORE_SUMMARY.json`
   (`NO_GIT_COMMIT_SERVER_ONLY_EVIDENCE`; checksum in the source manifest).
2. **Problem.** A single-seed partial run cannot establish multi-seed persistence
   utility, task generality, matched-random specificity, residualized robustness,
   or the frozen Gates A-E. The near-zero ERP seed-0 value is not a contradiction
   to a later small multi-seed effect; it is an underpowered preliminary estimate.
3. **Repaired result.** P2 ran the frozen five-seed closure and passed all Gates
   A-E. MI `erase_UL` mean BA delta was `-0.04646296`, 95% hierarchical-bootstrap
   CI `[-0.05696343, -0.03598148]`, with 5/5 negative seeds. Residualized MI was
   `-0.04688889`, CI `[-0.05696389, -0.03687037]`; ERP was `-0.00316461`, CI
   `[-0.00501085, -0.00116175]`; SSVEP was `-0.00975926`, CI
   `[-0.01366667, -0.00612963]`. Matched-random MI `U_L` erasure was near neutral
   (mean `-0.000037745`). Cross-session verification erasure was `-0.11075198`,
   CI `[-0.12214669, -0.10008145]`, and the target-dependence contrast was
   `-0.04329835`, CI `[-0.05426327, -0.03231043]`. Authoritative source:
   `paper_closure/evidence_sources/server_only_early/outputs/persist_eeg_p2p3/P2_FINAL_REPORT.json`
   (`NO_GIT_COMMIT_SERVER_ONLY_EVIDENCE`; checksum in the source manifest).
4. **Final authoritative interpretation.** The terminal is
   `P2_PASS_MULTI_SEED_PERSISTENCE_UTILITY`. The five-seed P2 result, not the
   Stage-0 seed-0 point estimates, is authoritative for persistence-erasure
   consequence and multi-task calibration.
5. **Manuscript consequence.** Use P2 for main quantitative claims that removal
   of the learned persistent subspace is task-harmful in the tested OpenBMI
   settings and substantially more harmful than matched random removal in MI.
   Keep Stage-0 only as historical provenance; do not quote its seed-0 ERP value
   as evidence of absence.

## 2. Positive event learning -> negative P3 trajectory/compression claim

**Conflict class:** scoped partial support versus a negative primary gate.

1. **Earlier result.** The trajectory audit observed real learning: MI event
   verification increased from epoch 0 to the best checkpoint by `+0.190222`,
   CI `[0.172777, 0.207445]`, and long cross-session verification increased by
   `+0.155701`, CI `[0.140579, 0.170820]`. These positive subclauses could be
   misread as evidence that training compresses subject structure into a stable,
   low-rank, task-useful subspace. Source:
   `paper_closure/evidence_sources/server_only_early/outputs/persist_eeg_p2p3/P3_FINAL_REPORT.json`.
2. **Problem.** Event learning was not the full frozen trajectory claim. Broad
   compression, low-rank concentration, and multi-seed compression consistency
   were separate required clauses. The mean `U_L` rank increased from `2.8` to
   `4.88`, and the number of seeds showing the required compression direction was
   `0`.
3. **Repaired result.** The final P3 evaluator retained the positive event-learning
   observations but set `broad_compression=false`,
   `low_rank_concentration=false`, and `multi_seed_consistency=false`, yielding
   `P3_TRAJECTORY_CLAIM_NOT_SUPPORTED`. The joint decision file records
   `GO_PERSIST_UTILITY_NO_TRAJECTORY_CLAIM` and confirms that P4 training had not
   started. Sources:
   `paper_closure/evidence_sources/server_only_early/outputs/persist_eeg_p2p3/P3_FINAL_REPORT.json`
   and `paper_closure/evidence_sources/server_only_early/outputs/persist_eeg_p2p3/P2P3_FINAL_DECISION.json`
   (`NO_GIT_COMMIT_SERVER_ONLY_EVIDENCE`; checksums in the source manifest).
4. **Final authoritative interpretation.** Task learning and stronger subject
   verification during training are supported; a general low-rank
   compression/emergence trajectory is not.
5. **Manuscript consequence.** Retain the P2 intervention-consequence result.
   Do not explain it through a claimed universal training-time compression or
   low-rank emergence process. P3 belongs in a boundary/appendix analysis, not
   as a positive mechanism claim.

## 3. Signed V3 -> deterministic Signed V3.1

**Conflict class:** invalid provenance followed by a reproducibility repair.

1. **Earlier result.** Signed V3 reported that protected persistent directions
   were harmful to erase, but its terminal was
   `P4_SIGNED_PERSISTENCE_HAS_NO_ACTIONABLE_HEADROOM`: mean actionable gain was
   `-0.00028935`, CI `[-0.00086806, 0]`. Method training was not started and the
   outer resource was not used. Source at commit
   `dff9305415f3f31237f8e559ba0173d040faaffa`:
   `experiments/persist_eeg_p4_signed/outputs/persist_eeg_p4_signed/audit_v3/SIGNED_AUDIT_DECISION.json`.
2. **Problem.** V3 used process-dependent Python `hash()` sampling and did not
   persist the selected indices. The exact protected/random assignments could
   not be replayed across processes, so the assignment-specific positive
   evidence lacked adequate provenance. This is a reproducibility defect, not
   evidence that the scientific headroom terminal was positive or negative for a
   different reason.
3. **Repaired result.** Signed V3.1 replaced the sampler with SHA256-derived
   deterministic seeds and persisted all indices. It reproduced protected
   assignment in 6/6 MI runs and found protected erasure more harmful than
   matched-rank random erasure in 6/6, with mean BA difference `0.09592448`, CI
   `[0.06791811, 0.12301360]`. Terminal:
   `PERSISTENCE_UTILITY_ASSIGNMENT_REPRODUCIBLE`. Source at commit
   `c7e68c18301da6c04100686f59e9523b1d7d9575`:
   `experiments/persist_eeg_p4_signed_v3_1/results_v3_1/SIGNED_V3_1_FINAL_REPORT.json`.
4. **Final authoritative interpretation.** V3.1 is the sole authoritative source
   for the positive protected-assignment consequence. It does not overturn the
   distinct V3 finding that the tested source-defined suppression policy had no
   actionable headroom. Protected consequence and prospective actionability are
   different estimands.
5. **Manuscript consequence.** Use V3.1 for the protected-versus-matched-random
   result. If the no-headroom result is discussed, attribute it to the V3
   actionability audit and keep the estimands separate. Never combine the V3.1
   consequence effect with V3 into a claim that suppression is beneficial.

## 4. Shared Geometry V1/V1.1 -> provenance-valid V1.2

**Conflict class:** invalid upstream provenance followed by a full replay.

1. **Earlier result.** Shared Geometry V1 (commit
   `0d73caa4d5a9e927d2385dab5d0bd68afb44c641`) produced an initial geometry
   package. V1.1 (commit `f92660090bafb978c7aaacaf0bf9e0618663bc54`)
   then closed as `SHARED_GEOMETRY_V1_1_BLOCKED_BY_UPSTREAM_PROVENANCE`.
   Sources: `experiments/persist_eeg_p4_shared_geometry/results_v1_1/PREVIOUS_SHARED_GEOMETRY_V1_INVALID.json`,
   `experiments/persist_eeg_p4_shared_geometry/results_v1_1/SHARED_GEOMETRY_FINAL_REPORT.json`,
   and the run-level `BASIS_RECONSTRUCTION_VALIDATION.json` files under
   `results_v1_1/runs/`.
2. **Problem.** The block was caused by the unreplayable Signed V3 sampling
   ancestry. It was an upstream provenance failure, not a negative test of
   shared geometry and not evidence that the proposed mechanism failed.
3. **Repaired result.** V1.2 replayed the analysis from the deterministic Signed
   V3.1 assignments and passed Gates A-F. The terminal was
   `SHARED_GEOMETRY_V1_2_PASS`. Key estimates were: Gate A `0.336709`, CI
   `[0.239373, 0.436462]`; Gate B `0.158137` BA, CI
   `[0.121542, 0.197137]`; Gate C `0.339453`, CI
   `[0.247125, 0.436688]`; Gate D `0.403651`, CI
   `[0.315825, 0.499107]`; geometry coefficient `0.0244634`, CI
   `[0.0136531, 0.0422572]`; Spearman `0.555849`; 43 blocks. Source at commit
   `c7e68c18301da6c04100686f59e9523b1d7d9575`:
   `experiments/persist_eeg_p4_shared_geometry/results_v1_2/SHARED_GEOMETRY_FINAL_REPORT.json`.
4. **Final authoritative interpretation.** V1.2 is the authoritative shared-
   geometry result. V1/V1.1 are provenance history. V1.2 is still development
   evidence; neither outer evaluation nor method training was performed.
5. **Manuscript consequence.** Report only V1.2 quantitative geometry evidence.
   Describe V1/V1.1, if needed, as a provenance invalidation and repair, never as
   failed scientific replications.

## 5. Matched Identity V1/V1.1 -> final V1.2 non-identifiability closure

**Conflict class:** infeasible control design, metric-resolution failure, and a
final predeclared measurement repair.

1. **Earlier result.** V1 passed held-out persistence (`R_persist=0.566274`, CI
   `[0.434534, 0.703639]`) but stopped at
   `MATCHED_NONPROTECTED_CONTROL_UNAVAILABLE`: one fold/seed required a rank-8
   control from an eight-dimensional eligible pool, giving only one legal
   combination instead of at least 20. V1.1 changed the causal unit to already-
   frozen blocks and obtained 6/6 run coverage, but the original G2 code accepted
   a zero-drop protected arm whenever the P/N difference was within `0.01` BA.
   After the required semantic repair, V1.1 correctly closed as
   `IDENTITY_MATCH_FAILED`: MEDIUM identity drops were `P=0.0` and
   `N=0.00287474`; task harms were `H_P=0.0`, `H_N=0.00010927`, with
   `Delta_H=-0.00010927`, CI `[-0.00062401, 0.00034986]`. Sources at commits
   `ab496f203625388249601986c3367e53f55bb9d8` and
   `cc2cc91b67b73a739267325c02a2f9cab128798c`:
   `experiments/persist_eeg_matched_identity_causal_v1/outputs/SCIENTIFIC_REPORT.md`
   and `experiments/persist_eeg_matched_identity_causal_v1_1/outputs/SCIENTIFIC_REPORT.md`.
2. **Problem.** V1 was a train-only control-feasibility failure, not a negative
   causal result. V1.1's pre-repair G2 omitted the required measurable reduction
   in both arms; its repaired result also showed that top-1 subject-ID BA and the
   21-point dose grid selected alpha 0 for most protected MEDIUM interventions.
   Therefore the causal effect was not identified. Neither failure permits a
   positive or negative claim about utility at matched identity reduction.
3. **Repaired result.** V1.2 prospectively replaced the discrete top-1 metric
   with symmetric cross-session subject-ID log-loss skill, retained the frozen
   protected blocks and V1.1 controls, and added a train-only noise-floor gate.
   Zero of 10 protected blocks and zero of 6 runs supported a measurable
   protected intervention; no MEDIUM block entered the held-out task phase.
   Terminal: `PROTECTED_PERSISTENCE_NOT_IDENTITY_BEARING_UNDER_OPERATIONAL_METRICS`;
   final label and Theory-3 state: `NOT_IDENTIFIABLE`. Source at commit
   `96a96a89d63aaca6560544e56fad070a1316f840`:
   `experiments/persist_eeg_matched_identity_causal_v1_2/SCIENTIFIC_REPORT.md`,
   `experiments/persist_eeg_matched_identity_causal_v1_2/outputs/FINAL_DECISION.json`,
   and `experiments/persist_eeg_matched_identity_causal_v1_2/outputs/G2_EQUIVALENCE_TEST.json`.
4. **Final authoritative interpretation.** Under the final operational identity
   metrics, the protected persistent blocks were not measurably identity-bearing
   enough to identify the proposed matched-removal causal comparison. The final
   result is non-identifiability, not evidence of zero causal effect and not a
   negative causal result.
5. **Manuscript consequence.** Use this chain only as calibrated evidence that
   persistence/protection cannot be equated with the measured identity construct.
   Do not report `Delta_H` as a matched-identity causal estimate, do not claim
   Theory 3 was falsified, and do not imply that V1/V1.1 were independent
   negative replications.

## 6. DDA-B/DDA-C passes do not override DDA-A failure

**Conflict class:** downstream mechanism support versus an incomplete primary
mechanism chain.

1. **Earlier result.** Within the frozen Decision Dependence Audit, DDA-B passed:
   protected/random Jacobian ratio `3.445844`, CI
   `[2.624825, 4.289905]`; finite-logit ratio `1.679936`, CI
   `[1.566798, 1.799979]`; protected exceeded matched non-protected blocks in
   5/6 runs, and signed utility agreed with held consequence in 6/6. DDA-C also
   passed: baseline LORO RMSE `0.04597840`, full RMSE `0.03149284`, relative
   improvement `31.5051%`, run-cluster CI `[22.8293%, 40.9711%]`, 6/6 improved
   runs, permutation `p=0.000200`.
2. **Problem.** The frozen protocol required a consistent A-B-C chain. DDA-A
   failed its behavioral-null/equivalence gate: relative representation movement
   was `0.227604`, but flip rate was `0.017725` and total variation was
   `0.013083`; their one-sided 95% upper bounds exceeded the respective `0.01`
   limits, and the matched-random logit ratio (`1.009803`) did not meet the
   alternative `<=0.5` rule. A nonsignificant or small effect cannot be relabeled
   as equivalence.
3. **Repaired result.** The final report aggregates the frozen gates rather than
   promoting B/C alone: `DDA_A_FAIL`, `DDA_B_PASS`, `DDA_C_PASS`, terminal
   `DDA_PARTIAL_MECHANISM_ONLY`, stop state
   `STOP_AGDI_DDA_CHAIN_INCOMPLETE`, and AGDI/external actionability authorization
   `false`. Protocol commit:
   `1eca3976d62d38fb4291e217ca06add484babd41`; final commit:
   `78f010644e86639d44a844558ab37bd865815082`. Sources:
   `experiments/persist_eeg_dda_v1/outputs/protocol/DDA_PROTOCOL_LOCK.json`,
   `experiments/persist_eeg_dda_v1/outputs/results/DDA_A_RESULT.json`,
   `experiments/persist_eeg_dda_v1/outputs/results/DDA_B_RESULT.json`,
   `experiments/persist_eeg_dda_v1/outputs/results/DDA_C_RESULT.json`, and
   `experiments/persist_eeg_dda_v1/outputs/scientific_report.md`.
4. **Final authoritative interpretation.** Decision-dependence activity and its
   incremental prediction of held intervention consequence are supported in the
   frozen OpenBMI MI audit. The stronger CF behavioral-null explanation and the
   complete actionability chain are not supported.
5. **Manuscript consequence.** DDA-B/C may support a scoped mechanistic statement
   about decision dependence. They cannot authorize AGDI, establish a complete
   mechanism, or be presented as prospective actionability. The DDA-A failure
   must be visible wherever the DDA result is summarized.

## 7. P4B -> P4C safety boundary -> P4D method bridge

**Conflict class:** secondary/local boundary evidence versus negative primary and
method-level gates.

1. **Earlier result.** P4B prospectively tested whether identity plus task
   entanglement predicted future suppression utility. Its primary terminal was
   `P4B_IDENTITY_RELIABILITY_CONDITION_NOT_SUPPORTED`: M0 RMSE `0.008142`, MI
   RMSE `0.007967`, MINT RMSE `0.022864`; the interaction CI crossed zero and
   Gates G1-G3 failed. S4/S6 future utilities were untouched. Source at result
   commit `b4d6fcabf4e8e33501d32c9bb1b70625ac93106b` and closure commit
   `8b26c073cea98743f73734cff4f60b58c8e3fe71`:
   `experiments/persist_eeg_p4b_identity_reliability_discovery_v1/P4B_FINAL_REPORT.md`.
2. **Problem.** A later, separately frozen regime or safety analysis cannot
   retroactively change the P4B estimand or primary gates. In particular, a
   favorable-looking coarse regime contrast is not evidence that the P4B
   interaction predictor worked, and method manipulation competence is not
   evidence that the moderator predicts generalization benefit.
3. **Repaired result.** P4C independently closed as
   `P4C_SAFETY_BOUNDARY_PARTIAL_SUPPORTED`: pooled `DeltaRegime=0.004170`, CI
   `[-0.001973, 0.011134]`; low-entanglement utility `-0.000838`; high-entanglement
   utility `-0.005008`; explicit conclusion `LOW_E_SUPPRESSION_NOT_BENEFICIAL`.
   P4D then closed as `P4D_METHOD_LEVEL_BRIDGE_NOT_SUPPORTED`; only DANN passed
   manipulation competence and the primary interaction was `0.017616`, CI
   `[-0.009110, 0.041107]`. Sources at commits
   `cc4b6bb3dc9f5182a80b1c658ccb2c278d397226` and
   `57d5e4f1ae0a7c80d95ca27983fedad2ec3f690c`:
   `experiments/persist_eeg_p4c_suppression_safety_validation_v1/P4C_SAFETY_FINAL_REPORT.md`
   and `experiments/persist_eeg_p4d_method_level_bridge_v1/P4D_FINAL_REPORT.md`.
4. **Final authoritative interpretation.** P4B remains a negative primary result.
   P4C contributes only a coarse, partial safety-boundary observation. P4D does
   not establish that task entanglement moderates a real subject-invariance
   method's generalization effect. P4E was not authorized.
5. **Manuscript consequence.** Present the chain as targeted falsification of the
   jump from source diagnostics to future suppression utility. Do not write that
   P4C rescued P4B, that low entanglement makes suppression beneficial, or that
   P4D supplied a method-level bridge.

## 8. SCST V0 -> magnitude Repair-1 -> source-support Repair-2

**Conflict class:** implementation repairs separated from prospectively bounded
scientific repairs.

1. **Earlier result.** V0 used the predeclared empirical subject-class residual
   at alpha 1 and closed as `TRANSPORT_NOT_SUBJECT_FAITHFUL`: residual stability
   passed, but WBCIC target-affinity improvement was negative and centroid-
   manifold ratios were `1.76-2.68`; several pre-embedding class gates also
   failed. Two earlier execution failures (a read-only Pandas mask and slow
   one-row sklearn dispatch) occurred before a completed metric unit and were
   implementation-only repairs. Repair-1 then evaluated the prelocked global
   alphas `{0.25, 0.5}` and closed as `TRANSPORT_OFF_MANIFOLD`, with no globally
   eligible alpha. Sources:
   `experiments/persist_eeg_final_scst_dr/ITERATION_LEDGER.md`,
   `experiments/persist_eeg_final_scst_dr/TRANSPORT_VALIDITY_REPORT.md`,
   `experiments/persist_eeg_final_scst_dr/STAGE0_REPAIR1_REPORT.md`, and the
   corresponding validators under `experiments/persist_eeg_final_scst_dr/results/`.
2. **Problem.** V0's completed scientific failure diagnosed magnitude overshoot,
   not an arbitrary-direction implementation error. Repair-1 showed that a
   smaller global step could recover subject/class fidelity but still violated
   the frozen WBCIC manifold gate. Neither result could be counted as a final
   test of a locally support-constrained operator, while the pre-metric execution
   failures cannot be counted as scientific failures at all.
3. **Repaired result.** Repair-2 kept the residual direction, used only
   `final_embedding`, capped alpha at `0.25`, and selected the largest value on a
   fixed `1/64` grid admitted by Session-1-only same-class source support; Session
   2 remained the independent validity partition. OpenBMI EEGNet and
   EEGConformer passed all gates. WBCIC retained positive subject affinity,
   matched-random advantage, class fidelity, and the binary off-manifold gate,
   but the independent-session 3NN ratios were `1.3079565` and `1.3407996`, above
   the frozen `<=1.25` threshold. All 20 units validated; only 2/4 settings
   passed. Start/freeze commit:
   `0312681a960f066e8a760edf839521d62d61f60e`; final commit:
   `ad732ba9be5c1886acc4f5cb264d7528934a5db4`. Sources:
   `experiments/persist_eeg_final_scst_dr/STAGE0_REPAIR2_REPORT.md`,
   `experiments/persist_eeg_final_scst_dr/results/STAGE0_REPAIR2_VALIDATION.json`,
   and `experiments/persist_eeg_final_scst_dr/results/FINAL_CLOSURE_VALIDATION.json`.
4. **Final authoritative interpretation.** Repair-2 terminal:
   `TRANSPORT_VALIDITY_NOT_SUPPORTED`; final terminal:
   `FINAL_CONSTRUCTIVE_HYPOTHESIS_NOT_SUPPORTED`. The result is a validated
   cross-dataset transport-validity failure, not a runtime failure. Stage 1,
   Repair-3, future-performance evaluation, and outer evaluation were correctly
   not run.
5. **Manuscript consequence.** Use Repair-2 as the sole final transport-validity
   result. It supports a precise boundary: source-faithful, class-compatible,
   matched-random-superior transport was still insufficient for independent-
   session local support on WBCIC. Do not write that SCST training failed; it was
   never authorized or run.

## 9. SCAA Pearson/subgroup signals -> primary Spearman and reliability closure

**Conflict class:** exploratory/subgroup signal versus preregistered primary and
prospective reliability gates.

1. **Earlier result.** Several secondary analyses looked favorable. Pooled
   Pearson S2-to-S3 utility correlation was `0.729823`, CI
   `[0.044905, 0.903391]`. In the EEGConformer subgroup, certified future harm
   was `0.1667` versus always-adapt harm `0.3902`, an absolute reduction
   `0.2236`, CI `[0.0668, 0.3920]`, with coverage `0.4390`; sign concordance was
   `27/41=0.6585` (`p=0.0596`). Sources:
   `experiments/persist_eeg_scaa_stage0/results/STATISTICAL_TESTS.json` and
   `experiments/persist_eeg_scaa_stage0/results/UTILITY_TRANSFER_CORRELATION.csv`.
2. **Problem.** The frozen primary Gate A used pooled subject-level Spearman with
   a CI lower bound above zero and positive estimates in both backbones, not
   Pearson. Gate C used pooled future-harm reduction, not selection of the more
   favorable backbone. Backbones were correlated measurements of the same 41
   subjects, not independent replications. Secondary Pearson sensitivity and a
   post hoc favorable subgroup cannot override the frozen pooled gates.
3. **Repaired result.** The final Stage-0 report retained every signal but applied
   the primary rules. Spearman estimates were EEGNet `0.1862`, CI
   `[-0.1722, 0.5058]`; EEGConformer `0.3107`, CI
   `[-0.0163, 0.5914]`; pooled `0.3150`, CI
   `[-0.0183, 0.5986]`. Pooled sign concordance was `22/41=0.5366`, exact
   `p=0.7552`; pooled certified-harm reduction was only `10.3%`, with CI crossing
   zero; the S2-gated policy did not beat always adapt (`-0.000183` BA, CI
   `[-0.001829, 0.001382]`). Gates A-C failed, authorization was denied, and the
   terminal was `TARGET_HISTORY_UTILITY_TRANSFER_PARTIAL`. Freeze/start commit:
   `e9de7ebc6eaf6f96aa5717bff10a6dca51b27a95`; validated final/figure commit:
   `46b8ecf2c39b0e32045cad9d78ca12327f0a3f0d`. Source:
   `experiments/persist_eeg_scaa_stage0/SCAA_STAGE0_FINAL_REPORT.json`.
4. **Final authoritative interpretation.** Stage-0 supports architecture-
   dependent, favorable but insufficient utility transfer. Stage-0.5 did not
   provide a certificate-of-certificate: best decision-stability OOF AUROC was
   `0.580918`, CI `[0.478464, 0.681375]`; reliability-gated harm was `0.3333`
   versus `0.3077` for the simple S2 gate (relative harm reduction `-0.0833`),
   and no legal target-level Identity control existed. Terminal:
   `RELIABILITY_MECHANISM_PARTIAL`; authorization
   `RELIABILITY_GATED_SCAA_DEVELOPMENT_NOT_AUTHORIZED`. Freeze commit:
   `6398186d1717a84e8d1120c49f7274a4c47d5ed5`; final commit:
   `d0a9b1a006a488da929f1c1b914e7fd903d925fb`. Source:
   `experiments/persist_eeg_scaa_reliability_stage05/RELIABILITY_STAGE05_FINAL_REPORT.json`.
5. **Manuscript consequence.** Report the frozen Spearman, sign, pooled harm, and
   Stage-0.5 reliability results as primary. Pearson and the EEGConformer subgroup
   may appear only as clearly labeled sensitivity/boundary evidence. Do not claim
   that historical target utility universally certifies future utility or that a
   deployable SCAA gate was established.

## 10. Experiment-local OpenBMI internal-14 purity -> project-global invalidity

**Conflict class:** experiment-local purity versus project-global confirmation
status.

1. **Earlier result.** SCST and several later P4 workflows correctly recorded
   that their local 14-subject OpenBMI internal cohort was unmaterialized,
   untouched, or unenumerated. For example,
   `experiments/persist_eeg_final_scst_dr/protocol/DATA_ACCESS_AUDIT.json`
   records 40 development subjects, 14 internal subjects, and
   `sealed_internal_holdout_membership_materialized=false`.
2. **Problem.** A local no-access statement cannot establish that the same people
   were globally unseen. There are two independent disqualifiers. First, the
   server-only Stage-0 and P2 analyses used the full OpenBMI offline/train cohort
   of subjects 1-54 before the later 40/14 partition; see
   `paper_closure/evidence_sources/server_only_early/delivery/persist_eeg_stage0/PROTOCOL.md`,
   `paper_closure/evidence_sources/server_only_early/delivery/persist_eeg_stage0/OPENBMI_STAGE0_SEED0_CORE_SUMMARY.json`,
   and `paper_closure/evidence_sources/server_only_early/outputs/persist_eeg_p2p3/P2_FINAL_REPORT.json`.
   The 14 cannot become globally untouched retroactively. Second, V8 used the
   same partition digest (`6b771f...`) and produced a non-empty internal-holdout
   result artifact. V8 lock/result commit:
   `b147226d2d82bb47948d1cca6c7b403c590eb8fa`. Sources:
   `experiments/persist_eeg_final_model_v8/outputs/protocol/V8_INTERNAL_HOLDOUT_LOCK.json`
   and the existence only of
   `experiments/persist_eeg_final_model_v8/outputs/final_candidate/INTERNAL_HOLDOUT_RESULTS.json`.
   The latter's outcome content was not opened.
3. **Repaired result.** The closure-level audit separates local and global status.
   `paper_closure/DATA_AND_HOLDOUT_AUDIT.md` and
   `paper_closure/protocol/FINAL_PAPER_DATA_STATUS.json` assign
   `OPENBMI_INTERNAL_14_PARTITION_6B771F` the project-global status
   `INVALID_FOR_CONFIRMATION` while preserving the truth that individual later
   workflows did not access it.
4. **Final authoritative interpretation.** The internal 14 are locally untouched
   in SCST/P4 where so recorded, but globally already observed through the early
   full-54 analyses and independently disqualified by the V8 artifact. Either
   reason alone is sufficient. They are not a sealed confirmation resource for
   this paper.
5. **Manuscript consequence.** Do not claim a fresh OpenBMI internal-14
   confirmation, do not list the cohort as globally sealed, and do not open the
   V8 outcome. Local purity statements may be used only to characterize the
   corresponding experiment's leakage controls.

## 11. WBCIC persistence harm replication -> no replication of matched-random specificity

**Conflict class:** component replication versus failed specificity replication.

1. **Earlier result.** OpenBMI P2/V3.1 showed substantial persistent/protected
   erasure harm relative to matched random controls. The frozen WBCIC replication
   then found positive harm for the predeclared `P01_04` block: BA harm
   `0.0144245`, CI `[0.0055187, 0.0266370]`. This is a valid replication of a
   task-supportive persistent-structure consequence for that block.
2. **Problem.** The stronger independent-replication claim also required excess
   harm over matched random. For `P01_04`, matched-random harm was `0.0084276`
   and persistent-minus-random was only `0.0059968`, CI
   `[-0.0026776, 0.0167406]`. The second predeclared block `P05_08` had harm
   `0.0002066`, CI `[-0.0039431, 0.0042083]`, and persistent-minus-random
   `-0.0010486`, CI `[-0.0043631, 0.0021297]`. Neither block passed specificity.
3. **Repaired result.** The validated final report records
   `R1_PARTIAL_SUPPORT`, identifies `P01_04` as the only reliable task-supportive
   block, and lists `strong_blocks=[]`. Protocol commit:
   `9f76a4437c0635febea190cc51ea89500703a5c6`; final commit:
   `1ff8edda656372d8d36a2bcdb7d96311f88f8da6`. Sources:
   `experiments/persist_eeg_wbcic_independent_replication_v1/FINAL_REPORT.md`,
   `experiments/persist_eeg_wbcic_independent_replication_v1/results/summary.json`,
   and `experiments/persist_eeg_wbcic_independent_replication_v1/results/FINAL_VALIDATION.json`.
4. **Final authoritative interpretation.** WBCIC replicates persistence-erasure
   harm for one predeclared block, but does not replicate persistence-specific
   excess over matched random. Separately, its D-versus-I consequence result is
   strong: MI RMSE `0.0165056`, MD RMSE `0.0118692`, difference `0.00463634`, CI
   `[0.00123702, 0.00719447]`, with 10/15 runs favoring D. That separate result
   cannot rescue R1 specificity.
5. **Manuscript consequence.** Write: “persistence-erasure harm replicated for
   one predeclared WBCIC block, whereas excess harm over matched random did not.”
   Do not write that persistence specificity fully replicated independently.
   Keep the WBCIC D-versus-I replication as a separate finding.

## 12. Historical v7 FM routing != PERSIST EEG-FM generality audit

**Conflict class:** task-name/model-family overlap mistaken for scientific
equivalence.

1. **Earlier result.** The surviving historical FM ref
   `refs/remotes/github/v7-0c-admissible-pool-full-pipeline` (tip
   `f744778dc9cdfff00cb94c4754bdd0c25659396a`) contains three stopped routes:
   v7 at `86b6ea46c8b5a3497e68472d75f490520503637d` with
   `V7_STAGE0A_STOP_MODEL_QUALIFICATION_FAILURE`; v7R at
   `4cb4ffa8973cbb98de29c6fcfe10cb667da61d42` with
   `V7R_STOP_ADAPTER_FIDELITY_FAILURE`; and v7full at
   `f744778dc9cdfff00cb94c4754bdd0c25659396a` with
   `V7_STOP_NO_ADMISSIBLE_EXPERT_POOL`. Primary paths are
   `delivery/fm_routing_v7/V7_STAGE0A_DECISION.json`,
   `delivery/fm_routing_v7_repair/V7R_DECISION.json`, and
   `delivery/fm_routing_v7_full/FINAL_DECISION.json` on that ref.
2. **Problem.** Those routes used HMC/EEGMMIDB and CBraMod/LaBraM/BIOT routing or
   compatibility tests. They did not run OpenBMI or WBCIC, did not produce a
   valid EEGPT result, did not test the PERSIST FM-Q1-Q6 ladder, did not measure
   the I/P/D/C/G/A chain, and did not perform the required controlled fine-tuning.
   Model-family overlap and the string “FM” do not make them an answer to the
   PERSIST generality question. Their stop terminals are qualification/adapter/
   pool failures, not a scientific no-reversal result.
3. **Repaired result.** The closure reconstructed the full surviving branch
   history and task semantics in
   `paper_closure/evidence_sources/FM_BRANCH_HISTORY_AUDIT.md`, without reading
   any protected/sealed outcome. It maps every historical terminal to its actual
   dataset, model, intervention, and authorization scope.
4. **Final authoritative interpretation.** Historical v7 evidence does not answer
   whether the PERSIST diagnostic-actionability gap reverses or persists in
   pretrained EEG foundation representations. For the current paper matrix, the
   EEG-FM cell is `NOT TESTED`, not “no reversal” and not “reversal.” The closure
   decision is `EEG_FM_AUDIT_NEEDED = YES`.
5. **Manuscript consequence.** Do not cite v7 as EEG-FM generality evidence and
   do not infer a no-reversal conclusion from model-qualification failures. Any
   eventual FM claim must come from a separately frozen PERSIST FM-Q1-Q6 audit
   and must report non-qualification, reversal, or no reversal regardless of
   sign.

## Final source-priority table

| Priority | Source class | Use in closure | Examples | Rule |
| ---: | --- | --- | --- | --- |
| 1 | Closure-level global audits | Resolve cross-experiment scope and data status | `paper_closure/DATA_AND_HOLDOUT_AUDIT.md`; `paper_closure/protocol/FINAL_PAPER_DATA_STATUS.json`; `paper_closure/evidence_sources/FM_BRANCH_HISTORY_AUDIT.md` | Authoritative for project-global status only; they do not replace experiment-level effect estimates |
| 2 | Latest prospectively frozen, validated experiment result | Main quantitative and terminal evidence | P2 final; Signed V3.1; Shared Geometry V1.2; DDA final; P4B/P4C/P4D finals; SCST Repair-2; SCAA/Stage-0.5; WBCIC final | Use the frozen primary metric, gate, and validator; preserve negative terminals |
| 3 | Scientifically valid repaired version | Replaces an invalid implementation/provenance version | Signed V3.1, Shared Geometry V1.2, Matched Identity V1.2, SCST Repair-2 | A repair is authoritative only for the estimand it validly implements; it cannot retroactively change a different earlier estimand |
| 4 | Valid scoped secondary/subgroup analysis | Boundary or sensitivity evidence | DDA-B/C, SCAA Pearson and EEGConformer subgroup, P4C coarse safety boundary | May refine interpretation but never override a failed preregistered primary gate |
| 5 | Server-only compact evidence with checksum provenance | Historical/main evidence when no Git commit exists | Stage-0/P2/P3 package under `paper_closure/evidence_sources/server_only_early/` | Cite repo-relative path plus `SOURCE_MANIFEST.csv`; label `NO_GIT_COMMIT_SERVER_ONLY_EVIDENCE` |
| 6 | Preliminary, incomplete, or superseded result | History only | Stage-0 seed 0, Shared Geometry V1/V1.1, Matched Identity V1/V1.1, SCST V0/Repair-1 | Do not use its stronger conclusion or obsolete point estimate as the manuscript result |
| 7 | Invalid or outcome-inaccessible artifact | Exclusion/provenance fact only | unreplayable Signed V3 assignments; V8 internal-holdout result artifact | Do not infer scientific outcome; artifact existence may establish invalidity or provenance only |

## Closure verdict

The resolved evidence supports a calibrated paper chain: persistent/protected
structure can be task consequential; Decision Dependence can predict
intervention consequence better than measured Identity; neither fact by itself
establishes beneficial future suppression or prospective actionability. The
audit does **not** support a universal persistence-specific effect across
datasets, a matched-identity causal effect, a successful constructive method, a
fresh OpenBMI internal confirmation, or an EEG-foundation-model generality
claim at the current evidence state.
