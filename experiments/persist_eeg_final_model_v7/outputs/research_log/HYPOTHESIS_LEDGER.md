# V7 hypothesis ledger

All verdicts are adaptive development verdicts from seed `20260820`. They are
not independent confirmations. `OUTER_TEST_USED=false` for every entry.

| ID | Falsifiable hypothesis | Main evidence | Verdict |
|---|---|---|---|
| H0 | A selector over existing V1-V6 experts can supply +5 pp headroom. | V6 expert subject-oracle headroom was +2.482 pp OpenBMI and +2.099 pp WBCIC. | Rejected before new model search. |
| H1 | Future-session component utility is predictable from generic legal history. | Mean generic R2 was -0.0265 OpenBMI and 0.0576 WBCIC; outcome Pearson was -0.1797 and 0.4919. | Mixed: weak on WBCIC, failed on OpenBMI outcome transfer. |
| H2 | P/U/D/G/R add utility-prediction information beyond matched generic context. | Mean R2 changed by +0.0545 OpenBMI and +0.0541 WBCIC. PERSIST R2 was higher in 13/15 and 14/15 matched fold/controller configurations. | Supported only as an adaptive mechanistic signal. Configurations are correlated and one-seed. |
| H3 | PERSIST-Meta improves future BA or safety over matched META-GENERIC. | BA delta +0.093 pp `[-0.130,+0.333]` OpenBMI; +0.233 pp `[-0.316,+0.843]` WBCIC. Harmful fractions and worst deltas did not improve consistently. | Not established. |
| H4 | Coarse Meta-SGD/projected-gradient components provide sufficient action headroom. | Full Meta-SGD had small mean CE utility; projected groups were negative on average. The selected controller remained below the anchor alone. | Rejected as a route to the target. |
| H5 | Calibration, ridge/LDA, and prototype actions yield a reliable prospective controller. | Some components had positive mean CE utility, but CE and BA utility often disagreed and selection gains were small/unresolved. | Rejected as a performance solution; retained as utility probes. |
| H6 | History Euclidean alignment removes harmful session shift. | Standalone EA EEGNet reached 74.30% OpenBMI and 75.47% WBCIC versus identity controls 79.06% and 78.92%. | Rejected; alignment removed useful structure or introduced mismatch. |
| H7 | A filter-bank log-variance backbone gives robust cross-session gains. | Best corrected standalone FBC result was about 76.56% OpenBMI; anchor blends did not beat the generic Conformer blend. | Rejected. |
| H8 | Compact EEG-Conformer adds useful generic diversity. | Fixed-head anchor blend reached 83.778% OpenBMI and 82.497% WBCIC, +0.574 and +0.415 pp over V6 anchors, with CIs crossing zero. | Modest point-estimate support, not statistically resolved and far below target. |
| H9 | Class-conditional session alignment improves the Conformer. | WBCIC fixed-head standalone BA was 76.86%, below the unaligned Conformer; some variants collapsed further. | Rejected; forced alignment is harmful here. |
| H10 | A low-rank history hypernetwork converts P/U/D/G/R into adaptation gain. | PERSIST hypernetwork BA was 78.26% OpenBMI and 79.08% WBCIC; anchor residuals also remained below anchors. | Rejected; training-episode fit did not generalize. |
| H11 | A more elaborate router over the new experts can still reach +5 pp. | Outcome-only subject oracle reached 85.259% OpenBMI and 83.461% WBCIC: only +2.056 and +1.379 pp headroom. | Rejected within the evaluated expert set. |
| H12 | Real EEG evidence requires SUPPRESS. | No independently certified harmful subspace was found; PRESERVE can reject negative predicted utility. | Not supported; suppression remained disabled. |
| H13 | The selective mechanism can recover a known constructed decomposition. | Synthetic sensitivity/specificity/harmful rejection were all 1.0; synthetic utility R2 was 0.955. | Wiring check passed; no real-EEG inference allowed. |
| H14 | V7 reaches +5 pp over the strongest matched baseline on both benchmarks. | Best gains were +0.574 pp and +0.415 pp, both unresolved. | Rejected. |

## Stop rationale

The negative families cover action representation, transport geometry,
backbone, alignment objective, conditional hypernetwork, risk policy, and
mixture headroom. A tenth micro-variant chosen after inspecting the same
development outcomes would mostly increase adaptivity. The next credible test
is a pre-registered architecture/controller evaluated on fresh confirmation
subjects; the WBCIC outer cohort must remain sealed until a candidate is frozen
independently of it.
