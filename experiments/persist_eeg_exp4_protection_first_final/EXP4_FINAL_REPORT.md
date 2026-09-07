# PERSIST-EEG Experiment 4 — Final report

## Terminal decision

`EXP4_PROTECTION_FAILED`

The primary development comparison was run on 41 WBCIC development subjects across five frozen subject-disjoint folds. The 10 sealed outer subjects were never opened. No outer command was authorized because the development gate failed.

## 1. Deployment and model definition

The selected deployment setting was sequential S1→S2→S3: train an EEGNet anchor on S1, construct the fold-specific persistence basis from legal S1/S2 discovery subjects, fit a global adapter on S2 model-fit subjects, and evaluate unseen outcome subjects on S3. This preserves the requested past-session-to-future-session question and avoids using an outcome subject's S3 information.

The anchor is the 58-channel, 1000-sample EEGNet with a 32-dimensional embedding, dropout 0.25, S1-only training for 30 epochs, AdamW learning rate `3e-4`, weight decay `5e-4`, and batch size 64. The adapter is a zero-initialized 32×32 linear residual `A(h)=hW+b`, fitted on S2 with the frozen anchor classifier. The generic configuration selected before the Guard comparison is `lr=1e-3`, `25` epochs, and weight decay `5e-4`; all controls use matched capacity and budget.

`U_P` is the rank-four `P01_04` block of the S1/S2 cross-session subject-centroid persistence basis. Its task-protected status is inherited from the frozen actionability audit. No held-out S3 information is used. V1 uses:

`h_guard = h_anchor + (I-U_P U_P^T)A(h_anchor)`.

The complement is not called nuisance. Identity is a control, not the selection criterion.

## 2. Generic baseline and primary result

The three generic candidates had S2 validation means 0.7868, 0.7902, and 0.7890; `GEN_LINEAR_LR1E3_E25` was selected by generic validation quality alone.

| method | S3 subject BA | macro-F1 | negative transfer vs Frozen | worst-quartile ΔBA vs Frozen |
|---|---:|---:|---:|---:|
| Frozen | 0.7012 | 0.6694 | 0.0% | 0.0000 |
| Generic | 0.7719 | 0.7701 | 14.6% (6/41) | -0.0041 |
| V1 PERSISTGuard | 0.7732 | 0.7715 | 17.1% (7/41) | -0.0050 |

The paired V1 Guard−Generic BA delta is `0.001340` (`+0.134 pp`), median `0`, 18/41 subjects favoring Guard, subject-bootstrap 95% CI `[-0.000855, 0.003535]`, deterministic sign-flip Monte-Carlo `p=0.13663`, and exact positive-sign binomial `p=0.82556`. This is not a practically established improvement. Generic protected-coordinate drift is `0.3512`; V1 Guard drift is `3.7e-8`. Generic decision-response drift is `1.4767`; V1 is `1.3489`. The Generic drift/ΔBA correlation is `0.3483`, suggestive but not a validated causal endpoint.

## 3. Controls and mechanism

Relative to Generic, the mean PERSISTGuard BA differences for V1 were:

* RandomGuard `+0.146 pp`, CI `[-0.037,+0.345] pp`;
* PCAGuard `+0.061 pp`, CI `[-0.122,+0.244] pp`;
* Persistence-only Guard `+0.183 pp`, CI `[-0.037,+0.403] pp`;
* IdentityGuard `+0.134 pp`, CI `[-0.024,+0.293] pp`.

Every interval crosses zero. Thus the Protected criterion is not specific. V1 clearly protects coordinates, but it does not fully preserve the decision response identified in Experiment 3. V2 added an exact minimum-norm frozen-head response correction and reduced response drift to `4.5e-9`, but its Guard−Generic delta was `-0.006596` (−0.660 pp; CI `[-0.011707,-0.001728]`). V3 used a fixed `alpha=0.25` correction and obtained `+0.085 pp` (CI `[-0.147,+0.305] pp`, `p=0.26705`) with response drift `1.0680`; negative transfer remained 17.1%. These variants do not support a stable protection mechanism-performance tradeoff.

## 4. Replication, outer access, and claims

No DeepConvNet, TeCh, or EEGConformer replication was run because EEGNet failed the primary gate. FBCNet is listed as excluded because the prior competence audit found it near chance, not because it was selectively dropped after this result.

Outer subjects were not enumerated, materialized, or evaluated. The final protocol file records `outer_evaluation_authorized=false`; there is no outer result and no post-outer tuning.

The justified claim is narrow: a strong generic representation update existed, but the tested protection-first variants did not establish safer or more generalizable future-session EEG performance. The stronger claim that protecting task-protected persistence improves generalization is not justified. The broader claim that identity is the wrong criterion remains supported by Experiment 3, not by a positive Exp4 result.

The remaining ICLR weakness is external validity and alignment of the historical P01_04 assignment with the S1-only anchor. A new prospective basis/utility protocol could address that; it cannot be repaired by opening or retuning on the sealed outer subjects.
