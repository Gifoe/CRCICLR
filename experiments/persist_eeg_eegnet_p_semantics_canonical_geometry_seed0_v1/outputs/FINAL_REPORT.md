# EEGNet P semantics and canonical geometry, seed 0

Scope: canonical EEGNet, OpenBMI_MI, five folds. The checkpoint is frozen; each fold's final P and earlier PathFit pathways were refitted from inner-train subjects only. OUTER_DEVELOPMENT is a subject-disjoint evaluation population, not a fitting or selection pool. Numbers below are equal-weight means of the five folds unless otherwise stated. The detailed fold estimates, 20,000-draw subject bootstrap intervals, transformation arms, and control results are in the companion CSVs.

## Q1. What semantic information dominates P?

At the 64-D embedding, P's subject/session baseline persistence cosine is **0.622** (C 0.570; equal-rank random 0.540; full 0.568). P's within-subject, cross-session class-relation cosine is **0.856** (C 0.309; random 0.607; full 0.659), and its held-subject relation-to-TRAIN-reference cosine is **0.922** (C 0.432; random 0.748; full 0.785). P exceeds C and rank-matched random on the last two quantities in all five folds.

The baseline and class relation behave differently under controls. Permuting session-2 subject identities leaves the P task relation essentially unchanged (observed 0.856 versus null mean 0.857; per-fold permutation p 0.109–0.746), consistent with a common task direction. The same permutation disrupts the P baseline (observed 0.622 versus null mean 0.130; per-fold p 0.005–0.020), so P also carries stable subject context. Permuting within-subject/session trial labels collapses held-subject task-relation agreement (observed 0.922 versus null mean −0.006; p = 1/201 in each fold). These are descriptive permutation tests, not causal interventions.

The TRAIN subject-ID probe has BA **0.154** for P versus 0.304 C, 0.121 random, and chance 1/26 = 0.038. In OUTER_DEVELOPMENT centroid variance, the embedding P class main effect is 0.544 and subject main effect 0.213; C reverses that balance (class 0.089, subject 0.545). P's held-subject class probe BA is 0.710, versus C 0.617 and random 0.642. Thus task structure dominates final P, but subject context is not absent. Earlier PathFit P is more subject-heavy (subject fraction 0.431 temporal, 0.404 spatial, 0.357 shared); it is a fitted pathway to final P, not independent biological persistence.

## Q2. Is P more task-transferable than C or random?

Yes operationally, for this frozen geometry and checkpoint. The embedding P decoder trained on TRAIN session 1 attains OUTER session-2 BA **0.701**, versus C 0.612, random 0.637, and full 0.729; the reverse session-2→session-1 BA is 0.718, versus C 0.623, random 0.646, and full 0.732. A decoder trained on both TRAIN sessions transfers to held subjects at BA **0.716**, versus C 0.611, random 0.644, and full 0.730. P exceeds C and random in each of the five folds on both forward-session and both-session-to-held-subject transfer tests. The native frozen EEGNet head has BA 0.735 on full embedding; P-retained construction has 0.728, C-retained 0.604, and random-P-retained 0.617. Head-retention comparisons are descriptive and are not additive causal attribution.

This is not proof of a *Protected-specific* property independent of energy. On OUTER_DEVELOPMENT the mean squared centroid norm is 4.978 for P versus 0.686 for rank-matched random (about 7.3-fold), and centroid variance is 4.826 versus 0.654. The random control matches rank but not energy/covariance. In addition, the P-selection rule itself uses TRAIN-label-informed erasure utility, so class content is partly expected by construction. The held-subject results demonstrate transfer of this defined P, not discovery of an entirely label-free invariant.

## Q3. Is P shared task structure, subject context, or mixed?

The supported labels are `P_SHARED_TASK_CORE_SUPPORTED` and `P_MIXED_TASK_AND_SUBJECT_STRUCTURE`. A transferable class relation is the dominant final-embedding signal, while biological-subject-matched baselines and nonchance identity remain inside P. Calling P purely subject-invariant would discard measured context; calling it primarily subject-specific would contradict the five-fold held-subject relation and frozen-decoder transfer. C is not classified as nuisance: it carries substantial subject structure and some task information.

## Q4. Are sessions mainly different coordinate systems?

No tested low-complexity, TRAIN-fitted *unsupervised* transform meets the locked materiality rule: at least +0.03 held-subject frozen-decoder BA and at least 10% geometric-residual reduction in four or more folds. The simplest useful family is **NONE**. At embedding P, equal-fold global BA recovery is +0.001 translation, +0.004 diagonal, −0.077 orthogonal, and −0.010 at rank-4 low-rank residual. The best mean zero-shot gain among these is only +0.004 BA. The TRAIN-label-assisted oracle is diagnostic, not deployable; even its best embedding mean is +0.017 BA. The target-unlabeled per-subject diagonal diagnostic reaches +0.022 BA at embedding and remains below the +0.03 rule; it is not zero-shot. Part C evaluates on a label-blind held-trial inventory to protect the unlabeled arm, so its absolute baseline BA is not directly paired with Parts A/B; its transformed-minus-untransformed recovery is paired within Part C. These results reject the tested simple-coordinate-drift explanation, not every possible nonlinear or high-capacity mapping.

## Q5. Where is task structure easiest to transfer?

The **64-D embedding** has the strongest P task relation and frozen held-subject transfer: cross-subject relation cosine 0.922 and cross-subject BA 0.716, compared with temporal 0.207/0.547, spatial 0.750/0.681, and shared 0.776/0.706. The shared layer has a smaller session-decoder boundary angle (12.6° versus embedding 20.0°), but its task relation and held-subject BA are lower. The layerwise CSV retains these dimensions separately rather than collapsing them into one score. Previous erasure evidence used a different P geometry and is not numerically merged into this refitted-geometry summary.

## Q6. Does the decision boundary survive session/subject change?

At embedding the P session-specific linear-decoder normals differ by **20.0°**, versus C 60.1°, random 24.4°, and full 38.1°. P's normal angle is not uniformly smaller than random across folds, so the stronger evidence is the paired frozen-decoder transfer above, not angle alone. A low angle at the temporal stage (P 8.4°) does not imply a useful core there: temporal P held-subject BA is only 0.547.

## Q7. What should the next model modify?

Exactly one primary direction: **`FACTORIZED_TASK_CONTEXT_MODEL`**. It should explicitly represent a shared class-relation component alongside a stable context component inferred from EEG or calibration statistics. The decisive measurements are high held-subject P class-relation agreement and frozen-decoder transfer *together with* subject-matched baseline persistence and nonchance subject-ID information. Generic P amplification, another P loss, and an amortized canonicalizer are not justified by this audit. This branch does not train the proposed model or estimate its prospective performance.

## Integrity and limits

All five fresh geometries, 80 fold×stage×part audits, and 285 compact source CSVs were independently checked against declared hashes and row counts. Checkpoint and model/BatchNorm state hashes match frozen provenance. Fit populations are TRAIN_GEOMETRY only; CHECKPOINT_VALIDATION is identified as checkpoint-selection-exposed, never untouched. The existing prior mechanism-closure provenance records historical outer-development access, disclosed here; it did not enter current geometry/probe/transform fitting. The V1/V2 transform attempts were halted before scientific output when their supposedly unsupervised rank selection was found to score labeled class relations. The V3 rank criterion uses unlabeled TRAIN subject means and Gram covariance only. Formal final-heldout EEG reads: **0**.
