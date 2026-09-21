# Protected P/C arbitration reliability audit

Cells: 20/20 COMPLETE; 0 fail-closed. Final-heldout access: NO. Frozen backbone, native head, bases, Protected dimensions, splits and locked random subsets were reused. Statistical unit: biological subject; repeated fold measurements are averaged per subject before bootstrapping. Oracle rows are label-aware diagnostics only and are never presented as deployable-policy results.


## EEGNet / OpenBMI_MI

Completed folds: 5/5.

1. **P-specific oracle headroom.** Mean P suppression-oracle BA gain was 0.07775; P minus the equal-rank-random mean was -0.0242, mean P percentile was 0.106, and mean empirical p was 0.89901. Therefore Protected-specific headroom is not supported; this report does not attribute generic reweighting headroom to P.

2. **How native errors are reweight-recoverable.** Mean subject-fold rescue counts were C_DOWN_ONLY=1.68, P_UP_ONLY=2.55, P_DOWN_ONLY=1.93, C_UP_ONLY=2.35, JOINT_REWEIGHT=2.88, NOT_REWEIGHT_RECOVERABLE=14.9; the largest mean category was NOT_REWEIGHT_RECOVERABLE. Mean minimum log-scale intervention distance was 0.437678. These are response-surface diagnostics, not policy performance.

3. **P-only and C-only reliability.** FEATURE-D outer metrics (AUROC/AUPRC/Brier) were rP=0.74747/0.855678/0.180809 and rC=0.788294/0.846303/0.180206. These are outer-development evaluations only.

4. **P/C disagreement arbiter.** Disagreement fraction was 0.49075; the direct arbiter FEATURE-D outer AUROC/AUPRC/Brier/accuracy was 0.703991/0.75038/0.218016/0.650626.

5. **Pathway-consistency contribution.** For rP, FEATURE-A to C to D AUROC was 0.751586 → 0.748061 → 0.74747; for rC it was 0.791337 → 0.788354 → 0.788294. This ablation is descriptive across the fixed outer folds; no untested causal claim is made.

6. **Where suppression oracle headroom occurs.** On disagreement trials, mean oracle-recoverable fraction was 0.881592; the rescue taxonomy above identifies whether attenuation, P scaling, joint scaling, or no fixed-grid reweighting was involved.

7. **Can TRAIN-only gain models distill suppression benefit?** FEATURE-D gain-alpha BA delta versus native was 0.00125 (95% CI [-0.00175, 0.00425]; positive folds 3/5).

8. **Does the learned alpha policy improve outer subjects?** The result in item 7 is the biological-subject bootstrap estimate; an interval spanning zero is not evidence of a reliable improvement.

9. **Does 2D arbitration clearly exceed suppression-only?** FEATURE-D 2D gain-policy BA delta was 0.00075 (95% CI [-0.0025, 0.00425]; positive folds 3/5), versus suppression-only 0.00125 (95% CI [-0.00175, 0.00425]; positive folds 3/5). This is a comparison of fixed TRAIN-only policies, not an oracle comparison.

10. **Why learned policies may fail.** FEATURE-D reliability-routing BA delta was 0 (95% CI [0, 0]; positive folds 0/5). Reliability, disagreement, pathway ablation, random-policy controls and gain-distillation results are all retained in their compact tables; a near-zero or uncertain policy delta must not be attributed to one mechanism without those diagnostics agreeing.


### Policy bootstrap summary

- C_only: -0.10675 (95% CI [-0.125, -0.08875]; positive folds 0/5)
- P_only: -0.041 (95% CI [-0.058, -0.0245]; positive folds 0/5)
- fixed_alpha_0.25: -0.02075 (95% CI [-0.0357563, -0.00549375]; positive folds 0/5)
- fixed_alpha_0.5: -0.00575 (95% CI [-0.018, 0.0055]; positive folds 2/5)
- fixed_alpha_0.75: 0.00225 (95% CI [-0.0065, 0.01075]; positive folds 3/5)
- fixed_alpha_0: -0.041 (95% CI [-0.057, -0.0244938]; positive folds 0/5)
- fixed_alpha_1: 0 (95% CI [0, 0]; positive folds 0/5)
- gain_2d_A: -2.77556e-18 (95% CI [-0.003, 0.00275]; positive folds 3/5)
- gain_2d_C: -0.00125 (95% CI [-0.0045, 0.00175]; positive folds 1/5)
- gain_2d_D: 0.00075 (95% CI [-0.0025, 0.00425]; positive folds 3/5)
- gain_alpha_A: 0.00125 (95% CI [-0.001, 0.00350625]; positive folds 4/5)
- gain_alpha_C: 0.001 (95% CI [-0.0025, 0.00425]; positive folds 3/5)
- gain_alpha_D: 0.00125 (95% CI [-0.00175, 0.00425]; positive folds 3/5)
- oracle_2d: 0.11375 (95% CI [0.1015, 0.126]; positive folds 5/5)
- oracle_suppression: 0.07775 (95% CI [0.0685, 0.0867562]; positive folds 5/5)
- reliability_A: 0 (95% CI [0, 0]; positive folds 0/5)
- reliability_B: 0 (95% CI [0, 0]; positive folds 0/5)
- reliability_C: -0.00025 (95% CI [-0.00075, 0]; positive folds 0/5)
- reliability_D: 0 (95% CI [0, 0]; positive folds 0/5)

## EEGNet / OpenBMI_SSVEP

Completed folds: 5/5.

1. **P-specific oracle headroom.** Mean P suppression-oracle BA gain was 0.02125; P minus the equal-rank-random mean was -0.0018125, mean P percentile was 0.418, and mean empirical p was 0.617822. Therefore Protected-specific headroom is not supported; this report does not attribute generic reweighting headroom to P.

2. **How native errors are reweight-recoverable.** Mean subject-fold rescue counts were C_DOWN_ONLY=0.475, P_UP_ONLY=0.925, P_DOWN_ONLY=0.55, C_UP_ONLY=0.875, JOINT_REWEIGHT=0.7, NOT_REWEIGHT_RECOVERABLE=4.42; the largest mean category was NOT_REWEIGHT_RECOVERABLE. Mean minimum log-scale intervention distance was 0.411041. These are response-surface diagnostics, not policy performance.

3. **P-only and C-only reliability.** FEATURE-D outer metrics (AUROC/AUPRC/Brier) were rP=0.979642/0.992927/0.042134 and rC=0.98473/0.994406/0.0359785. These are outer-development evaluations only.

4. **P/C disagreement arbiter.** Disagreement fraction was 0.45275; the direct arbiter FEATURE-D outer AUROC/AUPRC/Brier/accuracy was 0.965473/0.955646/0.0552778/0.929114.

5. **Pathway-consistency contribution.** For rP, FEATURE-A to C to D AUROC was 0.978441 → 0.982049 → 0.979642; for rC it was 0.986285 → 0.985091 → 0.98473. This ablation is descriptive across the fixed outer folds; no untested causal claim is made.

6. **Where suppression oracle headroom occurs.** On disagreement trials, mean oracle-recoverable fraction was 0.93547; the rescue taxonomy above identifies whether attenuation, P scaling, joint scaling, or no fixed-grid reweighting was involved.

7. **Can TRAIN-only gain models distill suppression benefit?** FEATURE-D gain-alpha BA delta versus native was 0 (95% CI [-0.002, 0.00275]; positive folds 1/5).

8. **Does the learned alpha policy improve outer subjects?** The result in item 7 is the biological-subject bootstrap estimate; an interval spanning zero is not evidence of a reliable improvement.

9. **Does 2D arbitration clearly exceed suppression-only?** FEATURE-D 2D gain-policy BA delta was -0.00025 (95% CI [-0.00175, 0.0015]; positive folds 3/5), versus suppression-only 0 (95% CI [-0.002, 0.00275]; positive folds 1/5). This is a comparison of fixed TRAIN-only policies, not an oracle comparison.

10. **Why learned policies may fail.** FEATURE-D reliability-routing BA delta was 0.0005 (95% CI [0, 0.00125]; positive folds 2/5). Reliability, disagreement, pathway ablation, random-policy controls and gain-distillation results are all retained in their compact tables; a near-zero or uncertain policy delta must not be attributed to one mechanism without those diagnostics agreeing.


### Policy bootstrap summary

- C_only: -0.23925 (95% CI [-0.284506, -0.198244]; positive folds 0/5)
- P_only: -0.12825 (95% CI [-0.164013, -0.0922375]; positive folds 0/5)
- fixed_alpha_0.25: -0.04525 (95% CI [-0.0607562, -0.0314938]; positive folds 0/5)
- fixed_alpha_0.5: -0.00975 (95% CI [-0.01625, -0.00374375]; positive folds 0/5)
- fixed_alpha_0.75: -0.001 (95% CI [-0.0055, 0.003]; positive folds 2/5)
- fixed_alpha_0: -0.12825 (95% CI [-0.168506, -0.0934938]; positive folds 0/5)
- fixed_alpha_1: 0 (95% CI [0, 0]; positive folds 0/5)
- gain_2d_A: -2.77556e-18 (95% CI [-0.001, 0.001]; positive folds 2/5)
- gain_2d_C: -0.00075 (95% CI [-0.00175, 0.00025]; positive folds 0/5)
- gain_2d_D: -0.00025 (95% CI [-0.00175, 0.0015]; positive folds 3/5)
- gain_alpha_A: -0.00025 (95% CI [-0.002, 0.002]; positive folds 1/5)
- gain_alpha_C: -0.00025 (95% CI [-0.00225, 0.002]; positive folds 1/5)
- gain_alpha_D: 0 (95% CI [-0.002, 0.00275]; positive folds 1/5)
- oracle_2d: 0.03425 (95% CI [0.01925, 0.0517562]; positive folds 5/5)
- oracle_suppression: 0.02125 (95% CI [0.01025, 0.03475]; positive folds 5/5)
- reliability_A: 0 (95% CI [-0.00075, 0.00075]; positive folds 1/5)
- reliability_B: 0.00025 (95% CI [0, 0.00075]; positive folds 1/5)
- reliability_C: 0.0005 (95% CI [0, 0.0015]; positive folds 1/5)
- reliability_D: 0.0005 (95% CI [0, 0.00125]; positive folds 2/5)

## EEGConformer / OpenBMI_MI

Completed folds: 5/5.

1. **P-specific oracle headroom.** Mean P suppression-oracle BA gain was 0.05325; P minus the equal-rank-random mean was -0.03082, mean P percentile was 0.158, and mean empirical p was 0.855446. Therefore Protected-specific headroom is not supported; this report does not attribute generic reweighting headroom to P.

2. **How native errors are reweight-recoverable.** Mean subject-fold rescue counts were C_DOWN_ONLY=0.95, P_UP_ONLY=1.7, P_DOWN_ONLY=1.68, C_UP_ONLY=1.9, JOINT_REWEIGHT=3.12, NOT_REWEIGHT_RECOVERABLE=17; the largest mean category was NOT_REWEIGHT_RECOVERABLE. Mean minimum log-scale intervention distance was 0.477143. These are response-surface diagnostics, not policy performance.

3. **P-only and C-only reliability.** FEATURE-D outer metrics (AUROC/AUPRC/Brier) were rP=0.72109/0.845242/0.178946 and rC=0.791891/0.81762/0.181596. These are outer-development evaluations only.

4. **P/C disagreement arbiter.** Disagreement fraction was 0.44925; the direct arbiter FEATURE-D outer AUROC/AUPRC/Brier/accuracy was 0.713744/0.807293/0.199571/0.690161.

5. **Pathway-consistency contribution.** For rP, FEATURE-A to C to D AUROC was 0.720967 → 0.720736 → 0.72109; for rC it was 0.799723 → 0.792079 → 0.791891. This ablation is descriptive across the fixed outer folds; no untested causal claim is made.

6. **Where suppression oracle headroom occurs.** On disagreement trials, mean oracle-recoverable fraction was 0.886721; the rescue taxonomy above identifies whether attenuation, P scaling, joint scaling, or no fixed-grid reweighting was involved.

7. **Can TRAIN-only gain models distill suppression benefit?** FEATURE-D gain-alpha BA delta versus native was 0.001 (95% CI [-0.001, 0.00325]; positive folds 1/5).

8. **Does the learned alpha policy improve outer subjects?** The result in item 7 is the biological-subject bootstrap estimate; an interval spanning zero is not evidence of a reliable improvement.

9. **Does 2D arbitration clearly exceed suppression-only?** FEATURE-D 2D gain-policy BA delta was 0.0025 (95% CI [-0.00375, 0.00825]; positive folds 3/5), versus suppression-only 0.001 (95% CI [-0.001, 0.00325]; positive folds 1/5). This is a comparison of fixed TRAIN-only policies, not an oracle comparison.

10. **Why learned policies may fail.** FEATURE-D reliability-routing BA delta was 0.00075 (95% CI [-0.00025, 0.00225]; positive folds 3/5). Reliability, disagreement, pathway ablation, random-policy controls and gain-distillation results are all retained in their compact tables; a near-zero or uncertain policy delta must not be attributed to one mechanism without those diagnostics agreeing.


### Policy bootstrap summary

- C_only: -0.15125 (95% CI [-0.18275, -0.122244]; positive folds 0/5)
- P_only: -0.0175 (95% CI [-0.03175, -0.004]; positive folds 2/5)
- fixed_alpha_0.25: -0.00775 (95% CI [-0.01925, 0.0035]; positive folds 1/5)
- fixed_alpha_0.5: -0.00125 (95% CI [-0.00975, 0.00675]; positive folds 3/5)
- fixed_alpha_0.75: -2.77556e-18 (95% CI [-0.00575, 0.00575]; positive folds 2/5)
- fixed_alpha_0: -0.0175 (95% CI [-0.03225, -0.00375]; positive folds 2/5)
- fixed_alpha_1: 0 (95% CI [0, 0]; positive folds 0/5)
- gain_2d_A: 0.0015 (95% CI [-0.00325, 0.00625]; positive folds 4/5)
- gain_2d_C: 0.00175 (95% CI [-0.00425, 0.00825]; positive folds 3/5)
- gain_2d_D: 0.0025 (95% CI [-0.00375, 0.00825]; positive folds 3/5)
- gain_alpha_A: 0.0025 (95% CI [0, 0.0055]; positive folds 2/5)
- gain_alpha_C: 0.0025 (95% CI [0.00025, 0.005]; positive folds 2/5)
- gain_alpha_D: 0.001 (95% CI [-0.001, 0.00325]; positive folds 1/5)
- oracle_2d: 0.0935 (95% CI [0.0809937, 0.10725]; positive folds 5/5)
- oracle_suppression: 0.05325 (95% CI [0.04475, 0.0617562]; positive folds 5/5)
- reliability_A: 0 (95% CI [-0.00075, 0.00075]; positive folds 0/5)
- reliability_B: 0 (95% CI [0, 0]; positive folds 0/5)
- reliability_C: 0.0005 (95% CI [-0.0005, 0.00175]; positive folds 2/5)
- reliability_D: 0.00075 (95% CI [-0.00025, 0.00225]; positive folds 3/5)

## EEGConformer / OpenBMI_SSVEP

Completed folds: 5/5.

1. **P-specific oracle headroom.** Mean P suppression-oracle BA gain was 0.04; P minus the equal-rank-random mean was -0.0050025, mean P percentile was 0.42, and mean empirical p was 0.59802. Therefore Protected-specific headroom is not supported; this report does not attribute generic reweighting headroom to P.

2. **How native errors are reweight-recoverable.** Mean subject-fold rescue counts were C_DOWN_ONLY=0.875, P_UP_ONLY=1.45, P_DOWN_ONLY=0.95, C_UP_ONLY=1.35, JOINT_REWEIGHT=1.6, NOT_REWEIGHT_RECOVERABLE=11.8; the largest mean category was NOT_REWEIGHT_RECOVERABLE. Mean minimum log-scale intervention distance was 0.452513. These are response-surface diagnostics, not policy performance.

3. **P-only and C-only reliability.** FEATURE-D outer metrics (AUROC/AUPRC/Brier) were rP=0.935704/0.956749/0.0949546 and rC=0.944754/0.963557/0.0919921. These are outer-development evaluations only.

4. **P/C disagreement arbiter.** Disagreement fraction was 0.6125; the direct arbiter FEATURE-D outer AUROC/AUPRC/Brier/accuracy was 0.944542/0.939967/0.082713/0.89365.

5. **Pathway-consistency contribution.** For rP, FEATURE-A to C to D AUROC was 0.943167 → 0.936555 → 0.935704; for rC it was 0.946538 → 0.944641 → 0.944754. This ablation is descriptive across the fixed outer folds; no untested causal claim is made.

6. **Where suppression oracle headroom occurs.** On disagreement trials, mean oracle-recoverable fraction was 0.865318; the rescue taxonomy above identifies whether attenuation, P scaling, joint scaling, or no fixed-grid reweighting was involved.

7. **Can TRAIN-only gain models distill suppression benefit?** FEATURE-D gain-alpha BA delta versus native was 0.002 (95% CI [-0.0035, 0.0075]; positive folds 2/5).

8. **Does the learned alpha policy improve outer subjects?** The result in item 7 is the biological-subject bootstrap estimate; an interval spanning zero is not evidence of a reliable improvement.

9. **Does 2D arbitration clearly exceed suppression-only?** FEATURE-D 2D gain-policy BA delta was 0.0055 (95% CI [-0.00075, 0.01275]; positive folds 3/5), versus suppression-only 0.002 (95% CI [-0.0035, 0.0075]; positive folds 2/5). This is a comparison of fixed TRAIN-only policies, not an oracle comparison.

10. **Why learned policies may fail.** FEATURE-D reliability-routing BA delta was 0.00175 (95% CI [-0.00125, 0.00525]; positive folds 2/5). Reliability, disagreement, pathway ablation, random-policy controls and gain-distillation results are all retained in their compact tables; a near-zero or uncertain policy delta must not be attributed to one mechanism without those diagnostics agreeing.


### Policy bootstrap summary

- C_only: -0.2005 (95% CI [-0.23975, -0.1615]; positive folds 0/5)
- P_only: -0.222 (95% CI [-0.265506, -0.1795]; positive folds 0/5)
- fixed_alpha_0.25: -0.0735 (95% CI [-0.0967563, -0.0527438]; positive folds 0/5)
- fixed_alpha_0.5: -0.0185 (95% CI [-0.0295062, -0.00775]; positive folds 0/5)
- fixed_alpha_0.75: -0.004 (95% CI [-0.01125, 0.003]; positive folds 1/5)
- fixed_alpha_0: -0.222 (95% CI [-0.264269, -0.178994]; positive folds 0/5)
- fixed_alpha_1: 0 (95% CI [0, 0]; positive folds 0/5)
- gain_2d_A: 0.0045 (95% CI [-0.00050625, 0.01025]; positive folds 4/5)
- gain_2d_C: 0.00625 (95% CI [-0.0005, 0.01375]; positive folds 5/5)
- gain_2d_D: 0.0055 (95% CI [-0.00075, 0.01275]; positive folds 3/5)
- gain_alpha_A: 0.0015 (95% CI [-0.004, 0.00725]; positive folds 3/5)
- gain_alpha_C: 0.00225 (95% CI [-0.0035, 0.00825]; positive folds 3/5)
- gain_alpha_D: 0.002 (95% CI [-0.0035, 0.0075]; positive folds 2/5)
- oracle_2d: 0.0615 (95% CI [0.0475, 0.0757562]; positive folds 5/5)
- oracle_suppression: 0.04 (95% CI [0.02875, 0.05275]; positive folds 5/5)
- reliability_A: 0.0025 (95% CI [-0.0005, 0.006]; positive folds 3/5)
- reliability_B: 0.00225 (95% CI [-0.001, 0.00575]; positive folds 3/5)
- reliability_C: 0.00175 (95% CI [-0.001, 0.005]; positive folds 2/5)
- reliability_D: 0.00175 (95% CI [-0.00125, 0.00525]; positive folds 2/5)