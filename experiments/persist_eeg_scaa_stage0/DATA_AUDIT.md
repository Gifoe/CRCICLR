# SCAA Stage-0 data audit

- Dataset: WBCIC / NEMAR nm000348 only.
- Authorized development subjects: 41; IDs: `1,2,3,5,6,7,9,11,12,13,14,16,17,18,19,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,41,42,44,45,47,48,49,50`.
- Cache: `(24591, 58, 1000)` `float16`; sessions S1/S2/S3 = 0/1/2.
- Sealed outer: 10 subjects; identifiers absent and not enumerated, accessed, preprocessed, or evaluated.
- Folds: the frozen five subject-disjoint folds. Each development subject is an outcome target exactly once.
- Anchors: 30 competent ERM checkpoints (2 backbones x 5 folds x 3 seeds).
- For each target, the checkpoint is taken from its outcome fold; target membership is disjoint from model-fit subjects.
- EEGNet anchor source: P3 WBCIC independent replication; S1+S2 model-fit subjects, validation-discovery S3, held-target S3 competence only.
- EEGConformer anchor source: P4A S4; S1+S2 model-fit subjects, S1+S2 validation subjects, held-target S3 competence only.
- No target S2/S3 label is used by the new adaptation recipe, hyperparameter selection, or checkpoint selection.
- No adaptation utility was inspected during this audit.
