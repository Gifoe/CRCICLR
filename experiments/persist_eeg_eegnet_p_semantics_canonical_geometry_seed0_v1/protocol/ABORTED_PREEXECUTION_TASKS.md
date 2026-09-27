# Pre-execution task correction

Three server-side scheduled tasks were stopped while they were waiting for the decoder queue. None had launched a Python transform/permutation process or created scientific output:

- `PERSIST_EEG_P_SEMANTICS_EEGNET_MI_F0_EMBEDDING_TRANSFORM_PILOT_V1`
- `PERSIST_EEG_P_SEMANTICS_EEGNET_MI_F0_EMBEDDING_PERMUTATION_PILOT_V1`
- `PERSIST_EEG_P_SEMANTICS_EEGNET_MI_F0_SHARED_TRANSFORM_V2_PILOT`

The V1/V2 unsupervised transform fitting itself used unlabeled TRAIN moments, but its low-rank cross-validation scored a *labeled class-relation residual*. Under the protocol's strict “No labels” interpretation, this is not an unsupervised rank choice. The waiting tasks and their logs were preserved on the original server, and no completed geometry, semantic, or decoder result was rerun. The version-forward V3 runner selects the unsupervised rank by held-TRAIN-subject unlabeled session-mean and Gram-covariance residuals. The label-assisted oracle still uses TRAIN class correspondence and is marked diagnostic-only. A contract test verifies that replacing all class-centroid values leaves the unsupervised rank scores unchanged.
