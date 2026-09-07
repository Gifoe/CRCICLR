# Decision

Representation-only audit of frozen Stage-1 seed0 checkpoints. No retraining or forbidden cohort access.

1. OpenBMI absolute centroids: 0.7438
2. WBCIC absolute centroids: 0.8020
3. OpenBMI directions: 0.8692
4. WBCIC directions: 0.8445
5. Source¡úfuture direction transfer: see DIRECTION_NULL_TEST.json.
6. Session persistence: see SUBJECT_SESSION_PERSISTENCE.csv.
7. WBCIC offset/separation: 1.0016
8. Unlabeled centering: see CENTERING_DIAGNOSTIC.csv.
9. TG absolute alignment change: compare AGGREGATE_GEOMETRY.csv.
10. TG discrimination cost: compare TG_VS_ERM_GEOMETRY.csv.
11. align relations, not absolute locations: supported
12. Protocol invalidity: NO (artifact hashes, fold-local coordinates, and holdout isolation passed).

H1 discriminative-direction transfer: PASS
H2 absolute-location mismatch: PASS
H3 alignment-discrimination tradeoff: PASS

Terminal:
RELATIONAL_GEOMETRY_SUPPORTED_STOP
