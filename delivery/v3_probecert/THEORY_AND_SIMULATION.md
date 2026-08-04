# Theory and simulation

For a frozen policy, let each exchangeable calibration subject contribute one scalar joint critical index. With `k=ceil((m+1)(1-delta))`, the k-th order statistic has marginal subject-level coverage at least `1-delta`; when `k>m`, returning the full-set sentinel is required. This is a standard split-conformal order-statistic argument, not a new theorem.

Proof sketch: append the test subject's exchangeable score to the m calibration scores. Its rank is uniform up to ties; the probability its rank exceeds k is at most delta. Encoding either risk failure or noninferiority failure as the sentinel makes the scalar event equivalent to the requested joint event. Exchangeability violation removes the rank argument.

The simulation used 5,000 repetitions for every one of 342 settings. Under exchangeability the minimum observed validity gap was 0.0166. Small calibration sizes correctly returned the sentinel when k exceeded m. Policy-level calibration was less conservative than actionwise simultaneous calibration. Site shift reduced validity, demonstrating that the guarantee is not transportable without exchangeability.

Site-shift sensitivity:

|   site_shift |   joint_validity |   nominal_validity |   validity_gap |   sentinel_probability |   policy_efficiency |
|-------------:|-----------------:|-------------------:|---------------:|-----------------------:|--------------------:|
|       0.0000 |           0.9438 |             0.9000 |         0.0438 |                 0.6024 |              0.1660 |
|       0.2500 |           0.9270 |             0.9000 |         0.0270 |                 0.5888 |              0.1721 |
|       0.5000 |           0.8980 |             0.9000 |        -0.0020 |                 0.5880 |              0.1730 |
