# PERSIST-EEG P5.1 + P6

Status: `PERSISTENCE_GEOMETRY_HAS_NO_DECISION_FUSION_HEADROOM`

## Required answers

1. TRAIN-only tuning rescued ICG: `False`. Best version: `V1`, mean Delta_BA `-0.000648`.

2. V3 authorized: `False`.

3. Stronger geometry alignment improved decoding under the frozen viability gates: `False`.

4. Protected geometry showed complementary errors: `True`.

5. TRAIN-only calibrated Protected fusion passed the strict-inductive gain requirements: `False`. Mean Delta_BA `-0.001019`; positive runs `0/6`; CI `[-0.0036111111111111122, 0.0]`.

6. Control mean effects:

- same_rank_random: mean Delta_BA `-0.000466`
- protected_uniform: mean Delta_BA `-0.001019`
- shuffled_weights: mean Delta_BA `-0.001019`
- all_persistence: mean Delta_BA `0.000000`
- full_canonical: mean Delta_BA `0.000000`

At least one mandatory control matches or exceeds the Protected mean gain: `True`.

7. Terminal label: `PERSISTENCE_GEOMETRY_HAS_NO_DECISION_FUSION_HEADROOM`.

Outer-test used: `false`.
