# MOA Stage-0 protocol amendments

## Pre-Gate operator coverage amendment

The first implementation contained three held-out operator definitions. After one B4 development seed had completed, but before any Gate was calculated, this was judged inadequate for Gate C: a correlation based on three operator points is not credible and two of those points can share the same observability spectrum.

The incomplete output tree was invalidated intact. The test catalog was expanded by construction, without selecting definitions from performance:

- sparse subsets with 12, 8, and 4 channels;
- three disjoint bipolar topology sets;
- two explicit polarity reversals;
- one CAR rereference derived only from the available eight channels.

Training and validation operators, subject splits, thresholds, seeds, model, and optimizer were unchanged. All new test operators must pass the same D0 `B_t=A_tB_0` audit before use.
