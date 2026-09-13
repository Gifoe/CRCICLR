# Controlled LiteBN-SC v2: WBCIC seed-0 report

ARCHITECTURE_CHANGE = NO
INFERENCE_PARAMETER_INCREASE = 0
INFERENCE_FORWARD_PASSES = 1
SC_REFERENCE = R_DROP_STYLE
DUAL_CE_CONTROL = YES
BN_PERSISTENT_UPDATES_PER_BATCH = 1
NEW_SEALED_TEST_ACCESSED = NO
SEED1_2_RUN = NO

## Primary results

Outer B0/C0/C1 BA = 0.79027432/0.79140701/0.79479696
Outer C1-B0 = +0.4523 pp; C0-B0 = +0.1133 pp; C1-C0 = +0.3390 pp
Heldout B0/C0/C1 BA = 0.79760000/0.79990000/0.79750000
Heldout C1-B0 = -0.0100 pp; C0-B0 = +0.2300 pp; C1-C0 = -0.2400 pp
Internal-heldout is DEVELOPMENT_MODEL_SELECTION_DATA, not a sealed final test.

## Required interpretation

1. C1 exceeds exact B0 on outer: True; on heldout: False.
2. C1 exceeds matched DualCE C0 on outer: True; on heldout: False.
3. C1-B0 outer/heldout directions are opposite.
4. C1-B0 subjects improved/harmed/tied: outer 15/13/3; heldout 5/4/1.
5. Mean selected-checkpoint stochastic pairwise KL B0/C0/C1 = 0.072491/0.074960/0.047153.
6. C1 is more consistent than C0 while lower in BA: outer=False; heldout=True.
7. Mean selected inner-val BA B0/C0/C1 = 0.795676/0.799672/0.799074.
8. Median observational ||grad(0.5J)||/||grad(CE)|| C0/C1 = 0.262985/0.182728.
9. Persistent BN state updated once per batch; all audits passed: True.
10. Recorded wall time B0/C0/C1 = 5558.7/2786.1/2793.6 s; these historical/sequential measurements are not a controlled timing benchmark. Peak allocated CUDA C0/C1 = 5.856/5.856 GiB; B0 peak was not recorded.
11. All methods use the exact historical LiteBN and one deterministic inference forward.
12. Phase2 executed: NO. WBCIC continuation gate failed.

SC_NO_CONTINUATION_SIGNAL
