# Controlled LiteBN-SC v2: C1 WBCIC seed-0 report

ARCHITECTURE_CHANGE = NO
INFERENCE_PARAMETER_INCREASE = 0
INFERENCE_FORWARD_PASSES = 1
SC_REFERENCE = R_DROP_STYLE
DUAL_CE_CONTROL = YES
BN_PERSISTENT_UPDATES_PER_BATCH = 1
NEW_SEALED_TEST_ACCESSED = NO
SEED1_2_RUN = NO

## Scope

C1 WBCIC_MI seed0 folds 0-4 were trained. B0 was replayed from exact frozen Linux checkpoints. C0 was not run under the current user instruction.
The internal-heldout cohort is DEVELOPMENT_MODEL_SELECTION_DATA, not a sealed final test.

## Results

Outer B0 BA = 0.79027432
Outer C1 BA = 0.79479696
Outer C1-B0 = +0.4523 pp
Outer improved/harmed/tied subjects = 15/13/3
Outer improved folds = 3/5; worst fold = -0.6633 pp
Heldout B0 BA = 0.79760000
Heldout C1 BA = 0.79750000
Heldout C1-B0 = -0.0100 pp
Heldout improved/harmed/tied subjects = 5/4/1

## Required interpretation

1. C1 versus original LiteBN: higher on outer; lower on heldout.
2. C1 versus DualCE: not answerable because C0 was not run in the current scope.
3. Outer and heldout direction: opposite.
4. Subject effects: outer 15 improved and 13 harmed; heldout 5 improved and 4 harmed.
5. Selected-checkpoint stochastic consistency B0/C1 pairwise KL = 0.072491/0.047153; this is observational.
6. Consistency improved while BA declined: outer=NO; heldout=YES. The consistency diagnostic is observational.
7. Mean selected inner-val BA B0/C1 = 0.795676/0.799074; no causal conclusion without C0.
8. Median ||grad(0.5J)||/||grad(CE)|| = 0.182728.
9. Persistent BN state updated once per original batch; exact BN/autograd audits passed.
10. Recorded C1 training wall time = 2793.6 s. Historical B0 artifacts record 5558.7 s, but this is not a controlled timing benchmark, so their ratio does not estimate compute change. C1 executes two training forwards versus B0's one. C1 peak allocated CUDA memory = 5.856 GiB; B0 peak memory was not recorded, so the memory increase is unknown.
11. Inference remains exact historical LiteBN with one deterministic forward.
12. Phase2 executed: NO. Reason: At least one C1-versus-B0 gate condition failed.

SC_NO_CONTINUATION_SIGNAL
