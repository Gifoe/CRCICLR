# LiteBN-StatsResidual WBCIC seed0 report

MODEL = LiteBN-StatsResidual
TASK = WBCIC_MI
SEED = 0

BASE_MODEL = EXACT_FROZEN_TRAINED_B0_LITEBN
BASE_MODEL_TRAINABLE = NO
BASE_MODEL_PARAMETER_DRIFT = 0

RESIDUAL_FEATURE = CONCAT(FINAL_TEMPORAL_FEATURE_MEAN, FINAL_TEMPORAL_FEATURE_STD_UNBIASED_FALSE)
RESIDUAL_DIM = 128
RESIDUAL_HEAD = LINEAR_128_TO_2
NEW_TRAINABLE_PARAMETERS = 258
RESIDUAL_INITIALIZATION = EXACT_ZERO
EPOCH0 = EXACT_B0
EPOCH0_CHECKPOINT_ALLOWED = YES

MLP = NO
GATE = NO
ALPHA = NO
C0 = NO
KL = NO
DISTILLATION = NO
TEST_TIME_ADAPTATION = NO

INFERENCE_MODEL_COUNT = 1
INFERENCE_FORWARD_PASSES = 1
NEW_SEALED_TEST_ACCESSED = NO
INTERNAL_HELDOUT_STATUS = DEVELOPMENT_MODEL_SELECTION_DATA

The user explicitly selected the recovered historical 64-channel LiteBN checkpoints on the Windows server. Consequently, exact replayed historical values below replace the prompt's different same-Linux numerical reference.

- B0 outer BA: 0.787040159661
- StatsResidual outer BA: 0.786394998371
- StatsResidual-B0 outer: -0.064516 pp
- Positive/negative/tied folds: 0/1/4
- Worst fold delta: -0.333333 pp
- Positive/negative/tied outer subjects: 0/3/28
- Outer subject bootstrap 95% CI: [-0.145161, +0.000000] pp
- B0 internal-heldout BA: 0.792200000000
- StatsResidual internal-heldout BA: 0.792100000000
- StatsResidual-B0 heldout: -0.010000 pp
- Positive/negative/tied heldout subjects: 4/5/1
- Heldout subject bootstrap 95% CI: [-0.070000, +0.050000] pp
- Selected epochs: [0, 0, 1, 0, 0]
- Folds selecting epoch0: 4
- Mean selected residual-head norm: 0.002835140
- Mean outer changed-prediction percentage: 0.129136%

STATSRES_NO_USEFUL_SIGNAL
