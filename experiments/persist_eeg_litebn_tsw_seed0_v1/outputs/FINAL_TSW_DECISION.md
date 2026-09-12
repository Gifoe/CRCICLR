# LiteBN-TSW seed0 fail-fast decision

`TASK_STOP_OPENBMI_MI_NO_BROAD_SIGNAL`

The user-requested task-level fail-fast gate stopped the experiment after the
first task. OpenBMI MI completed all five StaticScale and five TSW cells, all 60
epochs, followed by exact-baseline outer and current internal-heldout evaluation.
WBCIC MI, ERP and SSVEP were not launched; no four-task aggregate is claimed.

| Population | Model | LiteBN BA | Model BA | Delta BA pp |
|---|---|---:|---:|---:|
| Outer | StaticScale | 0.791500 | 0.767750 | -2.375 |
| Outer | TSW | 0.791500 | 0.766000 | -2.550 |
| Internal heldout diagnostic | StaticScale | 0.744857 | 0.735571 | -0.929 |
| Internal heldout diagnostic | TSW | 0.744857 | 0.734571 | -1.029 |

TSW was negative on all five outer folds and was 0.175 pp worse than StaticScale.
The internal heldout direction also reversed negatively. Consequently the required
4/4 broad signal is impossible in this staged run and the next task is unauthorized.

Correctness evidence: exact historical CompactLite was imported and common state
was bitwise identical; StaticScale and lambda-zero TSW initial logits matched in
eval/train and FP32/AMP with zero prediction mismatch. Historical baseline metrics
were replayed within 1e-10. TSW adds 1,252 parameters; StaticScale adds 3. Scale
behavior is recorded in `OpenBMI_MI_TSW_SCALE_DIAGNOSTICS.csv` and was not used
for checkpoint selection.

```text
MODEL = LiteBN-TSW
BASE_MODEL = EXACT_HISTORICAL_LITEBN
SEED = 0
COMPLETED_TASKS = OpenBMI_MI
UNRUN_TASKS = WBCIC_MI,OpenBMI_ERP,OpenBMI_SSVEP
TEMPORAL_KERNELS = 15,63,127
SCALE_GATE = SAMPLE_DEPENDENT
SCALE_DESCRIPTOR = BRANCH_TIME_MEAN
SCALE_MLP = 48-24-3
LAMBDA_INITIALIZATION = 0
RESIDUAL_TEMPORAL_MIXER = NO
CHANNEL_GATE = NO
MODEL_WIDENING = NO
LOCAL_STATS = NO
AUXILIARY_LOSS = NO
NEW_NORMALIZATION = NO
ENSEMBLE = NO
FUSION = NO
DISTILLATION = NO
ROUTING = NO
TEST_TIME_ADAPTATION = NO
NEW_SEALED_FINAL_DATA_ACCESSED = NO
```
