# P4B Final Report

Exact terminal: `P4B_IDENTITY_RELIABILITY_CONDITION_NOT_SUPPORTED`.

The validator passed before this final terminal was written.

## Frozen design

- P4A Lean input tip: `281e7c66c9818b5d2efe96968900cb585af20287`.
- Discovery settings: S1, S2, S3, S5.
- P4C reserved settings: S4, S6.
- Normalization: within-setting median/MAD with frozen SD fallback.
- E_task: `(z_D + z_C + z_O)/3`; D/C/O were source-only nondegenerate and retained without learned weights.
- Ridge alpha: 1; primary CV: leave-one-entire-setting-out.
- Bootstrap: 10,000 draws, setting -> fold -> seed/run -> direction -> outcome subject.

## Model results

| model   |   LOSO_setting_RMSE |
|:--------|--------------------:|
| M0      |          0.00814200 |
| MADD    |          0.02157296 |
| ME      |          0.02151801 |
| MI      |          0.00796733 |
| MINT    |          0.02286385 |

- RMSE_MI - RMSE_MINT: -0.014896525; CI [-0.032068833931208574, 0.0005675010482946285].
- RMSE_MADD - RMSE_MINT: -0.001290898; CI [-0.004933686191682303, 0.0009433960303707852].
- beta_IxE: 0.000094884; CI [-0.0015074821732454515, 0.0006757057062616963].
- slope_lowE: 0.000111484; slope_highE: 0.000301252.
- DeltaSlope: -0.000189769; CI [-0.0013514114125233927, 0.0030149643464909004].
- DeltaRegime: 0.006690019; CI [0.000983032858042108, 0.012903471187988244].
- Per-setting primary-direction consistency: 75.0%.
- Gates: {'G1': False, 'G2': False, 'G3': False, 'G4': True, 'G5': True, 'G6': True}.

## Purity and prospective lock

- S4/S6 future direction utilities: untouched during P4B.
- OpenBMI sealed internal holdout: untouched.
- WBCIC outer 10: untouched and unenumerated.
- Trial pseudoreplication: absent; utilities were computed subject-first.
- Serialized-basis metadata recovery: documented in `ENGINEERING_RECOVERY_BASIS_HASH.md`; every intervention direction was either byte-exact or passed the frozen source D/O/geometry/persistence equivalence gate.
- P4C lock: `NOT_REQUIRED_FOR_NOT_SUPPORTED`. P4C was not executed.

The result is reported without outcome-driven rescue, nonlinear model search, alpha tuning, learned D/C/O weights, threshold changes, or cherry-picked settings.
