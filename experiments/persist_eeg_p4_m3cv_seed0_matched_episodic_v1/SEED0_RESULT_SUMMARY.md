# M3CV seed-0 result summary

Downloaded cohort = 95. Eligible analysis cohort = 93 participants with at least eight valid trials per class in both sessions. Excluded by this pre-training availability criterion only: `sub-035`, `sub-080`.

Variable-length cache: 23,308 valid native [64,1000] epochs across eligible people; outer evaluation consumes all valid cached trials per subject/session.

| Model | Future S2 BA | Future S2 Macro-F1 | WS-BA |
|---|---:|---:|---:|
| EEGNet | 0.9151 | 0.9139 | 0.8812 |
| SIRE-EEG | 0.9159 | 0.9149 | 0.8808 |

## Paired SIRE-EEG minus EEGNet (N=93, 20,000 subject bootstrap draws)

| Metric | Mean delta | 95% CI | Improved / tied / harmed |
|---|---:|---:|---:|
| future S2 BA | +0.0009 (+0.09 pp) | [-0.0061, +0.0077] | 40 / 12 / 41 |
| WS-BA | -0.0005 (-0.05 pp) | [-0.0079, +0.0065] | 45 / 9 / 39 |

## Selected checkpoints

| Fold | Model | Selected epoch | Inner-val S2 BA | Outer S2 BA | Checkpoint SHA-256 |
|---:|---|---:|---:|---:|---|
| 0 | EEGNet | 14 | 0.9235 | 0.9115 | `f8b4c783e2058c4e20ff5cb26771eabe77ad5a2a22d1e9c95fa5017e26f4f2e8` |
| 0 | SIRE-EEG | 31 | 0.9282 | 0.9177 | `e50621c5ba773e2233340f93080c1c91336672eda5b922ed0148edcca13bf522` |
| 1 | EEGNet | 25 | 0.9408 | 0.9161 | `00e8d743c5edcb5329f81b8a0c4e05e83b32229e937aa8a2cec67baa522b133d` |
| 1 | SIRE-EEG | 22 | 0.9496 | 0.9057 | `fd1fdbe952b12043eacfa60ac3a60bac45c32d284a1f0a5bc31afd6326eb42af` |
| 2 | EEGNet | 31 | 0.9193 | 0.9269 | `9fe48219bfd6ae934039d51d9cea9e6d1882979c8fd1ef0d8242125d27556c92` |
| 2 | SIRE-EEG | 19 | 0.9176 | 0.9082 | `dc0c4c9d4280586fa17c7e42d20ada52a75988853c593f41113e11af8e40c2e5` |
| 3 | EEGNet | 24 | 0.9116 | 0.9219 | `49c10a31bdca970b44c3b34abe1f1358d1cf223ee7926b5e85af5300a740143e` |
| 3 | SIRE-EEG | 24 | 0.9020 | 0.9252 | `dc96640e69f0766baad4ae17aefe6eed6902db62a9c7d907ddd0601724c00865` |
| 4 | EEGNet | 10 | 0.9380 | 0.8984 | `8196341de04facdf1d0168df7c658808edc10d5dc7494ddf1f8eb0a21dd2986e` |
| 4 | SIRE-EEG | 26 | 0.9410 | 0.9236 | `183e5c44071b3b446060a6c3bc25a46de2782596965a853c4a80c66ba98262c0` |

Training used the user-authorized maximum-60/minimum-10/patience-8 early-stopping correction, retaining strict earliest-tie selection by inner-val S2 subject-equal BA.

Hard stop honored: seed 0 only; no PEEH, PSWA, ablation, ScaleCollapse, rank-matched control, PRD, BN-state, adaptation, suppression, other model, tuning or additional seed was run.
