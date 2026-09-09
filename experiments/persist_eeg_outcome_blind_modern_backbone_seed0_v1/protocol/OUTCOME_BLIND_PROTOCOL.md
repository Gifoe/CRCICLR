# Outcome-blind protocol — modern backbone seed0

This file is frozen before any fresh outer-dev pair utility is generated.

## Backbone pool
EEGNet, LiteBN, TCFormer, ST-EEGFormer-small, LaBraM-base, CBraMod

## Data and split
OpenBMI MI and WBCIC MI; exact carrier FIVEFOLD_SPLIT.json; seed 0 only; source sessions and future session semantics are inherited from the carrier experiment.

## Model-native adapters
The cache is 250 Hz × 1000 samples. TCFormer consumes it directly. ST-EEGFormer-small uses a fixed linear 250 Hz → 128 Hz, 512-sample adapter before its official tokenization. LaBraM-base uses a fixed linear 250 Hz → 200 Hz, 800-sample adapter and four 200-sample patches. CBraMod uses a fixed five 200-sample patch reshape preserving all cache samples. These adapters were fixed before outcome reveal and are not selected from outcomes.

## Stage A seal
Only inner-train/source sessions and inner-validation future-session rows were accessed. No outer-dev prediction or pair utility is present in this stage. Competence is fixed as mean inner-validation subject-equal BA >= EEGNet mean minus 3 percentage points. Primary diagnostic is Exclusive-Correct Complementarity (EC). Primary fusion is raw-logit 50/50; G_dev is descriptive only.

## Selection
For each dataset, the EC-selected and G_dev-selected pair are argmax over FRESH + competent pairs, frozen in STAGE_A_BLIND_PREDICTIONS.csv. Fresh pair definitions are fixed by PAIR_OUTCOME_STATUS.csv.
