# Adapter fidelity audit

Adapter selection was frozen before any repaired performance metric and did not use downstream performance.

## cbramod

- Fidelity: **PASS**
- V7 consistent: False
- Repair: restore structured channel-patch tokens for diagnostic heads; retain V6 probabilities as primary anchor
- Limitation: None
- Evidence: `data/embeddings_tokens_v2/*/*.h5`, `outputs/online_blockwise_v6/results/BACKBONE_GATE.csv`

## labram

- Fidelity: **FAIL**
- V7 consistent: False
- Repair: remove pre-head subwindow/global pooling and retain ordered tokens
- Limitation: HMC C3-M2/C4-M1 are bipolar derivations, but the V7 adapter strips M2/M1 and assigns referential C3/C4 positional tokens. Official LaBraM code provides no checkpoint-faithful mapping for those bipolar signals.
- Evidence: `external/LaBraM/README.md:54`, `external/LaBraM/utils.py:42-117`, `external/LaBraM/utils.py:713-717`, `external/LaBraM/modeling_finetune.py:349-388`, `external/LaBraM/dataset_maker/make_TUAB.py:66-69`

## biot

- Fidelity: **FAIL**
- V7 consistent: False
- Repair: remove pre-head sequence mean and retain ordered transformer tokens
- Limitation: HMC C3-M2/C4-M1 were assigned PREST token indices 10/14, whose frozen semantics are C3-P3/C4-P4. The signals are not those montages and cannot be reconstructed from the two HMC derivations. This mapping has no official/checkpoint fidelity basis.
- Evidence: `external/BIOT/README.md:52`, `external/BIOT/model/biot.py:75-143`, `external/BIOT/run_example.py:96-109`

## Gate F

- F1: **False**
- F2: **True**
- F3: **True**
- F4: **True**
- F5: **True**
- F6: **True**
- F7: **True**

The gate fails at F1 because the HMC bipolar derivations cannot be represented with checkpoint-faithful LaBraM or BIOT channel identities. Removing mean pooling repairs token preservation but cannot repair signal identity.
