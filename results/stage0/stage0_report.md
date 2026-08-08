# MOA-EEG Stage-0 report

## Scientific terminal state

`SCIENTIFIC_STOP_GATE_B`

This run is a falsification study. It does not use shared/self/cross consistency, a canonical decoder, rendering losses, uncertainty routing, conformal prediction, CVaR, or foundation-model adaptation.

## D0 reference and legality conclusion

The PhysioNet EEGMMIDB EDF metadata does not encode an exact acquisition reference. The pipeline therefore does not claim M1, M2, average mastoid, or absolute scalp potential. It uses an explicit CAR64 transformation under the documented condition that the amplifier channels shared one acquisition reference. Because direct dataset metadata is missing, EEGMMIDB is supporting controlled evidence, not the required clean explicit-reference replication. ISRUC remains mandatory before a strong Stage-1 decision.

All admitted synthetic views satisfy `Y_t=A_tY_0` by construction and `B_t=A_tB_0` numerically. Failed operators are excluded before training.

## Main table — EEGMMIDB

| method   | Method name          | status                | dataset   |   matched_ba |   unseen_ba |   operator_drop |   macro_f1 |
|:---------|:---------------------|:----------------------|:----------|-------------:|------------:|----------------:|-----------:|
| B2       | Coordinate           | complete              | eegmmidb  |       0.2664 |      0.2559 |          0.0105 |     0.2413 |
| B3       | Bipolar Midpoint     | complete              | eegmmidb  |       0.2664 |      0.2562 |          0.0102 |     0.2414 |
| B4       | Coord + Ref Metadata | complete              | eegmmidb  |       0.2711 |      0.2626 |          0.0085 |     0.2461 |
| B5       | Interpolation        | complete              | eegmmidb  |       0.2569 |      0.2566 |          0.0003 |     0.2276 |
| B6       | Signed Operator      | complete              | eegmmidb  |       0.2706 |      0.2554 |          0.0151 |     0.2392 |
| B7       | + Lifting            | not run (Gate B stop) | nan       |     nan      |    nan      |        nan      |   nan      |
| B8       | + Observability      | not run (Gate B stop) | nan       |     nan      |    nan      |        nan      |   nan      |

ISRUC is reported separately: no result is available until the official extracted-channel files and annotations are locally verified.

The absolute matched BA of the strongest non-MOA model is 0.2711. This is only marginally above four-class chance and materially limits the scientific strength of any observed operator drop.

## Gates

```json
{
  "status": "SCIENTIFIC_STOP_GATE_B",
  "gates": {
    "A": {
      "status": "PASS",
      "strongest_non_moa": "B4",
      "matched_ba": 0.2711147735869448,
      "mean_unseen_ba": 0.2625912809366824,
      "mean_operator_drop": 0.00852349265026242,
      "trigger_operator": "polarity_heldout",
      "trigger_family": "polarity",
      "trigger_unseen_ba": 0.2397250926241524,
      "trigger_operator_drop": 0.03138968096279233,
      "trigger_rei": 0.0433605194919979,
      "scope_warning": "Gate is driven by one operator; inspect all families before claiming broad operator-shift headroom"
    },
    "B": {
      "status": "FAIL",
      "strongest_baseline": "B4",
      "gain": -0.007149794799746201,
      "positive_seeds": 0,
      "seed_count": 3
    },
    "C": {
      "status": "NOT_RUN"
    },
    "D": {
      "status": "NOT_RUN"
    }
  }
}
```

## Required questions

1. Legal operator shift: Gate A was triggered by `polarity_heldout` with drop `0.0314` BA; the all-operator mean drop was `0.0085`.
2. Strongest geometry/reference baseline: `B4`.
3. Signed operator beyond geometry+metadata: `FAIL`; mean unseen gain was `-0.0071` BA with `0/3` positive seeds.
4. Lifting beyond signed operator: not run because Gate B failed; running B7 would violate fail-fast.
5. Observability conditioning beyond lifting: not run because Gate B failed.
6. Hardest observed family/method combination: polarity for B6 (mean drop 0.0288 BA).
7. Task-aware observability prediction: not estimated because B8 was prohibited by Gate B.
8. Matched-performance sacrifice for B8: not evaluated because B8 was prohibited.
9. EEGMMIDB/ISRUC direction consistency: unresolved; ISRUC was not locally available and its official MEGA delivery could not yet be verified on this server.
10. Stage-1 decision: `DO NOT PROCEED YET`. The blocking gate is `SCIENTIFIC_STOP_GATE_B`.

## Final decision

`DO NOT PROCEED YET`

## Reproduce

```bash
cd /root/autodl-tmp/hsc_tta_eeg/repo
source /root/miniconda3/etc/profile.d/conda.sh
conda activate hsc_gpu
PYTHONPATH=src python scripts/moa_stage0/run.py --config configs/moa_stage0.yaml --phase audit
PYTHONPATH=src python scripts/moa_stage0/run.py --config configs/moa_stage0.yaml --phase run
```
