# PERSIST-EEG autonomous SCST repair: final source-only report

## Decision

Three preregistered constructive rounds were completed on the allowed
OpenBMI session-1 to session-2 and WBCIC S1 to S2 source units (5 folds x 3
seeds, 30 units per round).  Every round failed its frozen source gate.  The
final decision is:

`SCST_CONSTRUCTIVE_SEARCH_EXHAUSTED`

The round-level R3 terminal is `R3_SOURCE_GATE_FAILED`.  No additional
constructive search, threshold change, recipe change, or architecture-level
confirmation is justified by these data.

## Round summary

| round | method | class fidelity OpenBMI/WBCIC | coverage OpenBMI/WBCIC | candidate survival | target-distance CI lower OpenBMI/WBCIC | utility gate |
|---|---|---:|---:|---:|---:|---|
| R1 | Task-protected Bures | 0.3130 / 0.3264 | 0.4037 / 0.4095 | 0.4162 | 0.2158 / 0.3231 | failed |
| R2 | Low-rank local OT | 0.1961 / 0.2245 | 0.3370 / 0.3600 | 0.3486 | 0.2631 / 0.3595 | failed |
| R3 | Task-protected local OT | 0.1970 / 0.2329 | 0.3404 / 0.3675 | 0.3542 | 0.2257 / 0.3198 | failed |

All three rounds had positive target-affinity distance and NLL confidence
lower bounds and non-negligible displacement.  This does not rescue the
method: semantic validity and utility are the admissibility requirements.

## Final R3 utility evidence

The six frozen recipes were `(q, lambda_T) = (0.25, 0.25), (0.25, 0.50),
(0.25, 1.00), (0.50, 0.25), (0.50, 0.50), (0.50, 1.00)`.  Every recipe failed.
Primary delta versus ERM was negative for both datasets in every recipe:

* OpenBMI: `-0.0029166667` to `-0.0022420635`; paired CI lower bounds
  `-0.0056746032` to `-0.0047023809`.
* WBCIC: `-0.0035866798` to `-0.0015475897`; paired CI lower bounds
  `-0.0063775971` to `-0.0033272768`.

Mean delta versus Mixup was negative for all recipes.  The structured-minus-
matched-random differences were small and had no positive CI lower bound:

* OpenBMI range: `-0.0003670635` to `+0.0003075397`.
* WBCIC range: `-0.0012708385` to `+0.0007682512`.

Thus R3 did not establish structure-specific utility.  Its class-fidelity
means (`0.1970`, `0.2329`) and coverage (`0.3404`, `0.3675`) also remain far
below the frozen gates (`0.90`, `0.50`).

## Resource and claim boundaries

* Source results: 30/30 R3 units, 3,645 subject-method rows.
* Future architecture confirmation: `NOT_RUN_SOURCE_GATE`.
* ATCNet/EEGNeX/other architecture-level confirmation: not authorized.
* WBCIC outer and all sealed resources: `NOT_OPENED`.
* S3 resources: `NOT_OPENED`.
* No runtime, checkpoint, raw EEG, or large embedding files are part of this
  submission.

The strongest defensible claim is negative and source-scoped: within the
tested OpenBMI/WBCIC development protocol, the three specified SCST repair
operators did not satisfy semantic-validity or utility gates.  The results do
not support a positive SCST utility claim or any outer/sealed generalization
claim.

## Reproduction commands

The server ran the following final-round command after the frozen protocol was
committed:

```text
D:\P2\.conda\gpu-baseline-v1\python.exe -u D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC\experiments\persist_eeg_scst_autonomous_repair\code\repair_r3.py --all --aggregate
```

The R1 and R2 commands were identical apart from `repair_r1.py` and
`repair_r2.py`.  Compact outputs are under `results/`; the runtime namespace
is intentionally excluded by `.gitignore`.
