# V5.1 repository and result audit

- Branch: `v5-1-calibration-granularity-diagnostic`
- Audit commit: `85933f70565b9b48d31105ecca7c9b51cc4c54ee`
- V5 start commit: `85933f70565b9b48d31105ecca7c9b51cc4c54ee`
- Stage-0 run commit: `ae6c5a039935736d350da6d952463ff1bd83e574`
- Ordinal fix commit: `b3c2e25a0bb5b13b09cb48736663b6a16da41c69`
- `git merge-base --is-ancestor`: **True**
- Git status at audit: `?? configs/budgeted_risk_v51/
?? delivery/budgeted_risk_v51/
?? scripts/budgeted_risk_v51/
?? src/hsc_tta/budgeted_risk/diagnostics/
?? tests/budgeted_risk_v51/`
- Old state/verdict: `STOPPED_NO_GO` / `STAGE0_NO_GO`
- Method-development subjects: HMC=90, EEGMMIDB=65
- BUDGET_RESULTS: 113,925 rows (temporal=5,425, random=108,500)
- BUDGET_QUERY_TRANSCRIPTS: 1,295,910 rows
- Source-cache manifest rows: 3,875; protected overlap: 0
- Protected flags: formal=false, internal_final=false, CAP=false

## Fold sizes and finite-sample ranks

| dataset | screening_fold | n | s1_m | s1_k | s2_m | s2_k |
| --- | --- | --- | --- | --- | --- | --- |
| eegmmidb | 0 | 14 | 14 | 14 | 28 | 27 |
| eegmmidb | 1 | 14 | 14 | 14 | 24 | 23 |
| eegmmidb | 2 | 14 | 14 | 14 | 23 | 22 |
| eegmmidb | 3 | 10 | 10 | 10 | 27 | 26 |
| eegmmidb | 4 | 13 | 13 | 13 | 28 | 27 |
| hmc | 0 | 12 | 12 | 12 | 42 | 39 |
| hmc | 1 | 26 | 26 | 25 | 32 | 30 |
| hmc | 2 | 16 | 16 | 16 | 36 | 34 |
| hmc | 3 | 16 | 16 | 16 | 32 | 30 |
| hmc | 4 | 20 | 20 | 19 | 38 | 36 |

S1 has m=13 (EEGMMIDB) or m=18 (HMC), hence k=m at delta=0.10 in every fold: the selected correction is the maximum residual. S2 has m=26 or 36 and k=25 or 34.

## Input hashes

```json
{
  "delivery/budgeted_risk/stage0/STAGE0_DECISION.json": "be7bcd49dbd3af510b4d0486c40a8c639b1f3e60669b4a0ab3953861c04b52fb",
  "delivery/budgeted_risk/stage0/STAGE0_METHOD_FREEZE.json": "640941ad0b8aa8249f97b3844681e383941848ac6c73eeaef56cc9821824422c",
  "outputs/budgeted_risk/RUN_STATE.json": "549b0b6d56923fd5c86efa75d157d491fcfac3d7f72243c614e13ab1f9ae3c5f",
  "outputs/budgeted_risk/features/UNLABELED_CONTEXT_FEATURES.parquet": "81915a277ebcf82d731070306aa5329e674bfea849b20473cf023617e3d4cd7b",
  "outputs/budgeted_risk/stage0/BUDGET_QUERY_TRANSCRIPTS.parquet": "a1d73634163a2d3afd1a990e4e01f1aa87d6d219a1dd802e1667d17a0f9909df",
  "outputs/budgeted_risk/stage0/BUDGET_RESULTS.parquet": "3f26e2d3e835f9ef605ab68ec50b52e72c4a4c830f992b873385d8a9c979fefb",
  "outputs/budgeted_risk/stage0/BUDGET_TUNING.parquet": "dbbfd1d261818398fccaf3c44efc41db745f6f925e54363a56c4bba856d2ed8e",
  "outputs/budgeted_risk/stage0/FULL_CONTEXT_MODEL_SELECTION.parquet": "dfd5f8d71f39eb645fb3a1cfc07519a64f9fada1441ca9446d6f18d9156050a3",
  "outputs/budgeted_risk/stage0/FULL_CONTEXT_RESULTS.parquet": "806cacc772da6aa02e36d870ace884c145281b56b775d91c5a99ad15337ce522"
}
```
