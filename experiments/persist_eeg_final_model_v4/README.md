# PERSIST-EEG final model V4

V4 attacks the prospective expert-selection/aggregation bottleneck above the
frozen B6 KEEP ensemble. OpenBMI is an exploratory architecture sandbox;
WBCIC is restricted to the 41 authorized development subjects. The sealed
WBCIC outer cohort is not read.

Final terminal state: `GENERIC_DYNAMIC_ENSEMBLE_WINS`.

- OpenBMI masked positive KEEP pool: BA `0.850962`, Delta vs B6
  `+0.452 pp`, subject-bootstrap 95% CI `[+0.202,+0.721] pp`, 5/5 folds
  positive. OpenBMI remains exploratory.
- WBCIC-dev strongest static reference: five-expert probability mean, BA
  `0.803626`.
- Direct architecture transfer: `-0.170 pp` vs the corrected WBCIC reference.
- Best WBCIC-dev adapted generic linear stack: BA `0.806789`, `+0.316 pp`,
  CI `[-0.184,+0.817] pp`; this is not robust.
- ACTION beyond dynamic KEEP: `-0.452 pp` with a fully negative CI.
- PERSIST beyond KEEP+ACTION: `-0.087 pp`, CI crosses zero, with no safety
  improvement.
- `READY_FOR_OUTER_FREEZE=false` and `OUTER_TEST_USED=false`.

Server execution order (all modelling and result generation ran on the
server):

```powershell
$python = 'D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe'
& $python experiments\persist_eeg_final_model_v4\code\reconstruct.py
& $python experiments\persist_eeg_final_model_v4\code\run_search.py --stage initial
& $python experiments\persist_eeg_final_model_v4\code\run_keep_refine.py
& $python experiments\persist_eeg_final_model_v4\code\run_keep_dynamic.py
& $python experiments\persist_eeg_final_model_v4\code\build_wbcic_experts.py --device cuda --workers 0
& $python experiments\persist_eeg_final_model_v4\code\audit_wbcic_static.py
& $python experiments\persist_eeg_final_model_v4\code\run_wbcic_transfer.py
& $python experiments\persist_eeg_final_model_v4\code\run_wbcic_keep_search.py
& $python experiments\persist_eeg_final_model_v4\code\run_wbcic_linear_refine.py
& $python experiments\persist_eeg_final_model_v4\code\run_final_ablations.py
& $python experiments\persist_eeg_final_model_v4\code\run_final.py
```

Every OOF estimate uses subject-disjoint outer folds and a separate inner
calibration subject set. V1/V2/V2.1/V3 artifacts are read-only provenance.

`OUTER_TEST_USED=false`

Primary artifacts:

- `outputs/SCIENTIFIC_REPORT.md`: full 25-question scientific report.
- `outputs/leaderboard/FINAL_DEVELOPMENT_TABLE.csv`: two-benchmark table.
- `outputs/ablations/FINAL_MODEL_ABLATIONS.csv`: A0-A9 results.
- `outputs/final_lock/FINAL_MODEL_SPEC.*`: development research freeze.
- `outputs/figures/`: editable/vector/raster figures plus exact source data.
