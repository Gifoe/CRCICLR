# Reproducibility

- Parent experiment commit: `57d5e4f1ae0a7c80d95ca27983fedad2ec3f690c`.
- Server Python: `D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe`.
- Development folds: frozen OpenBMI 40-subject and WBCIC 41-subject five-fold
  protocols; historical ERM seed 0.
- Repair-2 layer: `final_embedding` only.
- Repair-2 alpha solver: fixed grid 0..0.25 in 1/64 increments; largest
  Session-1 source-support-admissible value.
- Independent validity: Session 2 only; subject-cluster bootstrap with 10,000
  deterministic resamples.
- Runtime features are not committed.  Compact metrics, hashes, reports, and
  figures are committed.

`protocol/STAGE0_REPAIR2_PROTOCOL_LOCK.json` freezes the scientific protocol.
`protocol/PRE_STAGE0_REPAIR2_FREEZE.json` hashes the lock and execution code
before outcomes.  All 20 units verify their historical feature scope, scaling,
probe BA, and V0 hash.  `protocol/STAGE0_REPAIR2_E1_ENGINEERING_FREEZE.json`
locks the seven numerical outputs produced before the presentation-only E1
figure fix and requires byte-identical values after rerun.

`results/STAGE0_REPAIR2_VALIDATION.json` reports validator pass, 20 units, 2/4
all-gate settings, no future-session access, and sealed resources untouched.
Repair-2 result SHA256: `88257bb74c50636a0bf084f905178906253f43736af4bf58178539c659fd3c80`.
Repair-2 validation SHA256: `311d993373f3b385308b73c902830b4a84f1ae8c144fbeff856a13b6e5991b1c`.

Commands:

```powershell
& $python code\freeze_stage0_repair2.py
& $python code\run_stage0_repair2.py
& $python code\analyze_stage0_repair2.py
& $python code\validate_stage0_repair2.py
& $python code\finalize_stage0_repair2_failure.py
& $python code\validate_final_closure.py
```
