# PERSIST-EEG final constructive attempt: SCST-DR

This experiment tests **Source-Certified Class-Conditional Subject Transport
for Decision Robustness (SCST-DR)**.  The experiment is deliberately gated:
Stage 0 tests whether latent residual arithmetic is subject-faithful,
class-preserving, and non-pathological using source/development resources only.
No SCST-DR model training or sealed-outer access is legal unless Stage 0 ends in
`TRANSPORT_VALIDITY_SUPPORTED`.

The experiment starts from P4D commit
`57d5e4f1ae0a7c80d95ca27983fedad2ec3f690c`.  Historical runtime caches and
checkpoints are read-only inputs and are not committed here.

Run order on the GPU server:

```powershell
$python = 'D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe'
& $python code\audit_data_access.py
& $python code\freeze_stage0.py
& $python code\run_stage0.py
& $python code\analyze_stage0.py
& $python code\validate_stage0.py
```

If the validator reports a negative Stage-0 terminal, execution stops.  It is
not permissible to train SCST-DR or open sealed resources after that result.

## Final status

`FINAL_CONSTRUCTIVE_HYPOTHESIS_NOT_SUPPORTED`.  V0 failed subject fidelity at
alpha=1.  Repair-1 restored subject/class fidelity at alpha=0.25 but failed the
WBCIC absolute manifold gate.  The final source-support-constrained Repair-2
passed 2/4 settings; both WBCIC 3NN ratios remained above the unchanged 1.25
limit.  SCST training and sealed outer evaluation were not authorized.  See
`STAGE0_REPAIR2_REPORT.md` and `SCST_DR_FINAL_REPORT.md`.

Repair and closure commands:

```powershell
& $python code\freeze_stage0_repair1.py
& $python code\run_stage0_repair1.py
& $python code\analyze_stage0_repair1.py
& $python code\validate_stage0_repair1.py
& $python code\freeze_stage0_repair2.py
& $python code\run_stage0_repair2.py
& $python code\analyze_stage0_repair2.py
& $python code\validate_stage0_repair2.py
& $python code\finalize_stage0_repair2_failure.py
& $python code\validate_final_closure.py
```
