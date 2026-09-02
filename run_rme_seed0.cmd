@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
cd /d D:\nips-temp\TotalP\P1\CRCICLR_CANONICAL_EEGNET
if not exist experiments\persist_eeg_risk_mode_marginalization_final\runtime mkdir experiments\persist_eeg_risk_mode_marginalization_final\runtime
"D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe" experiments\persist_eeg_risk_mode_marginalization_final\code\rme_seed0_pilot.py --run-seed0 --device cuda > experiments\persist_eeg_risk_mode_marginalization_final\runtime\seed0.log 2>&1
echo %ERRORLEVEL% > experiments\persist_eeg_risk_mode_marginalization_final\runtime\seed0.exit
