@echo off
setlocal
set "PERSIST_WBCIC_ROOT=D:\nips-temp\TotalP\P1\CRCICLR_WBCIC_EEGNET"
set "PERSIST_WBCIC_RAW_ROOT=D:\nips-temp\TotalP\P2\nm000348_v1.0.4_bids"
set "EXP=D:\nips-temp\TotalP\P1\CRCICLR_EXP4_PROTECTION_FIRST_FINAL\experiments\persist_eeg_exp4_protection_first_final"
set "PY=D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe"
if not exist "%EXP%\logs" mkdir "%EXP%\logs"
"%PY%" -u "%EXP%\code\run_exp4.py" all_dev --device cuda > "%EXP%\logs\all_dev.log" 2>&1
exit /b %errorlevel%
