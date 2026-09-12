$ErrorActionPreference = 'Stop'
$root = 'D:/nips-temp/TotalP/P1'
$work = "$root/CRCICLR_LITEBN_OUTER_HELDOUT_WORK"
$code = "$work/experiments/persist_eeg_seven_backbone_litebn_seed0_outer_heldout_v1/code"
$python = 'E:/Anaconda/envs/persist_stable_251/python.exe'
$task = 'PERSIST_LITEBN_SEED0_OUTER_HELDOUT_FINALIZE_V1'
& $python -m py_compile "$code/litebn_seed0_outer_heldout.py"
if ($LASTEXITCODE -ne 0) { throw 'Python syntax check failed' }
$existing = Get-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue
if ($existing -and $existing.State -eq 'Running') { throw 'finalizer already running' }
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$code/finalize_litebn_seed0.ps1`"" -WorkingDirectory $code
$principal = New-ScheduledTaskPrincipal -UserId 'fyl412' -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $task -Action $action -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $task
Get-ScheduledTask -TaskName $task | Select-Object TaskName,State
