$ErrorActionPreference = 'Stop'
$root = 'D:/nips-temp/TotalP/P1'
$work = "$root/CRCICLR_LITEBN_OUTER_HELDOUT_WORK"
$code = "$work/experiments/persist_eeg_seven_backbone_litebn_seed0_outer_heldout_v1/code"
$python = 'E:/Anaconda/envs/persist_stable_251/python.exe'
$task = 'PERSIST_LITEBN_WBCIC_SEED0_COMPLETE_V1'
& $python -m py_compile "$code/run_litebn_wbcic_seed0.py" "$code/litebn_seed0_outer_heldout.py"
if ($LASTEXITCODE -ne 0) { throw 'Python syntax check failed' }
$active = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'run_litebn_wbcic_seed0.py' }
if ($active) { throw 'LiteBN WBCIC supervisor already exists' }
$registered = Get-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue
if ($registered -and $registered.State -eq 'Running') { throw 'Scheduled LiteBN completion already running' }
$action = New-ScheduledTaskAction -Execute $python -Argument "-u `"$code/run_litebn_wbcic_seed0.py`"" -WorkingDirectory $code
$principal = New-ScheduledTaskPrincipal -UserId 'fyl412' -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $task -Action $action -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $task
Get-ScheduledTask -TaskName $task | Select-Object TaskName,State
