$ErrorActionPreference = 'Stop'
$name = 'PERSIST_EEG_PC_MECHANISM_CELL_EEGNET_MI_F1_ENDPOINT_NATIVE_V1'
$script = 'D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1\experiments\persist_eeg_pc_mechanism_closure_seed0_v1\code\run_eegnet_mi_fold1_endpoint_native_v1.ps1'
if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) { throw 'task already exists' }
if ((Get-FileHash -LiteralPath $script -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'e36762c0f4edc2adb32e8f848687ecede13cbc129e54857584ec831645ce14fa') { throw 'script SHA mismatch' }
if (@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(\.exe)?$' -and $_.CommandLine -and $_.CommandLine -match 'persist_eeg_pc_mechanism_closure_seed0_v1|pc_mechanism_closure_runtime' }).Count -ne 0) { throw 'mechanism process already running' }
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument ('-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' + $script + '"')
$principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 12) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings | Out-Null
Start-ScheduledTask -TaskName $name
Write-Output ('TASK ' + (Get-ScheduledTask -TaskName $name).State)
