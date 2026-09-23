$ErrorActionPreference = 'Stop'
$name = 'PERSIST_EEG_PC_MECHANISM_CELL_EEGNET_MI_F1_DIRECTIONAL_QUEUE_V3'
$script = 'D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1\experiments\persist_eeg_pc_mechanism_closure_seed0_v1\code\run_eegnet_mi_fold1_directional_queue_v3.ps1'
$runtime = 'D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) { throw 'task already exists' }
if (Test-Path -LiteralPath (Join-Path $runtime 'scheduled_directional_cell_EEGNet_OpenBMI_MI_f1_queue_v3.log')) { throw 'log already exists' }
if ((Get-FileHash -LiteralPath $script -Algorithm SHA256).Hash.ToLowerInvariant() -ne
    '3ca49cbc31c663015ccc2ee1d9cffa38c099b89e2ab2f8ee7141f0b095a89ba1') { throw 'script SHA mismatch' }
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument ('-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' + $script + '"')
$principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 48) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings | Out-Null
Start-ScheduledTask -TaskName $name
Write-Output ('TASK ' + (Get-ScheduledTask -TaskName $name).State)
