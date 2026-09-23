<# One-shot, fail-closed registration of the fold1 resource-monitored resume task. #>
$ErrorActionPreference = 'Stop'
$name = 'PERSIST_EEG_PC_MECHANISM_CELL_EEGNET_MI_F1_MEDIATION_RESUME_V2'
$script = 'D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1\experiments\persist_eeg_pc_mechanism_closure_seed0_v1\code\run_eegnet_mi_fold1_mediation_resume_v2.ps1'
$runtime = 'D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) { throw 'task already exists' }
if (Test-Path -LiteralPath (Join-Path $runtime 'scheduled_mediation_cell_EEGNet_OpenBMI_MI_f1_resume_v2.log')) {
  throw 'resume log already exists'
}
if (@(Get-ScheduledTask | Where-Object {
  $_.TaskName -like 'PERSIST_EEG_PC_MECHANISM_CELL_*' -and $_.State -eq 'Running'
}).Count -ne 0) { throw 'another mechanism cell task is running' }
if ((Get-FileHash -LiteralPath $script -Algorithm SHA256).Hash.ToLowerInvariant() -ne
    'b977263dfa477ff821d4713ae011e1ed098761d049675ba4faf4e75f2a72c771') {
  throw 'resume script SHA mismatch'
}
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument ('-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' + $script + '"')
$principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 26) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings | Out-Null
Start-ScheduledTask -TaskName $name
Write-Output ('TASK ' + (Get-ScheduledTask -TaskName $name).State)
