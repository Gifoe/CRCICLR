$ErrorActionPreference='Stop'
$runtime='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$cell=Join-Path $runtime 'cells\eegnet\openbmi_mi\fold4_seed0'
$targetTask='PERSIST_EEG_PC_MECHANISM_CELL_EEGNET_MI_F4_DIRECTIONAL_QUEUE_V4'
$auditLog=Join-Path $runtime 'scheduled_directional_cell_EEGNet_OpenBMI_MI_f4_resume_after_reboot_v1.log'
$suffix='.interrupted_before_reboot_20260923_2302'

try {
  if(Test-Path -LiteralPath $auditLog){throw 'resume audit log already exists'}
  if((Test-Path -LiteralPath (Join-Path $cell 'DIRECTIONAL_V4.json')) -or
     (Test-Path -LiteralPath (Join-Path $cell 'DIRECTIONAL_V4_FAIL_CLOSED.json'))){throw 'terminal directional evidence exists'}
  $partial=Join-Path $cell 'directional_v4\temporal_bn.partial.json'
  if(-not (Test-Path -LiteralPath $partial)){throw 'directional partial missing'}
  $p=Get-Content -LiteralPath $partial -Raw|ConvertFrom-Json
  if($p.status -ne 'PARTIAL_RESOURCE_SAFE' -or @($p.random_controls).Count -ne 6 -or $p.final_heldout_accessed -ne $false){throw 'directional partial invalid'}
  $t=Get-ScheduledTask -TaskName $targetTask
  if($t.State -ne 'Ready'){throw "target task state is $($t.State)"}

  $files=Get-ChildItem -LiteralPath $runtime -File | Where-Object {
    $_.Name -like 'scheduled_directional_cell_EEGNet_OpenBMI_MI_f4_queue_v4*' -and
    $_.Name -notlike "*$suffix"
  }
  if($files.Count -ne 7){throw "unexpected directional log count $($files.Count)"}
  foreach($f in $files){
    $dst=$f.FullName+$suffix
    if(Test-Path -LiteralPath $dst){throw "archive target exists: $dst"}
  }
  foreach($f in $files){Move-Item -LiteralPath $f.FullName -Destination ($f.FullName+$suffix) -ErrorAction Stop}
  "RESUME_F4_DIRECTIONAL archived=$($files.Count) controls=6 boot=$((Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString('o')) utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $auditLog -Encoding utf8
  Start-ScheduledTask -TaskName $targetTask
  Start-Sleep -Seconds 5
  $after=Get-ScheduledTask -TaskName $targetTask
  "RESUME_F4_DIRECTIONAL target_state=$($after.State) utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $auditLog -Append -Encoding utf8
  if($after.State -ne 'Running'){throw "target task failed to start: $($after.State)"}
  exit 0
} catch {
  if(-not (Test-Path -LiteralPath $auditLog)){"RESUME_F4_DIRECTIONAL_EXCEPTION $_ utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $auditLog -Encoding utf8}
  else{"RESUME_F4_DIRECTIONAL_EXCEPTION $_ utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $auditLog -Append -Encoding utf8}
  exit 1
}
