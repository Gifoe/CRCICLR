$ErrorActionPreference='Stop'
$runtime='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$cell=Join-Path $runtime 'cells\eegnet\openbmi_mi\fold4_seed0'
$auditLog=Join-Path $runtime 'scheduled_fold4_waiters_resume_after_reboot_v1.log'
$items=@(
  @{Name='PERSIST_EEG_PC_PIPELINE_EEGNET_MI_F4_REPLAY_V2';Output=(Join-Path $cell 'replay_v2\REPLAY_AUDIT.json')},
  @{Name='PERSIST_EEG_PC_PIPELINE_EEGNET_MI_F4_ENDPOINT_NATIVE_V4';Output=(Join-Path $cell 'REPLICA_ENDPOINT_AUDIT_V1.json')}
)
try {
  if(Test-Path -LiteralPath $auditLog){throw 'waiter resume audit exists'}
  foreach($x in $items){
    $t=Get-ScheduledTask -TaskName $x.Name
    if($t.State -ne 'Ready'){throw "$($x.Name) state $($t.State)"}
    if(Test-Path -LiteralPath $x.Output){throw "$($x.Name) terminal output exists"}
  }
  foreach($x in $items){Start-ScheduledTask -TaskName $x.Name}
  Start-Sleep -Seconds 5
  foreach($x in $items){
    $t=Get-ScheduledTask -TaskName $x.Name
    "$($x.Name) state=$($t.State)" | Out-File -LiteralPath $auditLog -Append -Encoding utf8
    if($t.State -ne 'Running'){throw "$($x.Name) did not restart"}
  }
  exit 0
} catch {
  "WAITER_RESUME_EXCEPTION $_ utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $auditLog -Append -Encoding utf8
  exit 1
}
