<# Wait for checkpoint mechanisms, then finish subject/session evidence and compact cell output. #>
$ErrorActionPreference = 'Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$experiment=Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$cell=Join-Path $runtime 'cells\eegnet\openbmi_mi\fold3_seed0'
$subjectEntry=Join-Path $experiment 'code\audit_subject_session_eegnet_v2.py'
$randomEntry=Join-Path $experiment 'code\audit_train_random_subject_eegnet_v2.py'
$completeEntry=Join-Path $experiment 'code\complete_subject_session_eegnet_v3.py'
$compactEntry=Join-Path $experiment 'code\build_cell_compact_preview_eegnet_v3.py'
$mainLog=Join-Path $runtime 'scheduled_subject_compact_EEGNet_OpenBMI_MI_f3_v2.log'
$env:PERSIST_SOURCE_REPO=$repo
$env:MECHANISM_RUNTIME=$runtime
$env:COUPLING_RUNTIME='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$env:SEVEN_RUNTIME='D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime'
$env:OFFICIAL_BACKBONE_ROOT=Join-Path $env:SEVEN_RUNTIME 'official'
$env:PYTHONUNBUFFERED='1'
$python='E:\Anaconda\envs\persist_stable_251\python.exe'

function Wait-Resources {
  param([string]$Phase)
  while($true){
    $freeGb=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
    $gpuMiB=[int](([string]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)|Select-Object -First 1)).Trim())
    if($freeGb -ge 40 -and $gpuMiB -ge 16000){"$Phase RESOURCE_READY free_gb=$([Math]::Round($freeGb,2)) gpu_mib=$gpuMiB utc=$([DateTime]::UtcNow.ToString('o'))"|Out-File $mainLog -Append -Encoding utf8; return}
    "$Phase RESOURCE_WAIT free_gb=$([Math]::Round($freeGb,2)) gpu_mib=$gpuMiB utc=$([DateTime]::UtcNow.ToString('o'))"|Out-File $mainLog -Append -Encoding utf8
    Start-Sleep -Seconds 30
  }
}

function Run-Python {
  param([string]$Phase,[string]$Arguments,[string]$LogName)
  Wait-Resources $Phase
  $phaseLog=Join-Path $runtime $LogName
  if(Test-Path $phaseLog){throw "$Phase log already exists"}
  $cmd='"'+$python+'" -u '+$Arguments+' >> "'+$phaseLog+'" 2>&1'
  & cmd.exe /d /c $cmd
  if($LASTEXITCODE -ne 0){throw "$Phase exited $LASTEXITCODE"}
}

try {
  if(Test-Path $mainLog){throw 'pipeline log already exists'}
  $expected=@{
    $subjectEntry='463861faddff7b0c72e473eacbc7f3d8fb47c19b720e1d0f734e725207bf167d'
    $randomEntry='d7d055dd5a03c17f07d5ae58ff436ff44af74a837e1e9e2f68162b2c481ef986'
    $completeEntry='546e30a39bbce20b31d0f62179b277d43cb6a2983a2abdb0486dd7838950b49b'
    $compactEntry='c272f4caa80c68ce9b3e5dc44b5f50451aa417d26fbe9a7d35e2e66cea224616'
  }
  foreach($entry in $expected.Keys){if((Get-FileHash $entry -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected[$entry]){throw "source SHA mismatch: $entry"}}
  if((Get-FileHash (Join-Path $experiment 'protocol\MECHANISM_CLOSURE_ANALYSIS_LOCK.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'ac12d33f7d30b2c846ec4c4fb07f8513bf25e372943a767b134883bf095a7c88'){throw 'analysis lock SHA mismatch'}
  if((Get-FileHash (Join-Path $experiment 'protocol\FINAL_PROJECTOR_AMENDMENT.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'aec1e604d38f1901bd9eca7e072e30a3d496a28061a726ebbe63133259b0e474'){throw 'amendment SHA mismatch'}
  "SUBJECT_COMPACT_F3_START utc=$([DateTime]::UtcNow.ToString('o'))"|Out-File $mainLog -Encoding utf8
  $endpointPath=Join-Path $cell 'REPLICA_ENDPOINT_AUDIT_V1.json'
  while(-not (Test-Path $endpointPath)){Start-Sleep -Seconds 15}
  $endpoint=Get-Content $endpointPath -Raw|ConvertFrom-Json
  $epochs=@($endpoint.schedule|ForEach-Object{[int]$_.epoch})
  while($true){
    $valid=0
    foreach($epoch in $epochs){$p=Join-Path $cell ("checkpoint_mechanism_v1\epoch_{0:d3}.json" -f $epoch); if(Test-Path $p){$j=Get-Content $p -Raw|ConvertFrom-Json; if($j.status -eq 'CHECKPOINT_MECHANISM_COMPLETE_REPLICA_ONLY' -and $j.final_heldout_accessed -eq $false){$valid++}}}
    if($valid -eq 6){break}
    "WAIT_CHECKPOINT_MECHANISM valid=$valid/6 utc=$([DateTime]::UtcNow.ToString('o'))"|Out-File $mainLog -Append -Encoding utf8
    Start-Sleep -Seconds 20
  }
  $subjectV1=Join-Path $cell 'SUBJECT_SESSION_AUDIT_V1.json'
  if(-not (Test-Path $subjectV1)){
    if(Test-Path (Join-Path $cell 'SUBJECT_SESSION_AUDIT_V1.FAIL_CLOSED.json')){throw 'subject V1 failed closed'}
    Run-Python 'SUBJECT_V1' ('"'+$subjectEntry+'" --task OpenBMI_MI --fold 3') 'scheduled_subject_session_EEGNet_OpenBMI_MI_f3_v2.log'
  }
  $sv1=Get-Content $subjectV1 -Raw|ConvertFrom-Json
  if($sv1.status -ne 'SUBJECT_SESSION_AUDIT_COMPLETE_POST_OUTCOME_DESCRIPTIVE' -or $sv1.final_heldout_accessed -ne $false -or @($sv1.subject_session_signature_rows).Count -ne 68){throw 'subject V1 invalid'}
  $baseSha=(Get-FileHash $subjectV1 -Algorithm SHA256).Hash.ToLowerInvariant()
  "SUBJECT_V1_COMPLETE sha=$baseSha utc=$([DateTime]::UtcNow.ToString('o'))"|Out-File $mainLog -Append -Encoding utf8
  foreach($draw in 0..19){
    $tag='{0:d2}' -f $draw
    $out=Join-Path $cell "train_random_subject_v1\draw_$tag.json"
    if(-not (Test-Path $out)){
      if(Test-Path (Join-Path $cell "train_random_subject_v1\draw_$tag.FAIL_CLOSED.json")){throw "random draw $draw failed closed"}
      Run-Python "TRAIN_RANDOM_D$tag" ('"'+$randomEntry+'" --task OpenBMI_MI --fold 3 --draw '+$draw) "scheduled_train_random_subject_EEGNet_OpenBMI_MI_f3_d${tag}_v2.log"
    }
    $j=Get-Content $out -Raw|ConvertFrom-Json
    if($j.status -ne 'TRAIN_RANDOM_SUBJECT_DRAW_COMPLETE' -or [int]$j.draw -ne $draw -or $j.final_heldout_accessed -ne $false -or @($j.train_subject_session_rows).Count -ne 52){throw "random draw $draw invalid"}
    "TRAIN_RANDOM_D${tag}_COMPLETE utc=$([DateTime]::UtcNow.ToString('o'))"|Out-File $mainLog -Append -Encoding utf8
  }
  $subjectV2=Join-Path $cell 'SUBJECT_SESSION_AUDIT_V2.json'
  if(-not (Test-Path $subjectV2)){
    if(Test-Path (Join-Path $cell 'SUBJECT_SESSION_AUDIT_V2.FAIL_CLOSED.json')){throw 'subject V2 failed closed'}
    Run-Python 'SUBJECT_V2' ('"'+$completeEntry+'" --task OpenBMI_MI --fold 3 --base-sha '+$baseSha) 'scheduled_subject_session_complete_EEGNet_OpenBMI_MI_f3_v3.log'
  }
  $sv2=Get-Content $subjectV2 -Raw|ConvertFrom-Json
  if($sv2.status -ne 'SUBJECT_SESSION_AUDIT_COMPLETE_POST_OUTCOME_DESCRIPTIVE' -or $sv2.final_heldout_accessed -ne $false -or @($sv2.subject_session_signature_rows).Count -ne 68){throw 'subject V2 invalid'}
  $subjectV2Sha=(Get-FileHash $subjectV2 -Algorithm SHA256).Hash.ToLowerInvariant()
  "SUBJECT_V2_COMPLETE sha=$subjectV2Sha utc=$([DateTime]::UtcNow.ToString('o'))"|Out-File $mainLog -Append -Encoding utf8
  $compact=Join-Path $runtime 'compact_preview\eegnet_openbmi_mi_fold3_seed0_v3\PROVENANCE.json'
  if(-not (Test-Path $compact)){
    Run-Python 'COMPACT' ('"'+$compactEntry+'" --task OpenBMI_MI --fold 3 --subject-v2-sha '+$subjectV2Sha) 'scheduled_compact_EEGNet_OpenBMI_MI_f3_v3.log'
  }
  $cp=Get-Content $compact -Raw|ConvertFrom-Json
  if($cp.status -ne 'CELL_COMPACT_PREVIEW_COMPLETE_NOT_20_CELL_RESULT' -or $cp.final_heldout_accessed -ne $false){throw 'compact output invalid'}
  "SUBJECT_COMPACT_F3_COMPLETE utc=$([DateTime]::UtcNow.ToString('o'))"|Out-File $mainLog -Append -Encoding utf8
  exit 0
} catch {
  if(Test-Path $mainLog){"SUBJECT_COMPACT_F3_EXCEPTION $_ utc=$([DateTime]::UtcNow.ToString('o'))"|Out-File $mainLog -Append -Encoding utf8}
  exit 1
}
