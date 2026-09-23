<# Recover 20 fixed TRAIN random-subject controls; one fresh Python per draw. #>
$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$exp=Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$cell=Join-Path $runtime 'cells\eegnet\openbmi_mi\fold0_seed0'
$entry=Join-Path $exp 'code\audit_train_random_subject_eegnet_mi_f0_v1.py'
$log=Join-Path $runtime 'scheduled_train_random_subject_EEGNet_OpenBMI_MI_f0_v1.log'
$python='E:\Anaconda\envs\persist_stable_251\python.exe'
$env:PERSIST_SOURCE_REPO=$repo
$env:MECHANISM_RUNTIME=$runtime
$env:COUPLING_RUNTIME='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$env:SEVEN_RUNTIME='D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime'
$env:OFFICIAL_BACKBONE_ROOT=Join-Path $env:SEVEN_RUNTIME 'official'
$env:PYTHONUNBUFFERED='1'
try {
  if((Get-FileHash -LiteralPath $entry -Algorithm SHA256).Hash.ToLowerInvariant() -ne '36d478fd0bd220b3862e2ee871c6c314722f03217df0521ade8ce0154c2423bc') {throw 'TRAIN random-subject source SHA mismatch'}
  if((Get-FileHash -LiteralPath (Join-Path $exp 'protocol\MECHANISM_CLOSURE_ANALYSIS_LOCK.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'ac12d33f7d30b2c846ec4c4fb07f8513bf25e372943a767b134883bf095a7c88') {throw 'analysis lock SHA mismatch'}
  if((Get-FileHash -LiteralPath (Join-Path $exp 'protocol\FINAL_PROJECTOR_AMENDMENT.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'aec1e604d38f1901bd9eca7e072e30a3d496a28061a726ebbe63133259b0e474') {throw 'final-projector amendment SHA mismatch'}
  "TRAIN_RANDOM_SUBJECT_V1_START utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $log -Append -Encoding utf8
  for($draw=0;$draw -lt 20;$draw++) {
    $result=Join-Path $cell ('train_random_subject_v1\draw_{0:d2}.json' -f $draw)
    $failure=Join-Path $cell ('train_random_subject_v1\draw_{0:d2}.FAIL_CLOSED.json' -f $draw)
    if(Test-Path -LiteralPath $failure) {throw "FAIL_CLOSED evidence already exists for draw $draw"}
    if(Test-Path -LiteralPath $result) {
      $row=Get-Content -LiteralPath $result -Raw|ConvertFrom-Json
      if($row.status -ne 'TRAIN_RANDOM_SUBJECT_DRAW_COMPLETE' -or $row.draw -ne $draw -or $row.implementation_sha256 -ne '36d478fd0bd220b3862e2ee871c6c314722f03217df0521ade8ce0154c2423bc' -or $row.final_heldout_accessed -ne $false -or @($row.train_subject_session_rows).Count -ne 52) {throw "existing draw $draw invalid"}
      "TRAIN_RANDOM_SUBJECT_V1_REUSED draw=$draw sha=$((Get-FileHash -LiteralPath $result -Algorithm SHA256).Hash.ToLowerInvariant())" | Out-File -LiteralPath $log -Append -Encoding utf8
      continue
    }
    $ram=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
    if($ram -lt 40) {throw "resource safety: RAM $ram GB < 40 GB at draw $draw launch"}
    $gpuFree=[int]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | Select-Object -First 1).Trim())
    if($gpuFree -lt 16000) {throw "resource safety: GPU $gpuFree MiB < 16000 MiB at draw $draw launch"}
    "TRAIN_RANDOM_SUBJECT_V1_LAUNCH draw=$draw ram_gb=$ram gpu_free_mib=$gpuFree utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $log -Append -Encoding utf8
    $command='"'+$python+'" -u "'+$entry+'" --draw '+$draw+' >> "'+$log+'" 2>&1'
    & cmd.exe /d /c $command
    if($LASTEXITCODE -ne 0) {throw "TRAIN random-subject draw $draw exited $LASTEXITCODE"}
    $row=Get-Content -LiteralPath $result -Raw|ConvertFrom-Json
    if($row.status -ne 'TRAIN_RANDOM_SUBJECT_DRAW_COMPLETE' -or $row.draw -ne $draw -or $row.final_heldout_accessed -ne $false -or @($row.train_subject_session_rows).Count -ne 52) {throw "TRAIN random-subject draw $draw invalid"}
    "TRAIN_RANDOM_SUBJECT_V1_COMPLETE draw=$draw sha=$((Get-FileHash -LiteralPath $result -Algorithm SHA256).Hash.ToLowerInvariant()) utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $log -Append -Encoding utf8
  }
  "TRAIN_RANDOM_SUBJECT_V1_ALL20_COMPLETE utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $log -Append -Encoding utf8
  exit 0
} catch {
  "TRAIN_RANDOM_SUBJECT_V1_EXCEPTION $_ utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $log -Append -Encoding utf8
  exit 1
}
