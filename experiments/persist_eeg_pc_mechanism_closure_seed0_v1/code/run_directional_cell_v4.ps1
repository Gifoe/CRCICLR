<# Exact directional phase with per-draw checkpoints and fresh Python processes. #>
$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$exp=Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$entry=Join-Path $exp 'code\run_directional_cell_v4.py'
$core=Join-Path $exp 'code\directional_core_v3.py'
$factor=Join-Path $exp 'code\directional_factorized_eegnet_v4.py'
$sampling=Join-Path $exp 'code\sampling_core.py'
$cellData=Join-Path $exp 'code\cell_data_v1.py'
$cell=Join-Path $runtime 'cells\eegnet\openbmi_mi\fold0_seed0'
$parts=Join-Path $cell 'directional_v4'
$manifest=Join-Path $cell 'DIRECTIONAL_V4.json'
$failure=Join-Path $cell 'DIRECTIONAL_V4_FAIL_CLOSED.json'
$probe=Join-Path $exp 'probe\EEGNet_OpenBMI_MI_fold0_directional_factorized_probe_v7.json'
$log=Join-Path $runtime 'scheduled_directional_cell_EEGNet_OpenBMI_MI_f0_v4.log'
$env:PERSIST_SOURCE_REPO=$repo
$env:COUPLING_RUNTIME='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$env:MECHANISM_RUNTIME=$runtime
$env:PYTHONUNBUFFERED='1'
function ProgressCount {
  if(-not (Test-Path -LiteralPath $parts)){return 0}
  $count=0
  foreach($file in (Get-ChildItem -LiteralPath $parts -Filter '*.partial.json' -File)){
    $partial=Get-Content -Raw -LiteralPath $file.FullName | ConvertFrom-Json
    if($partial.status -ne 'PARTIAL_RESOURCE_SAFE' -or $partial.final_heldout_accessed -ne $false){throw 'invalid partial checkpoint'}
    $count+=@($partial.random_controls).Count
  }
  $count+=100 * @(Get-ChildItem -LiteralPath $parts -Filter '*.json' -File | Where-Object {$_.Name -notlike '*.partial.json'}).Count
  return $count
}
try {
  if((Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath $manifest) -or (Test-Path -LiteralPath $failure) -or (Test-Path -LiteralPath $parts)){throw 'directional v4 evidence exists'}
  if(-not (Test-Path -LiteralPath (Join-Path $cell 'MEDIATION_V2.json'))){throw 'validated mediation phase absent'}
  if(-not (Test-Path -LiteralPath $probe)){throw 'factorization probe absent'}
  $probeRecord=Get-Content -Raw -LiteralPath $probe | ConvertFrom-Json
  if($probeRecord.status -ne 'ENGINEERING_FACTORIZATION_EQUIVALENT' -or $probeRecord.cached_max_abs_metric_difference -ge 1e-5 -or $probeRecord.final_heldout_accessed -ne $false){throw 'factorization probe invalid'}
  $expected=@{
    $entry='7c77da55f753d372e394c44e9d851e4f2390a42ebc549993a69e176e7ce858bc';
    $core='5f50f0f649039f2cb83c2b951952e04a645908bf5328bfc277b79948e05608af';
    $factor='03c1f0262885546ea70ebfc7db93471968f9952e2aac62097d13b4317ff91729';
    $sampling='7b8f5b94caca3924de3c714a780987a52337359963f38d5135c78410c8b1a631';
    $cellData='c5e4f90865445bcc1ec313b1ca5b1970cf484a9ef7f44e8aa1bf38a0fd721416'
  }
  foreach($path in $expected.Keys){if((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected[$path]){throw "source SHA mismatch: $path"}}
  if((Get-FileHash -LiteralPath (Join-Path $exp 'protocol\MECHANISM_CLOSURE_ANALYSIS_LOCK.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'ac12d33f7d30b2c846ec4c4fb07f8513bf25e372943a767b134883bf095a7c88'){throw 'analysis lock SHA mismatch'}
  if((Get-FileHash -LiteralPath (Join-Path $exp 'protocol\FINAL_PROJECTOR_AMENDMENT.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'aec1e604d38f1901bd9eca7e072e30a3d496a28061a726ebbe63133259b0e474'){throw 'projector amendment SHA mismatch'}
  $gate=Get-Content -Raw -LiteralPath (Join-Path $exp 'gate\GATE_AUDIT.json') | ConvertFrom-Json
  if($gate.status -ne 'PASSED'){throw 'upstream gate not passed'}
  "DIRECTIONAL_CELL_V4_START utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Encoding utf8
  $python='E:\Anaconda\envs\persist_stable_251\python.exe'
  for($attempt=1;$attempt -le 80;$attempt++){
    $os=Get-CimInstance Win32_OperatingSystem
    $free=$os.FreePhysicalMemory/1MB
    $gpu=[int]([string]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)|Select-Object -First 1)).Trim()
    if($free -lt 40 -or $gpu -lt 16000){throw "resource gate before process $attempt free_gb=$free gpu_mib=$gpu"}
    $before=ProgressCount
    "DIRECTIONAL_CELL_V4_PROCESS_START attempt=$attempt progress=$before free_gb=$([math]::Round($free,2)) gpu_mib=$gpu utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Append -Encoding utf8
    $command='"'+$python+'" -u "'+$entry+'" --model EEGNet --task OpenBMI_MI --fold 0 >> "'+$log+'" 2>&1'
    & cmd.exe /d /c $command
    if($LASTEXITCODE -ne 0){throw "directional v4 process $attempt exited $LASTEXITCODE"}
    if(Test-Path -LiteralPath $manifest){
      $done=Get-Content -Raw -LiteralPath $manifest | ConvertFrom-Json
      if($done.status -ne 'DIRECTIONAL_PHASE_COMPLETE_PENDING_OTHER_REQUIRED_ANALYSES' -or $done.stage_count -ne 5 -or $done.final_heldout_accessed -ne $false){throw 'directional manifest invalid'}
      "DIRECTIONAL_CELL_V4_END exit=0 attempts=$attempt utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Append -Encoding utf8
      exit 0
    }
    $after=ProgressCount
    if($after -le $before){throw "no checkpoint progress in process $attempt"}
    "DIRECTIONAL_CELL_V4_PROCESS_RECYCLED attempt=$attempt progress=$after utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -FilePath $log -Append -Encoding utf8
    Start-Sleep -Seconds 2
  }
  throw 'directional v4 exceeded maximum process attempts'
} catch {
  "DIRECTIONAL_CELL_V4_EXCEPTION $_" | Out-File -FilePath $log -Append -Encoding utf8
  exit 1
}
