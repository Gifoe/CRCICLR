$ErrorActionPreference='Stop'
$repo='D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$exp=Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1'
$runtime='D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$cell=Join-Path $runtime 'cells\eegnet\openbmi_ssvep\fold4_seed0'
$directional=Join-Path $cell 'DIRECTIONAL_V4.json'
$stageDir=Join-Path $cell 'directional_v4'
$replayDir=Join-Path $cell 'replay_v2'
$replayAudit=Join-Path $replayDir 'REPLAY_AUDIT.json'
$entry=Join-Path $exp 'code\replay_eegnet_v2.py'
$log=Join-Path $runtime 'scheduled_replay_EEGNet_OpenBMI_SSVEP_f4_v4.log'
$python='E:\Anaconda\envs\persist_stable_251\python.exe'
$gatePath=Join-Path $exp 'gate\GATE_AUDIT.json'
$lockPath=Join-Path $exp 'protocol\MECHANISM_CLOSURE_ANALYSIS_LOCK.md'
$amendmentPath=Join-Path $exp 'protocol\FINAL_PROJECTOR_AMENDMENT.md'
$expectedEntrySha='944dc5ad03dc536870bd3338d5ee55069149e14b05a466ef175f75b59c18cc3a'
$expectedLockSha='ac12d33f7d30b2c846ec4c4fb07f8513bf25e372943a767b134883bf095a7c88'
$expectedAmendmentSha='aec1e604d38f1901bd9eca7e072e30a3d496a28061a726ebbe63133259b0e474'
$expectedProtocolSha='3ddb5006bf7159b00a0c80fed31856849ccdaa69ecf5060b0161c8673327e15c'
$expectedImplementationSha='4aab984969f4ba43128ed738a38ac138d94ee1c9b1252662af91b0c44e3f8eee'
$env:PERSIST_SOURCE_REPO=$repo
$env:COUPLING_RUNTIME='D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$env:MECHANISM_RUNTIME=$runtime
$env:SEVEN_RUNTIME='D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime'
$env:OFFICIAL_BACKBONE_ROOT=Join-Path $env:SEVEN_RUNTIME 'official'
$env:PYTHONUNBUFFERED='1'

function Log([string]$m){$m|Out-File -LiteralPath $log -Append -Encoding utf8}
function Resources {
  $ram=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
  $s=[string]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)|Select-Object -First 1)
  $gpu=0;if(-not[int]::TryParse($s.Trim(),[ref]$gpu)){throw 'GPU query failed'}
  return @{RAM=$ram;GPU=$gpu}
}
function Validate-Directional {
  if(-not(Test-Path -LiteralPath $directional)){throw 'directional manifest absent'}
  $d=Get-Content -LiteralPath $directional -Raw|ConvertFrom-Json
  $gateSha=(Get-FileHash -LiteralPath $gatePath -Algorithm SHA256).Hash.ToLowerInvariant()
  if($d.status -ne 'DIRECTIONAL_PHASE_COMPLETE_PENDING_OTHER_REQUIRED_ANALYSES' -or $d.model -ne 'EEGNet' -or
     $d.task -ne 'OpenBMI_SSVEP' -or [int]$d.fold -ne 4 -or [int]$d.stage_count -ne 5 -or
     [int]$d.random_draws_per_stage -ne 20 -or $d.final_heldout_accessed -ne $false -or
     $d.upstream_gate_sha256 -ne $gateSha -or $d.upstream_protocol_sha256 -ne $expectedProtocolSha -or
     $d.upstream_implementation_sha256 -ne $expectedImplementationSha){throw 'directional identity/status invalid'}
  $required=@('temporal_bn','spatial_elu_pool1','depth_point_elu_pool2','embedding_64d','classifier_input')
  foreach($stage in $required){
    $f=Join-Path $stageDir "$stage.json";if(-not(Test-Path -LiteralPath $f)){throw "missing stage $stage"}
    if((Get-FileHash -LiteralPath $f -Algorithm SHA256).Hash.ToLowerInvariant() -ne $d.stage_sha256.$stage){throw "stage hash mismatch $stage"}
    $s=Get-Content -LiteralPath $f -Raw|ConvertFrom-Json
    if($s.status -ne 'STAGE_COMPLETE' -or @($s.random_controls).Count -ne 20 -or $s.final_heldout_accessed -ne $false -or
       $s.analysis_lock_sha256 -ne $expectedLockSha -or $s.projector_amendment_sha256 -ne $expectedAmendmentSha -or
       [double]$s.full_train_equivalence_max_abs -ge 0.00001){throw "stage content invalid $stage"}
  }
}

try {
  if(Test-Path -LiteralPath $log){throw 'replay V4 log exists; refusing duplicate'}
  if(Test-Path -LiteralPath $replayAudit){throw 'replay audit exists; refusing duplicate'}
  if(Test-Path -LiteralPath $replayDir){throw 'replay_v2 directory exists without audit; manual review required'}
  "REPLAY_SSVEP_F4_V4_START utc=$([DateTime]::UtcNow.ToString('o'))"|Out-File -LiteralPath $log -Encoding utf8
  if((Get-FileHash $entry -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedEntrySha){throw 'replay source SHA mismatch'}
  if((Get-FileHash $lockPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedLockSha){throw 'analysis lock SHA mismatch'}
  if((Get-FileHash $amendmentPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedAmendmentSha){throw 'amendment SHA mismatch'}
  $gate=Get-Content $gatePath -Raw|ConvertFrom-Json
  if($gate.status -ne 'PASSED' -or [int]$gate.original_cells_complete -ne 20 -or $gate.final_heldout_accessed -ne $false){throw 'strict gate invalid'}
  Validate-Directional
  while($true){
    $r=Resources
    $heavy=@(Get-CimInstance Win32_Process|Where-Object{$_.Name -match '^python(\.exe)?$' -and $_.CommandLine -and $_.CommandLine -match 'persist_eeg_pc_mechanism_closure_seed0_v1|pc_mechanism_closure_runtime'})
    if($r.RAM -ge 40 -and $r.GPU -ge 16000 -and $heavy.Count -eq 0){break}
    Log "WAIT_RESOURCE ram_gb=$([math]::Round($r.RAM,2)) gpu_mib=$($r.GPU) phase2_python=$($heavy.Count) utc=$([DateTime]::UtcNow.ToString('o'))"
    Start-Sleep -Seconds 20
  }
  Validate-Directional
  Log "REPLAY_SSVEP_F4_V4_LAUNCH ram_gb=$([math]::Round($r.RAM,2)) gpu_mib=$($r.GPU) directional_sha=$((Get-FileHash $directional -Algorithm SHA256).Hash.ToLowerInvariant()) utc=$([DateTime]::UtcNow.ToString('o'))"
  $cmd='"'+$python+'" -u "'+$entry+'" --task OpenBMI_SSVEP --fold 4 >> "'+$log+'" 2>&1'
  & cmd.exe /d /c $cmd
  if($LASTEXITCODE -ne 0){throw "replay exited $LASTEXITCODE"}
  $result=Get-Content $replayAudit -Raw|ConvertFrom-Json
  if($result.status -ne 'REPLAY_COMPLETE' -or $result.model -ne 'EEGNet' -or $result.task -ne 'OpenBMI_SSVEP' -or
     [int]$result.fold -ne 4 -or [int]$result.epochs_saved -ne 60 -or $result.final_heldout_accessed -ne $false -or
     $result.trajectory_provenance -ne 'RETRAINED_REPLICA_TRAJECTORY'){throw 'replay output invalid'}
  Log "REPLAY_SSVEP_F4_V4_COMPLETE utc=$([DateTime]::UtcNow.ToString('o')) replay_sha=$((Get-FileHash $replayAudit -Algorithm SHA256).Hash.ToLowerInvariant())"
  exit 0
} catch {if(Test-Path $log){Log "REPLAY_SSVEP_F4_V4_EXCEPTION $($_.Exception.Message) utc=$([DateTime]::UtcNow.ToString('o'))"};exit 1}
