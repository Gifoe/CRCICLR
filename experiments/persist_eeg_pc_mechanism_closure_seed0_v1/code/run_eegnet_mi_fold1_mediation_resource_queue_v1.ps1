<# Wait for the frozen resource gate, then run exactly one EEGNet MI fold1 mediation phase. #>
$ErrorActionPreference = 'Stop'
$repo = 'D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$experiment = Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1'
$runtime = 'D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$couplingRuntime = 'D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$taskName = 'PERSIST_EEG_PC_MECHANISM_CELL_EEGNET_MI_F1_MEDIATION_QUEUE_V1'
$entry = Join-Path $experiment 'code\run_mediation_cell_v2.py'
$sampling = Join-Path $experiment 'code\sampling_core.py'
$cellData = Join-Path $experiment 'code\cell_data_v1.py'
$mediation = Join-Path $experiment 'code\mediation_core_v5.py'
$log = Join-Path $runtime 'scheduled_mediation_cell_EEGNet_OpenBMI_MI_f1_queue_v1.log'
$cell = Join-Path $runtime 'cells\eegnet\openbmi_mi\fold1_seed0'
$manifest = Join-Path $cell 'MEDIATION_V2.json'
$failure = Join-Path $cell 'MEDIATION_V2_FAIL_CLOSED.json'
$env:PERSIST_SOURCE_REPO = $repo
$env:COUPLING_RUNTIME = $couplingRuntime
$env:MECHANISM_RUNTIME = $runtime
$env:PYTHONUNBUFFERED = '1'
$python = 'E:\Anaconda\envs\persist_stable_251\python.exe'

function Confirm-FrozenInputs {
  $expected = @{
    $entry = 'b00479f7cdf6e3ea5b118773e6d3382e6fb29e9b39beb58b4842df10a54e3175';
    $sampling = '7b8f5b94caca3924de3c714a780987a52337359963f38d5135c78410c8b1a631';
    $cellData = 'c5e4f90865445bcc1ec313b1ca5b1970cf484a9ef7f44e8aa1bf38a0fd721416';
    $mediation = '22853e842a9dfcb5457c9eb96a84b738b4a2a47054b5778dd4e830bca6c27846'
  }
  foreach ($path in $expected.Keys) {
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected[$path]) {
      throw "source SHA mismatch: $path"
    }
  }
  $locks = @{
    (Join-Path $experiment 'protocol\MECHANISM_CLOSURE_ANALYSIS_LOCK.md') = 'ac12d33f7d30b2c846ec4c4fb07f8513bf25e372943a767b134883bf095a7c88';
    (Join-Path $experiment 'protocol\FINAL_PROJECTOR_AMENDMENT.md') = 'aec1e604d38f1901bd9eca7e072e30a3d496a28061a726ebbe63133259b0e474'
  }
  foreach ($path in $locks.Keys) {
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $locks[$path]) {
      throw "protocol SHA mismatch: $path"
    }
  }
  $gate = Get-Content -LiteralPath (Join-Path $experiment 'gate\GATE_AUDIT.json') -Raw | ConvertFrom-Json
  if ($gate.status -ne 'PASSED' -or $gate.original_cells_complete -ne 20 -or
      $gate.layer_stage_cells_reconstructed -ne 120 -or $gate.final_heldout_accessed -ne $false) {
    throw 'upstream gate invalid'
  }
  $frozenCell = Join-Path $couplingRuntime 'cells\EEGNet\OpenBMI_MI\fold1_seed0.json'
  $frozen = Get-Content -LiteralPath $frozenCell -Raw | ConvertFrom-Json
  if ($frozen.status -ne 'COMPLETE' -or $frozen.final_heldout_accessed -ne $false -or
      $frozen.protocol_sha256 -ne '3ddb5006bf7159b00a0c80fed31856849ccdaa69ecf5060b0161c8673327e15c' -or
      $frozen.implementation_sha256 -ne '4aab984969f4ba43128ed738a38ac138d94ee1c9b1252662af91b0c44e3f8eee') {
    throw 'upstream frozen cell invalid'
  }
  foreach ($stage in @('temporal_bn','spatial_elu_pool1','depth_point_elu_pool2','embedding_64d','classifier_input')) {
    if (-not (Test-Path -LiteralPath (Join-Path $couplingRuntime "stage_cache\EEGNet\OpenBMI_MI\fold1\$stage.json"))) {
      throw "frozen stage cache absent: $stage"
    }
  }
}

try {
  if (Test-Path -LiteralPath $log) { throw 'queue log already exists; refusing duplicate task' }
  if ((Test-Path -LiteralPath $manifest) -or (Test-Path -LiteralPath $failure) -or
      (Test-Path -LiteralPath (Join-Path $cell 'mediation_v2'))) {
    throw 'target mediation evidence already exists; refusing duplicate'
  }
  Confirm-FrozenInputs
  "MEDIATION_F1_QUEUE_START utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $log -Encoding utf8
  for ($check = 1; $check -le 1440; $check++) {
    if ((Test-Path -LiteralPath $manifest) -or (Test-Path -LiteralPath $failure) -or
        (Test-Path -LiteralPath (Join-Path $cell 'mediation_v2'))) {
      throw 'target evidence appeared while waiting; refusing duplicate'
    }
    $otherTask = @(Get-ScheduledTask | Where-Object {
      $_.TaskName -like 'PERSIST_EEG_PC_MECHANISM_CELL_*' -and
      $_.TaskName -ne $taskName -and $_.State -eq 'Running'
    })
    $otherProcess = @(Get-CimInstance Win32_Process | Where-Object {
      $_.Name -match '^python(\.exe)?$' -and $_.CommandLine -and
      $_.CommandLine -match 'persist_eeg_pc_mechanism_closure_seed0_v1|pc_mechanism_closure_runtime'
    })
    $freeGb = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
    $gpuLine = [string]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) | Select-Object -First 1)
    $gpuMiB = 0
    if (-not [int]::TryParse($gpuLine.Trim(), [ref]$gpuMiB)) { throw 'cannot read GPU free memory' }
    if ($check -eq 1 -or $check % 10 -eq 0) {
      "MEDIATION_F1_QUEUE_WAIT check=$check free_gb=$([Math]::Round($freeGb,2)) gpu_mib=$gpuMiB other_tasks=$($otherTask.Count) other_processes=$($otherProcess.Count) utc=$([DateTime]::UtcNow.ToString('o'))" |
        Out-File -LiteralPath $log -Append -Encoding utf8
    }
    if ($freeGb -ge 40 -and $gpuMiB -ge 16000 -and $otherTask.Count -eq 0 -and $otherProcess.Count -eq 0) {
      Confirm-FrozenInputs
      "MEDIATION_F1_CELL_START free_gb=$([Math]::Round($freeGb,2)) gpu_mib=$gpuMiB utc=$([DateTime]::UtcNow.ToString('o'))" |
        Out-File -LiteralPath $log -Append -Encoding utf8
      $command = '"' + $python + '" -u "' + $entry + '" --model EEGNet --task OpenBMI_MI --fold 1 >> "' + $log + '" 2>&1'
      & cmd.exe /d /c $command
      if ($LASTEXITCODE -ne 0) { throw "mediation process exited $LASTEXITCODE" }
      $done = Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
      if ($done.status -ne 'MEDIATION_PHASE_COMPLETE_PENDING_OTHER_REQUIRED_ANALYSES' -or
          $done.model -ne 'EEGNet' -or $done.task -ne 'OpenBMI_MI' -or $done.fold -ne 1 -or
          $done.stage_count -ne 5 -or $done.final_heldout_accessed -ne $false) {
        throw 'mediation terminal manifest invalid'
      }
      "MEDIATION_F1_CELL_END exit=0 utc=$([DateTime]::UtcNow.ToString('o'))" |
        Out-File -LiteralPath $log -Append -Encoding utf8
      exit 0
    }
    Start-Sleep -Seconds 60
  }
  "MEDIATION_F1_QUEUE_RESOURCE_TIMEOUT utc=$([DateTime]::UtcNow.ToString('o'))" |
    Out-File -LiteralPath $log -Append -Encoding utf8
  exit 2
} catch {
  if (Test-Path -LiteralPath $log) {
    "MEDIATION_F1_QUEUE_EXCEPTION $_ utc=$([DateTime]::UtcNow.ToString('o'))" |
      Out-File -LiteralPath $log -Append -Encoding utf8
  }
  exit 1
}
