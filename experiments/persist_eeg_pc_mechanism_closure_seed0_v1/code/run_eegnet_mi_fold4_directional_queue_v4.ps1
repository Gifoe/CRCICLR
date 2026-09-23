$ErrorActionPreference = 'Stop'
$repo = 'D:\nips-temp\TotalP\P1\CRCICLR_PERSIST_INCREMENTAL_VALUE_V1'
$experiment = Join-Path $repo 'experiments\persist_eeg_pc_mechanism_closure_seed0_v1'
$runtime = 'D:\nips-temp\TotalP\P1\pc_mechanism_closure_runtime'
$couplingRuntime = 'D:\nips-temp\TotalP\P1\protected_complement_coupling_runtime'
$taskName = 'PERSIST_EEG_PC_MECHANISM_CELL_EEGNET_MI_F4_DIRECTIONAL_QUEUE_V4'
$entry = Join-Path $experiment 'code\run_directional_cell_v4.py'
$core = Join-Path $experiment 'code\directional_core_v3.py'
$factor = Join-Path $experiment 'code\directional_factorized_eegnet_v4.py'
$sampling = Join-Path $experiment 'code\sampling_core.py'
$cellData = Join-Path $experiment 'code\cell_data_v1.py'
$cell = Join-Path $runtime 'cells\eegnet\openbmi_mi\fold4_seed0'
$parts = Join-Path $cell 'directional_v4'
$mediation = Join-Path $cell 'MEDIATION_V2.json'
$manifest = Join-Path $cell 'DIRECTIONAL_V4.json'
$failure = Join-Path $cell 'DIRECTIONAL_V4_FAIL_CLOSED.json'
$log = Join-Path $runtime 'scheduled_directional_cell_EEGNet_OpenBMI_MI_f4_queue_v4.log'
$python = 'E:\Anaconda\envs\persist_stable_251\python.exe'
$env:PERSIST_SOURCE_REPO = $repo
$env:COUPLING_RUNTIME = $couplingRuntime
$env:MECHANISM_RUNTIME = $runtime
$env:PYTHONUNBUFFERED = '1'

function Confirm-Inputs {
  $expected = @{
    $entry = '7c77da55f753d372e394c44e9d851e4f2390a42ebc549993a69e176e7ce858bc';
    $core = '5f50f0f649039f2cb83c2b951952e04a645908bf5328bfc277b79948e05608af';
    $factor = '03c1f0262885546ea70ebfc7db93471968f9952e2aac62097d13b4317ff91729';
    $sampling = '7b8f5b94caca3924de3c714a780987a52337359963f38d5135c78410c8b1a631';
    $cellData = 'c5e4f90865445bcc1ec313b1ca5b1970cf484a9ef7f44e8aa1bf38a0fd721416';
    (Join-Path $experiment 'protocol\MECHANISM_CLOSURE_ANALYSIS_LOCK.md') = 'ac12d33f7d30b2c846ec4c4fb07f8513bf25e372943a767b134883bf095a7c88';
    (Join-Path $experiment 'protocol\FINAL_PROJECTOR_AMENDMENT.md') = 'aec1e604d38f1901bd9eca7e072e30a3d496a28061a726ebbe63133259b0e474'
  }
  foreach ($path in $expected.Keys) {
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected[$path]) { throw "source/lock SHA mismatch: $path" }
  }
  $gate = Get-Content -LiteralPath (Join-Path $experiment 'gate\GATE_AUDIT.json') -Raw | ConvertFrom-Json
  if ($gate.status -ne 'PASSED' -or $gate.original_cells_complete -ne 20 -or
      $gate.layer_stage_cells_reconstructed -ne 120 -or $gate.final_heldout_accessed -ne $false) { throw 'upstream gate invalid' }
  if (Test-Path -LiteralPath $mediation) {
    $m = Get-Content -LiteralPath $mediation -Raw | ConvertFrom-Json
    if ($m.status -ne 'MEDIATION_PHASE_COMPLETE_PENDING_OTHER_REQUIRED_ANALYSES' -or
        $m.model -ne 'EEGNet' -or $m.task -ne 'OpenBMI_MI' -or $m.fold -ne 4 -or
        $m.stage_count -ne 4 -or $m.final_heldout_accessed -ne $false) { throw 'fold4 mediation manifest invalid' }
  }
}

function Get-Resource {
  $freeGb = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
  $gpuLine = [string]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) | Select-Object -First 1)
  $gpuMiB = 0
  if (-not [int]::TryParse($gpuLine.Trim(), [ref]$gpuMiB)) { throw 'GPU resource query failed' }
  return @{ FreeGb = $freeGb; GpuMiB = $gpuMiB }
}

function Other-Cell-Active {
  $otherTask = @(Get-ScheduledTask | Where-Object {
    $_.TaskName -like 'PERSIST_EEG_PC_MECHANISM_CELL_*' -and $_.TaskName -ne $taskName -and $_.State -eq 'Running'
  })
  $otherProcess = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(\.exe)?$' -and $_.CommandLine -and
    $_.CommandLine -match 'persist_eeg_pc_mechanism_closure_seed0_v1|pc_mechanism_closure_runtime'
  })
  return ($otherTask.Count -gt 0 -or $otherProcess.Count -gt 0)
}

function Progress-Count {
  if (-not (Test-Path -LiteralPath $parts)) { return 0 }
  $count = 0
  foreach ($file in (Get-ChildItem -LiteralPath $parts -Filter '*.partial.json' -File)) {
    $partial = Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json
    if ($partial.status -ne 'PARTIAL_RESOURCE_SAFE' -or $partial.final_heldout_accessed -ne $false) { throw "invalid directional partial: $($file.Name)" }
    $count += @($partial.random_controls).Count
  }
  $count += 100 * @(Get-ChildItem -LiteralPath $parts -Filter '*.json' -File | Where-Object { $_.Name -notlike '*.partial.json' }).Count
  return $count
}

try {
  if (Test-Path -LiteralPath $log) { throw 'directional queue log already exists' }
  if ((Test-Path -LiteralPath $manifest) -or (Test-Path -LiteralPath $failure)) { throw 'directional terminal evidence already exists' }
  Confirm-Inputs
  "DIRECTIONAL_F4_QUEUE_START utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $log -Encoding utf8
  for ($attempt = 1; $attempt -le 100; $attempt++) {
    $waits = 0
    while ($true) {
      Confirm-Inputs
      $resource = Get-Resource
      $blocked = Other-Cell-Active
      if ((Test-Path -LiteralPath $mediation) -and $resource.FreeGb -ge 40 -and $resource.GpuMiB -ge 16000 -and -not $blocked) { break }
      $waits++
      if ($waits -eq 1 -or $waits % 10 -eq 0) {
        "DIRECTIONAL_F4_WAIT attempt=$attempt mediation=$([bool](Test-Path -LiteralPath $mediation)) free_gb=$([Math]::Round($resource.FreeGb,2)) gpu_mib=$($resource.GpuMiB) blocked=$blocked utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $log -Append -Encoding utf8
      }
      Start-Sleep -Seconds 30
    }
    Confirm-Inputs
    $before = Progress-Count
    $stdout = Join-Path $runtime "scheduled_directional_cell_EEGNet_OpenBMI_MI_f4_queue_v4_attempt${attempt}.stdout.log"
    $stderr = Join-Path $runtime "scheduled_directional_cell_EEGNet_OpenBMI_MI_f4_queue_v4_attempt${attempt}.stderr.log"
    if ((Test-Path -LiteralPath $stdout) -or (Test-Path -LiteralPath $stderr)) { throw 'attempt log already exists' }
    "DIRECTIONAL_F4_ATTEMPT_START attempt=$attempt free_gb=$([Math]::Round($resource.FreeGb,2)) gpu_mib=$($resource.GpuMiB) utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $log -Append -Encoding utf8
    $child = Start-Process -FilePath $python -ArgumentList @('-u', $entry, '--model', 'EEGNet', '--task', 'OpenBMI_MI', '--fold', '4') -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $safetyStop = $false
    while ($true) {
      $child.Refresh()
      if ($child.HasExited) { break }
      $resource = Get-Resource
      if ($resource.FreeGb -lt 16 -or $resource.GpuMiB -lt 1000) {
        $owned = Get-CimInstance Win32_Process -Filter "ProcessId=$($child.Id)"
        if ($owned -and $owned.Name -match '^python(\.exe)?$' -and $owned.CommandLine -match 'run_directional_cell_v4.py' -and
            $owned.CommandLine -match 'OpenBMI_MI' -and $owned.CommandLine -match '--fold 4') {
          Stop-Process -Id $child.Id -Force -ErrorAction Stop
          $safetyStop = $true
          "DIRECTIONAL_F4_SAFETY_STOP attempt=$attempt pid=$($child.Id) free_gb=$([Math]::Round($resource.FreeGb,2)) gpu_mib=$($resource.GpuMiB) utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $log -Append -Encoding utf8
          break
        }
      }
      Start-Sleep -Seconds 5
    }
    $child.WaitForExit()
    if (Test-Path -LiteralPath $manifest) {
      $done = Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
      if ($done.status -ne 'DIRECTIONAL_PHASE_COMPLETE_PENDING_OTHER_REQUIRED_ANALYSES' -or
          $done.model -ne 'EEGNet' -or $done.task -ne 'OpenBMI_MI' -or $done.fold -ne 4 -or
          $done.stage_count -ne 5 -or $done.random_draws_per_stage -ne 20 -or $done.final_heldout_accessed -ne $false) { throw 'directional terminal manifest invalid' }
      "DIRECTIONAL_F4_COMPLETE attempt=$attempt utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $log -Append -Encoding utf8
      exit 0
    }
    if (Test-Path -LiteralPath $failure) { throw 'directional phase failed closed; do not retry' }
    $after = Progress-Count
    if (-not $safetyStop -and $after -le $before) { throw "directional process made no atomic progress: before=$before after=$after" }
    "DIRECTIONAL_F4_RECYCLE attempt=$attempt safety_stop=$safetyStop progress_before=$before progress_after=$after utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $log -Append -Encoding utf8
  }
  throw 'directional phase exceeded 100 attempts'
} catch {
  if (Test-Path -LiteralPath $log) { "DIRECTIONAL_F4_QUEUE_EXCEPTION $_ utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $log -Append -Encoding utf8 }
  exit 1
}
