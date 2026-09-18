param(
  [ValidateSet('all', 'peeh', 'pswa', 'aggregate')]
  [string]$Mode = 'all',
  [ValidateRange(1, 4)]
  [int]$Workers = 3
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Python = 'E:\Anaconda\envs\persist_stable_251\python.exe'
$PEEH = Join-Path $PSScriptRoot 'run_eegconformer_fbcnet_peeh.py'
$PSWA = Join-Path $PSScriptRoot 'run_eegconformer_fbcnet_pswa.py'
$Runtime = 'D:\nips-temp\TotalP\P1\eegconformer_fbcnet_peeh_runtime'
$PswaRuntime = 'D:\nips-temp\TotalP\P1\eegconformer_fbcnet_pswa_runtime'
$LogRoot = Join-Path $Runtime 'logs'

# No numerical knob is changed.  Independent fold/model processes supply the
# parallelism; limiting BLAS threads prevents three concurrent cells from
# oversubscribing the host into a slower all-core thrash.
$env:OMP_NUM_THREADS = '12'
$env:MKL_NUM_THREADS = '12'
$env:OPENBLAS_NUM_THREADS = '12'
$env:NUMEXPR_NUM_THREADS = '12'
$env:PEEH_RUNTIME = $Runtime
$env:PSWA_RUNTIME = $PswaRuntime
$env:TRUE_OUTER_WBCIC_CACHE = 'D:\nips-temp\TotalP\P2\wbcic_outer_cache\wbcic_epochss'
$env:FULL_OPENBMI_CACHE = 'D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full\outputs\persist_eeg_stage0\cache\openbmi'
$env:FULL_WBCIC_CACHE = 'D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC\experiments\persist_eeg_wbcic_independent_replication_v1\runtime\cache\wbcic_epochs'

New-Item -ItemType Directory -Force -Path $LogRoot, $Runtime, $PswaRuntime | Out-Null

function Invoke-Queue {
  param([string]$Stage, [string]$Script, [string]$DoneRoot)
  $models = @('EEGConformer', 'FBCNet')
  $tasks = @('OpenBMI_MI', 'OpenBMI_ERP', 'OpenBMI_SSVEP', 'WBCIC_MI')
  $jobs = foreach($model in $models) { foreach($task in $tasks) { foreach($fold in 0..4) {
    $done = Join-Path $DoneRoot ("cells\{0}\{1}\fold{2}_seed0.json" -f $model.ToLower(), $task.ToLower(), $fold)
    if(-not (Test-Path -LiteralPath $done)) {
      [PSCustomObject]@{Model=$model; Task=$task; Fold=$fold; Done=$done}
    }
  } } }
  Write-Output ("{0}_PENDING={1}" -f $Stage, @($jobs).Count)
  $running = @()
  foreach($job in $jobs) {
    while(@($running).Count -ge $Workers) {
      $finished = $running | Where-Object { $_.Process.HasExited }
      if(@($finished).Count -eq 0) { Start-Sleep -Seconds 2; continue }
      foreach($item in $finished) {
        $item.Process.Refresh()
        if($item.Process.ExitCode -ne 0) {
          throw ("{0} failed: {1}/{2}/fold{3}; see {4}" -f $Stage,$item.Model,$item.Task,$item.Fold,$item.Log)
        }
        $running = @($running | Where-Object { $_ -ne $item })
      }
    }
    $log = Join-Path $LogRoot ("{0}_{1}_{2}_fold{3}.log" -f $Stage,$job.Model,$job.Task,$job.Fold)
    $args = @($Script, '--stage', 'run', '--model', $job.Model, '--task', $job.Task, '--fold', [string]$job.Fold)
    $p = Start-Process -FilePath $Python -ArgumentList $args -RedirectStandardOutput $log -RedirectStandardError ($log + '.err') -PassThru -WindowStyle Hidden
    $running += [PSCustomObject]@{Process=$p; Model=$job.Model; Task=$job.Task; Fold=$job.Fold; Log=$log}
    Write-Output ("START {0} {1} {2} fold={3} pid={4}" -f $Stage,$job.Model,$job.Task,$job.Fold,$p.Id)
  }
  while(@($running).Count) {
    $finished = $running | Where-Object { $_.Process.HasExited }
    if(@($finished).Count -eq 0) { Start-Sleep -Seconds 2; continue }
    foreach($item in $finished) {
      $item.Process.Refresh()
      if($item.Process.ExitCode -ne 0) {
        throw ("{0} failed: {1}/{2}/fold{3}; see {4}" -f $Stage,$item.Model,$item.Task,$item.Fold,$item.Log)
      }
      Write-Output ("DONE {0} {1} {2} fold={3}" -f $Stage,$item.Model,$item.Task,$item.Fold)
      $running = @($running | Where-Object { $_ -ne $item })
    }
  }
}

if($Mode -in @('all','peeh')) {
  & $Python $PEEH --stage prelock
  if($LASTEXITCODE -ne 0) { throw 'PEEH prelock failed' }
  Invoke-Queue -Stage 'PEEH' -Script $PEEH -DoneRoot $Runtime
}
if($Mode -in @('all','pswa')) {
  Invoke-Queue -Stage 'PSWA' -Script $PSWA -DoneRoot $PswaRuntime
}
if($Mode -in @('all','aggregate')) {
  & $Python $PEEH --stage aggregate
  if($LASTEXITCODE -ne 0) { throw 'PEEH aggregate failed' }
  & $Python $PSWA --stage aggregate
  if($LASTEXITCODE -ne 0) { throw 'PSWA aggregate failed' }
}
