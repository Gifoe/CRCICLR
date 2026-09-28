$ErrorActionPreference = 'Stop'
$root = 'D:\nips-temp\TotalP\P1\context_relative_representation_runtime'
$source = Join-Path $root 'source\code'
$python = 'E:\Anaconda\envs\persist_stable_251\python.exe'
$stdout = Join-Path $root 'logs\finalize_v1.stdout.log'
$stderr = Join-Path $root 'logs\finalize_v1.stderr.log'
$status = Join-Path $root 'logs\finalize_v1.status.log'

foreach ($p in @($stdout, $stderr, $status, (Join-Path $root 'final_outputs'), (Join-Path $root 'final_protocol'))) {
  if (Test-Path -LiteralPath $p) { throw "output or log already exists: $p" }
}
& (Join-Path $source 'verify_train_lock_v1.ps1') -Root $root | Out-Null
foreach ($fold in 0..4) {
  foreach ($phase in @('geometry', 'outer', 'localization')) {
    $audit = Join-Path $root "$phase\fold$fold\AUDIT.json"
    if (-not (Test-Path -LiteralPath $audit)) { throw "missing input audit: $audit" }
  }
}

$env:REL_RUNTIME = $root
"START utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $status -Encoding utf8
$script = Join-Path $source 'finalize_v2.py'
$process = Start-Process -FilePath $python -ArgumentList @('-u', ('"' + $script + '"')) -Wait -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
"EXIT=$($process.ExitCode) utc=$([DateTime]::UtcNow.ToString('o'))" | Out-File -LiteralPath $status -Append -Encoding utf8
if ($process.ExitCode -ne 0) { throw "finalizer failed exit=$($process.ExitCode) stderr=$stderr" }
