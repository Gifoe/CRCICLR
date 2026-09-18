param(
    [ValidateRange(1, 2)]
    [int]$Workers = 2
)

$ErrorActionPreference = 'Stop'
$base = 'D:\chenyu-iclr\BASELINE'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$log = Join-Path $base "outputs\\cpu_exploratory_sequence_${stamp}.log"
$err = Join-Path $base "outputs\\cpu_exploratory_sequence_${stamp}.err.log"
$script = Join-Path $base 'code\\run_cpu_exploratory_sequence.ps1'
$args = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $script, '-Workers', "$Workers")
$process = Start-Process -FilePath (Get-Command powershell).Source -ArgumentList $args -RedirectStandardOutput $log -RedirectStandardError $err -PassThru -WindowStyle Hidden
[pscustomobject]@{ pid = $process.Id; log = $log; error_log = $err; workers = $Workers } | ConvertTo-Json -Compress
