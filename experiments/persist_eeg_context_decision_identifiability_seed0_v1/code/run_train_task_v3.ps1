param([Parameter(Mandatory=$true)][ValidateRange(0,4)][int]$Fold)
$ErrorActionPreference = 'Stop'
$runtime = 'D:\nips-temp\TotalP\P1\pc_context_decision_identifiability_runtime'
$python = 'E:\Anaconda\envs\persist_stable_251\python.exe'
$script = Join-Path $runtime 'code\run_v2.py'
$env:CID_RUNTIME = $runtime
$env:CID_UPSTREAM_TRAIN = 'D:\nips-temp\TotalP\P1\pc_differential_adaptation_necessity_runtime\train'
$stdout = Join-Path $runtime ("train_fold{0}_v3.stdout.log" -f $Fold)
$stderr = Join-Path $runtime ("train_fold{0}_v3.stderr.log" -f $Fold)
$exit = Join-Path $runtime ("train_fold{0}_v3.exit" -f $Fold)
if ((Test-Path -LiteralPath $stdout) -or (Test-Path -LiteralPath $stderr) -or
    (Test-Path -LiteralPath $exit) -or
    (Test-Path -LiteralPath (Join-Path $runtime ("fold{0}\TRAIN_SEAL.json" -f $Fold)))) {
    throw 'Existing train attempt; refuse duplicate'
}
$p = Start-Process -FilePath $python -ArgumentList @('-u', ('"'+$script+'"'), 'train', '--fold', "$Fold") -Wait -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
"EXIT $($p.ExitCode) $([DateTime]::UtcNow.ToString('o'))" | Set-Content -LiteralPath $exit -Encoding UTF8
if ($p.ExitCode -ne 0) { throw "Train analysis failed: $($p.ExitCode)" }
